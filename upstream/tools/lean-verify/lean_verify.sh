#!/bin/bash -p
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md; lean/TERMS.md.
# Runs the Lean verifier's executable form on one proof, inside Lean, and prints its verdict.
# What that verdict is worth, what is trusted, and what it costs: tools/lean-verify/README.md.
#
# usage: tools/lean-verify/lean_verify.sh [--no-native] CONFIG_DIR PUBLIC_KEY_DIR REQUEST_FILE TOKEN_FILE PROOF_FILE ID_FILE
#   The first five are the arguments of the same names of cp_verify (AGENTS.md). ID_FILE holds the
#   commitment ID: 64 hex digits in lower case, then at most a newline.
#   --no-native: Lean interprets all the code (hours); no helper library is built or loaded.
# Before the first call: build what this command asks for (README.md, step 1). That takes hours
# of processor time and much memory; this script does not do it.
# Needs: bash, GNU coreutils, diffutils, grep and sed, util-linux's prlimit, python3, lake on the
# PATH, and what tools/parser/parse_files.sh needs.
# The script itself removes nothing. It writes below a fresh directory that mktemp makes, which its
# first line names: delete it when you are done. Lake, which it calls, writes below lean/.lake and
# can replace or delete files there (lean/CHECKING.md, "Before you start"); the compilers that the
# parser and Lake start lay and remove temporary files of their own where TMPDIR says.
# Run it as the account that built lean/.lake, not with sudo. Run one call at a time.
#
# Standard output holds one line; all else goes to the error stream. The answer is the exit code and
# that line alone: Lean's own lines, which the error stream shows behind "lean: ", are no verdict.
# Exit code 0: accepted. The line is "VERDICT: ACCEPT (exact match)" where the parameter file's lines
# C, theta and cap read 256, 256 and 0 (and sh and sT 8 and 4); else it is "VERDICT: ACCEPT (under the
# files' policy C=.. theta=.. cap=..; NOT exact match: see "What an accept means")": that is a heading
# of README.md.
# 1 and "VERDICT: REJECT": rejected. 2 and "NO VERDICT": wrong arguments, a file that is missing, a
# claim that fails the tests of docs/format/FORMAT.md, 12.2, a tool or a build that failed, a signal,
# a run that did not end as it should; the error stream says which.
# After any code other than 0, and after an end by a signal, treat the reply as unproved.
#
# "-p" in the first line: bash then takes no functions and no start-up file from the environment.
set -u
umask go-w # mktemp's directory is the caller's alone whatever the mask; Lake's files keep the tree's modes
ulimit -c 0 # a panic aborts Lean (below): no core file, which would be laid into lean/
LC_ALL=C
export LC_ALL
unset POSIXLY_CORRECT # with it set, GNU grep takes the "--" behind a pattern for a file's name
# The reason goes to the error stream: it may hold a file's name, and no text of the caller's may
# stand on standard output, where it could pose as a verdict.
none() {
  printf 'lean_verify.sh: no verdict: %s\n' "$*" >&2
  echo "NO VERDICT"
  exit 2
}
usage() {
  echo "usage: tools/lean-verify/lean_verify.sh [--no-native] CONFIG_DIR PUBLIC_KEY_DIR REQUEST_FILE TOKEN_FILE PROOF_FILE ID_FILE" >&2
  none "wrong arguments: see the usage line above"
}
# <file> <command ..>: runs the command, its two streams into the file, and waits. "timeout" puts it
# into a process group of its own (0: no time limit) and hands a TERM on to that group, but for the
# instant of its own start (README.md, on a stray process): so the trap below reaches Lake, Lean and
# the compilers, which would else run on unread for up to half an hour. (The streams are turned here,
# not at the call: the trap runs inside "held", and its "NO VERDICT" must reach standard output.)
held() {
  local log=$1
  shift
  timeout 0 "$@" < /dev/null > "$log" 2>&1 &
  wait "$!"
}
# The jobs are signalled as jobs, not by number: bash sends nothing to a job whose process it has
# already collected, while a number may by then be another process's. The trap's first words: from there
# on a further signal is passed over, so that it starts no second trap inside the first.
# shellcheck disable=SC2154
trap 'trap "" INT TERM HUP
  for j in $(jobs -r | sed -n "s/^\[\([0-9]*\)\].*/\1/p"); do kill -TERM "%$j" 2> /dev/null; done
  none "a signal stopped this script"' INT TERM HUP
native=yes
if [ "${1:-}" = --no-native ]; then
  native=no
  shift
fi
[ $# -eq 6 ] || usage
# Looked for up front: further down a missing tool would be told as another fault.
for t in lake python3 timeout prlimit tail sed grep cmp realpath dirname stat mktemp sha256sum wc cp mkdir; do
  command -v "$t" > /dev/null || none "$t is not on the PATH"
done
# "$(realpath ..)" below drops newlines at a name's end, and would then open another file than the
# one that was named.
for a in "$@"; do
  case $a in *$'\n'*) none "an argument holds a newline" ;; esac
done
unset CDPATH # with it set, a "cd" below could enter another directory than the one named
# Lake puts the tree's own compiled files on these paths; nothing of the caller's is to stand there.
unset LEAN_PATH LEAN_SRC_PATH
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P) || none "the script's own directory cannot be entered"
# "cd" cuts "‹name›/.." away as text, while the system follows a link at ‹name›: called by such a path,
# this file could be one tree's and "here" another's.
[ "$here/lean_verify.sh" -ef "${BASH_SOURCE[0]}" ] ||
  none "this script was called through a symbolic link, or by a path in which \"..\" follows one: call it by its own path, without \"..\""
tree=$(cd -- "$here/../.." && pwd -P) || none "the repository's top directory cannot be entered"
# "$(..)" drops newlines at a name's end here too: what is worked in must be what lies above.
[ "$tree/tools/lean-verify" -ef "$here" ] || none "the repository's top directory is not the one above this script"
# The path is printed below: behind a newline in it, a folder's name could pose as a line of this script's.
case $tree in *$'\n'*) none "the repository's path holds a newline" ;; esac
{ config=$(realpath -e -- "$1") && keys=$(realpath -e -- "$2") && request=$(realpath -e -- "$3") &&
  token=$(realpath -e -- "$4") && proof=$(realpath -e -- "$5") && id=$(realpath -e -- "$6"); } ||
  none "one of the six arguments does not exist: the line above says which"
# .. and where an argument is a link to a name that ends in a newline.
{ [ "$config" -ef "$1" ] && [ "$keys" -ef "$2" ] && [ "$request" -ef "$3" ] && [ "$token" -ef "$4" ] &&
  [ "$proof" -ef "$5" ] && [ "$id" -ef "$6" ]; } || none "an argument leads to a name that ends in a newline"
for a in "$config" "$keys" "$request" "$token" "$proof" "$id"; do
  case $a in *$'\n'*) none "an argument leads to a name that holds a newline" ;; esac
done
{ [ -d "$config" ] && [ -d "$keys" ]; } || none "CONFIG_DIR and PUBLIC_KEY_DIR must be directories"
for f in "$config/policy.txt" "$config/layout.txt" "$config/KL.bin" "$request" "$token" "$proof" "$id"; do
  { [ -f "$f" ] && [ -r "$f" ] && [ "$(stat -L -c %s -- "$f")" -le 268435456 ]; } ||
    none "not a regular file that can be read, of at most 2^28 bytes: $f"
done
ask=$here/LeanVerify.lean list=$here/native-modules.txt reader=$here/read_claim.py
{ [ -f "$ask" ] && [ -f "$list" ] && [ -f "$reader" ]; } ||
  none "LeanVerify.lean, native-modules.txt or read_claim.py is missing beside this script"
# Lake replaces and can delete files below lean/.lake: through a link that would reach outside the
# tree.
for d in "$tree/lean" "$tree/lean/.lake" "$tree/lean/.lake/build" "$tree/lean/.lake/packages" "$tree"/lean/.lake/packages/* \
  "$tree"/lean/.lake/packages/*/.lake "$tree"/lean/.lake/packages/*/.lake/build; do
  case $d in *$'\n'*) none "a name in $tree/lean/.lake/packages holds a newline" ;; esac
  [ ! -L "$d" ] ||
    none "$d is a symbolic link: refused, since Lake could write through it: see lean/CHECKING.md, \"Before you start\""
done
# Without the packages' folder Lake would go to the network. (That it is whole is not looked at.)
[ -d "$tree/lean/.lake/packages" ] || none "$tree/lean/.lake/packages is missing: see lean/CHECKING.md, step 3"
[ -f "$tree/lean/.lake/build/lib/lean/Johnson/PVR/ProgramGuard.olean" ] ||
  none "the library is not built in $tree/lean: see tools/lean-verify/README.md, step 1"
# Under sudo Lake would leave root's files in the caller's tree, which then break the caller's
# builds.
[ -O "$tree/lean/.lake" ] ||
  none "$tree/lean/.lake belongs to another account: run this as that account, not with sudo"
work=$(mktemp -d) || none "mktemp -d failed"
# Absolute and without links, as the parser prints it below.
work=$(cd -- "$work" && pwd -P) || none "the work directory cannot be entered"
{ [ -n "$work" ] && [ -d "$work" ]; } || none "mktemp -d failed"
case $work in *$'\n'*) none "the work directory's path holds a newline: set TMPDIR to another place" ;; esac
echo "work directory: $work" >&2
echo "tree: $tree" >&2 # where Lake will work: the tree that really holds this script's folder
cd -- "$tree/lean" || none "lean/ cannot be entered"
# Without it a panic in Lean's code could count as a reject, and a build would not fail at one.
export LEAN_ABORT_ON_PANIC=1

# The claim and the public files are copied, first of all. The second reader, the parser and Lean
# read these copies: what is judged half an hour later is what lies here now. (The proof file and the
# key folder are not copied: the parser reads them in place, and soundness does not rest on it.)
{ mkdir -- "$work/claim" "$work/claim/config" &&
  cp -- "$config/policy.txt" "$config/layout.txt" "$config/KL.bin" "$work/claim/config/" &&
  cp -- "$id" "$work/claim/ID.txt" && cp -- "$request" "$work/claim/example.req" &&
  cp -- "$token" "$work/claim/example.y"; } || none "the copies below $work could not be made"
echo "what is judged (SHA-256), below $work/claim:" >&2
(cd -- "$work/claim" && sha256sum ID.txt example.req example.y config/policy.txt config/layout.txt config/KL.bin) >&2 ||
  none "the copies below $work could not be read"
# No theorem is about the three Lean functions that cut the claim out of the caller's files
# (lean/GAPS.md, 37). A second reader, written from the format page alone (docs/format/FORMAT.md,
# 12.2), reads the claim here; below, the lines in which Lean prints the claim must be its lines.
# "-I": the caller's PYTHON* variables and the script's directory do not reach it.
python3 -I -B "$reader" "$work/claim/config" "$work/claim/example.req" "$work/claim/example.y" "$work/claim/ID.txt" \
  > "$work/claim.txt" ||
  none "read_claim.py refuses the request, token, ID or parameter file: the line above says why (its tests: docs/format/FORMAT.md, 12.2, and its own head)"
[ "$(wc -l < "$work/claim.txt")" = 6 ] || none "read_claim.py did not print its six lines"
# Shown now, so that a caller need not wait for Lean to see which claim is judged.
sed 's/^/read_claim.py: /' -- "$work/claim.txt" >&2
public="the three public files are those of examples/full-size/config"
for f in policy.txt layout.txt KL.bin; do
  cmp -s -- "$work/claim/config/$f" "$tree/examples/full-size/config/$f" ||
    public="NOTE: the three public files are not those of examples/full-size/config: see tools/lean-verify/README.md, \"What an accept means\""
done
echo "$public" >&2
# The soundness theorem on "verify" has exact match as a premise on the parameter file, and "verify" has no
# test for it in its own text: in that file it is numbers. They are taken from the second reader's
# sixth line. A verdict is given only where Lean's sixth line is the same, byte for byte (below): so
# the numbers in an accept's line are those that Lean printed too.
# <name>: the number of that name in that line.
number() { sed -n "s/^public files: .*[[ ]$1 \([0-9][0-9]*\)[],].*/\1/p" "$work/claim.txt"; }
C=$(number C) theta=$(number theta) cap=$(number cap) sh=$(number sh) sT=$(number sT)
{ [ -n "$C" ] && [ -n "$theta" ] && [ -n "$cap" ] && [ -n "$sh" ] && [ -n "$sT" ]; } ||
  none "read_claim.py's sixth line lacks C, theta, cap, sh or sT"
# The format page's exact match has sh and sT in it too (docs/format/FORMAT.md, 12.5): a second lock.
# The words "exact match" are kept for an accept's line: the line "policy" is printed at every end, also
# for files that "verify" then refuses as out of its scope.
if [ "$C $theta $cap $sh $sT" = "256 256 0 8 4" ]; then
  how="exact match"
  policy="the policy lines (C, theta, cap, sh, sT) read (256, 256, 0, 8, 4)"
else
  how="under the files' policy C=$C theta=$theta cap=$cap; NOT exact match: see \"What an accept means\""
  policy="NOTE: the policy lines (C, theta, cap, sh, sT) read ($C, $theta, $cap, $sh, $sT), not (256, 256, 0, 8, 4): see tools/lean-verify/README.md, \"What an accept means\""
fi
echo "$policy" >&2

# Lake looks whether the compiled library still fits the sources: beneath LeanVerify.lean's one
# import, Johnson.PVR.Program, and beneath Johnson.PVR.ProgramGuard, which holds the library's own
# checks of that module and imports the file of the theorems on it. So there is no verdict where
# these have not been compiled. Lake builds nothing here (beside a target that is out of date it may
# lay a small file that says so): building the library takes hours of processor time and much memory,
# and is the reader's to start (README.md, step 1).
echo "asking Lake whether the compiled library fits the sources .." >&2
held "$work/fits.log" lake build --no-build Johnson.PVR.Program Johnson.PVR.ProgramGuard ||
  none "the compiled Lean library lacks files or is out of date: build it (tools/lean-verify/README.md, step 1); Lake's output: $work/fits.log"

# The parser comes before the helper library and Lean. It refuses a work directory that another
# account could swap: so the helper library is written, and Lean reads the copies, only at a place
# that has passed that test.
echo "parsing the prover's three files (minutes) .." >&2
held "$work/parser.out" "$tree/tools/parser/parse_files.sh" "$work/claim/config" "$keys" "$work/claim/example.req" "$proof" "$work/parser" ||
  none "the parser ended with code $?: see $work/parser.out"
tail -n 1 -- "$work/parser.out" | cmp -s -- - <(printf 'PARSED AND READ BACK: %s\n' "$work/parser/parsed") ||
  none "the parser's last line is not the expected one: see $work/parser.out"
leaves=("$work"/parser/parsed/leaf-*.parsed.bin)
{ [ ${#leaves[@]} -eq 1 ] && [ -f "${leaves[0]}" ]; } ||
  none "the parser did not leave exactly one parsed leaf file below $work/parser/parsed: see $work/parser.out"

# For the helper library Lake compiles the listed modules, which all lie beneath
# Johnson.PVR.Program, to native objects.
targets=() modules=() flags=() long="hours: Lean interprets all the code"
if [ "$native" = yes ]; then
  while IFS= read -r m; do
    case $m in
      '' | '#'*) continue ;;
      *[!A-Za-z0-9_.\']*) none "native-modules.txt has a line that is not a module name" ;;
    esac
    modules+=("$m") targets+=("+$m:c.o.export")
  done < "$list"
  [ ${#modules[@]} -gt 0 ] || none "native-modules.txt names no module"
  # Lake makes an object from the module's .c file, which lies beside the compiled library. Where one
  # is missing Lake would compile the module's Lean source anew: for the packages that is a long
  # build, which is the reader's to start, not a call's.
  for m in "${modules[@]}"; do
    p=${m//.//} c=''
    for b in .lake/build .lake/packages/*/.lake/build; do
      if [ -f "$b/ir/$p.c" ]; then
        c=$b/ir/$p.c
        break
      fi
    done
    [ -n "$c" ] || none "no .c file for the module $m: see tools/lean-verify/README.md, \"If a .c file is missing\""
  done
  echo "building the native helper library's ${#modules[@]} objects (minutes at the first call; Lake's words: $work/build.log) .." >&2
  held "$work/build.log" lake build "${targets[@]}" || none "lake build failed: see $work/build.log"
  objects=()
  for m in "${modules[@]}"; do
    p=${m//.//} o='' l=''
    for b in .lake/build .lake/packages/*/.lake/build; do
      if [ -f "$b/ir/$p.c.o.export" ]; then
        o=$b/ir/$p.c.o.export l=$b/lib/lean/$p.olean
        break
      fi
    done
    [ -n "$o" ] || none "Lake left no compiled object for the module $m: see $work/build.log"
    # An object that is not newer than what it was compiled from may hold an older text's code.
    # ("-nt" also holds where the second file is missing: so both must be there.)
    { [ -f "${o%.o.export}" ] && [ -f "$l" ] && [ "$o" -nt "${o%.o.export}" ] && [ "$o" -nt "$l" ]; } ||
      none "the object $o is not newer than its .c and its .olean: see tools/lean-verify/README.md, \"If an object is refused\""
    objects+=("$o")
  done
  # Linked anew at every call: a library from an earlier call is never loaded.
  held "$work/link.log" lake env leanc -shared -o "$work/libnative.so" "${objects[@]}" ||
    none "linking the helper library failed: see $work/link.log"
  echo "the helper library: ${#objects[@]} objects" >&2
  flags=("--load-dynlib=$work/libnative.so" -D interpreter.prefer_native=true)
  long="about half an hour"
fi

echo "running the verifier's executable form inside Lean ($long); its lines are shown when it has ended, and are written to $work/lean.err as they come .." >&2
# LeanVerify.lean takes its seven arguments from one variable, one to a line. No path here holds a
# newline: all lie below the work directory.
# Lean writes its two streams into files. Nothing runs beside it to show its lines as they come: so
# every job of this script's is one process, a "timeout", which the trap signals as a job and never
# by number.
# The proof may be made to provoke output, and the work directory may lie in memory: so Lean's job may
# write no more than "bound" bytes into any one file ("prlimit" sets that and becomes Lake; the system
# ends whichever process of the job would write more). A usual run writes some 10,000 bytes.
bound=1048576
LEAN_VERIFY_ARGS=$(printf '%s\n' "$work/claim/config" "$work/claim/ID.txt" "$work/claim/example.req" \
  "$work/claim/example.y" "$work/parser/parsed/key.parsed.bin" "${leaves[0]}" "$work/parser/parsed/proof.parsed.bin") \
  timeout 0 prlimit --fsize="$bound" lake env lean -j1 --tstack=4000000 ${flags[@]+"${flags[@]}"} "$ask" \
  < /dev/null > "$work/lean.out" 2> "$work/lean.err" &
wait "$!"
status=$?
# Lean's lines, each behind "lean: ": so that a line of Lean's, which can be "VERDICT: ACCEPT" in a call
# that ends with no verdict, is not taken for this script's.
sed 's/^/lean: /' -- "$work/lean.err" >&2
# Nothing is left to stop, and what follows takes no time: from here on a signal is passed over, so that
# none can lay a second line on standard output behind a verdict's.
trap '' INT TERM HUP
for f in "$work/lean.err" "$work/lean.out"; do
  [ "$(stat -c %s -- "$f")" -lt "$bound" ] ||
    none "Lean's run wrote $bound bytes into $f, the most that this script lets it write (a usual run writes some 10,000), and counts for nothing"
done

# Lean ends with 0 whenever a file merely elaborates: so 0 is no accept. LeanVerify.lean, beside
# this script, ends Lean with 10 for accept, 11 for reject and 12 for no verdict, and the last line
# says the same in words. A status counts only with its line, last and byte for byte, with no second line
# on a verdict before it, and with Lean's lines on the claim being the second reader's.
last_is() { tail -n 1 -- "$work/lean.err" | cmp -s -- - <(printf '%s\n' "$1"); }
said=$(grep -a -c -E '^(VERDICT: |NO VERDICT)' -- "$work/lean.err")
grep -a -E '^(claim: |not part of the claim: |public files: )' -- "$work/lean.err" > "$work/claim.lean.txt"
if cmp -s -- "$work/claim.lean.txt" "$work/claim.txt"; then fits=yes; else fits=no; fi
# Once more, right before the verdict: the first time lies half an hour back.
echo "$public" >&2
echo "$policy" >&2
if [ "$status" = 10 ] && [ "$said" = 1 ] && [ "$fits" = yes ] && last_is 'VERDICT: ACCEPT'; then
  echo "VERDICT: ACCEPT ($how)" || exit 2 # no code 0 without its line
  exit 0
fi
if [ "$status" = 11 ] && [ "$said" = 1 ] && [ "$fits" = yes ] && last_is 'VERDICT: REJECT'; then
  echo "VERDICT: REJECT" || exit 2 # nor code 1
  exit 1
fi
if [ ! -s "$work/claim.lean.txt" ]; then
  none "Lean ended, with status $status, before it printed the claim: see $work/lean.err and $work/lean.out"
elif [ "$fits" = no ]; then
  none "Lean's reading of the claim differs from read_claim.py's: compare $work/claim.lean.txt with $work/claim.txt"
fi
none "Lean's status ($status) and its last lines do not make one verdict: see $work/lean.err and $work/lean.out"
