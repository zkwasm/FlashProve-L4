#!/bin/bash -p
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Makes the list tools/lean-verify/native-modules.txt anew, inside Lean, and compares. It judges no
# proof. What the list is for, and how it is made: LeanVerifyWalk.lean, beside this script.
# A script of its own, so that exit code 0 of lean_verify.sh means an accept and nothing else.
#
# usage: tools/lean-verify/check_list.sh
# Before the call: build what lean_verify.sh asks for (README.md, step 1).
# Needs: bash, GNU coreutils, diffutils, grep and sed, lake on the PATH.
# The script itself removes nothing, and it builds nothing. It writes below a fresh directory that
# mktemp makes, which its first line names: delete it when you are done. Lake, which it calls, writes
# below lean/.lake and can replace or delete files there (lean/CHECKING.md, "Before you start").
# Run it as the account that built lean/.lake, not with sudo. Run one call at a time.
#
# Standard output holds one line, one of three; all else goes to the error stream.
# Exit code 0: "native-modules.txt: the same as made anew". 1: "native-modules.txt: NOT the same as
# made anew": the error stream names the two files to compare. 2: "native-modules.txt: not checked":
# wrong arguments, a library that is not built, a signal, a run of Lean that failed.
set -u
umask go-w
ulimit -c 0 # a panic aborts Lean: no core file, which would be laid into lean/
LC_ALL=C
export LC_ALL
unset POSIXLY_CORRECT # as in lean_verify.sh
stop() {
  printf 'check_list.sh: %s\n' "$*" >&2
  echo "native-modules.txt: not checked"
  exit 2
}
# <file> <command ..>: as in lean_verify.sh: the command's two streams into the file, a process group of
# its own, and a trap that reaches it, but for the instant of "timeout"'s own start (README.md).
held() {
  local log=$1
  shift
  timeout 0 "$@" < /dev/null > "$log" 2>&1 &
  wait "$!"
}
# shellcheck disable=SC2154
trap 'trap "" INT TERM HUP
  for j in $(jobs -r | sed -n "s/^\[\([0-9]*\)\].*/\1/p"); do kill -TERM "%$j" 2> /dev/null; done
  stop "a signal stopped this script"' INT TERM HUP
[ $# -eq 0 ] || stop "usage: tools/lean-verify/check_list.sh (it takes no argument)"
# Looked for up front: further down a missing tool would be told as another fault.
for t in lake timeout sed grep cmp dirname mktemp cat; do
  command -v "$t" > /dev/null || stop "$t is not on the PATH"
done
unset CDPATH LEAN_PATH LEAN_SRC_PATH
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P) || stop "the script's own directory cannot be entered"
# As in lean_verify.sh: the file that runs must be the one of its name in "here".
[ "$here/check_list.sh" -ef "${BASH_SOURCE[0]}" ] ||
  stop "this script was called through a symbolic link, or by a path in which \"..\" follows one: call it by its own path, without \"..\""
tree=$(cd -- "$here/../.." && pwd -P) || stop "the repository's top directory cannot be entered"
# "$(..)" drops newlines at a name's end: what is worked in must be what lies above.
[ "$tree/tools/lean-verify" -ef "$here" ] || stop "the repository's top directory is not the one above this script"
# The path is printed below: behind a newline in it, a folder's name could pose as a line of this script's.
case $tree in *$'\n'*) stop "the repository's path holds a newline" ;; esac
walk=$here/LeanVerifyWalk.lean list=$here/native-modules.txt
{ [ -f "$walk" ] && [ -f "$list" ]; } || stop "LeanVerifyWalk.lean or native-modules.txt is missing beside this script"
for d in "$tree/lean" "$tree/lean/.lake" "$tree/lean/.lake/build" "$tree/lean/.lake/packages" "$tree"/lean/.lake/packages/* \
  "$tree"/lean/.lake/packages/*/.lake "$tree"/lean/.lake/packages/*/.lake/build; do
  case $d in *$'\n'*) stop "a name in $tree/lean/.lake/packages holds a newline" ;; esac
  [ ! -L "$d" ] ||
    stop "$d is a symbolic link: refused, since Lake could write through it: see lean/CHECKING.md, \"Before you start\""
done
[ -d "$tree/lean/.lake/packages" ] || stop "$tree/lean/.lake/packages is missing: see lean/CHECKING.md, step 3"
[ -O "$tree/lean/.lake" ] || stop "$tree/lean/.lake belongs to another account: run this as that account, not with sudo"
work=$(mktemp -d) || stop "mktemp -d failed"
work=$(cd -- "$work" && pwd -P) || stop "the work directory cannot be entered"
case $work in *$'\n'*) stop "the work directory's path holds a newline: set TMPDIR to another place" ;; esac
echo "work directory: $work" >&2
echo "tree: $tree" >&2
# The walk takes a module for this library's if its source lies below the directory where Lean starts.
cd -- "$tree/lean" || stop "lean/ cannot be entered"
export LEAN_ABORT_ON_PANIC=1
held "$work/fits.log" lake build --no-build Johnson.PVR.Program Johnson.PVR.ProgramGuard ||
  stop "the compiled Lean library lacks files or is out of date: build it (tools/lean-verify/README.md, step 1); Lake's output: $work/fits.log"
LEAN_VERIFY_LIST=$work/walked held "$work/walk.out" lake env lean -j1 "$walk" ||
  stop "Lean's run of LeanVerifyWalk.lean failed: see $work/walk.out"
# Nothing is left to stop: from here on a signal is passed over, so that none can lay a second line on
# standard output.
trap '' INT TERM HUP
[ -s "$work/walked" ] || stop "Lean's run of LeanVerifyWalk.lean wrote no list: see $work/walk.out"
cat -- "$work/walk.out" >&2
grep -v -e '^#' -e '^$' -- "$list" > "$work/listed" || stop "native-modules.txt names no module"
if cmp -s -- "$work/walked" "$work/listed"; then
  echo "native-modules.txt: the same as made anew" || exit 2
  exit 0
fi
echo "check_list.sh: compare $work/walked (made anew) with $work/listed (the list's names)" >&2
echo "native-modules.txt: NOT the same as made anew" || exit 2
exit 1
