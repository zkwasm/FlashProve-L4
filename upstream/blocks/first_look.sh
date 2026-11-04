#!/bin/bash
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.

# usage: first_look.sh <a new folder of your own, outside the source tree>
# The directory above it must exist. Every directory and symbolic link on the way to it must belong
# to you or root, and no directory there may be writable by its group or others (mode 775, say)
# without the sticky bit: open_above() has the rule in full. The whole path, with links resolved,
# may have only letters, digits and _ . - /
# Needs Linux, GNU make, g++ 13 (C++17), python3 and under 1.5 GB of memory
# (docs/quickstart.md, the table of routes, row "2. CPU: make toy proofs").
# If your environment sets CXX or CXXFLAGS, make uses it in place of g++ or of the default flags in
# blocks/Makefile (-std=c++17 among them).
#
# Shows the proof system at toy size, without a GPU, in about a minute. The script builds one
# program of the library, which makes and checks toy proofs. It then has one of them checked again
# (honest.proof: `ACCEPTED`), and a copy of it with one bit flipped (bent.proof: rejected, with
# the name of the failed check). It writes into the folder that you name. Its own lines and the built
# program remove nothing; make and the compiler remove only files that they made themselves (the
# compiler's temporary files, in the system's temporary directory; an unfinished program in build/).
# Whatever has been written stays after success, failure or Ctrl-C. After a whole run that is the
# program, make.log, the two checked proofs, for each the log, verdict and exit code of its check
# (.log, .txt, .status), and over a hundred files in proofs/. The script rejects a name that exists:
# run it again under another name, and delete the folder yourself.
# The toy shares the library, the proof system and the 128-bit target of the error accounting with
# cp_commit, cp_prove and cp_verify. It is not a model's inference; nothing is hidden
# in its files (all random values come from small numbers in the source); it says nothing about
# speed. No malicious prover is tried: the rejected files are altered honest proofs.
# Exit code: 0 honest.proof was accepted and bent.proof rejected; 1 not so; 2 wrong usage, an
# unusable path, a missing tool or a failed step (also if the program accepts an altered proof).
[ $# -eq 1 ] && [ -n "$1" ] || { echo "usage: first_look.sh <a new folder of your own, outside the source tree>"; exit 2; }
for needs in make python3 id stat readlink realpath; do command -v "$needs" > /dev/null || { echo "first_look.sh needs $needs"; exit 2; }; done
unset CDPATH   # with it set, cd may go elsewhere and print
umask 077      # what this script makes is for its owner only
HERE="$(cd -P -- "$(dirname -- "$0")" && pwd -P)" || exit 2
TOP="${HERE%/*}"   # the folder that holds blocks/
case "$HERE" in *[[:space:]]*) echo "make cannot build from a path with a blank in it: $HERE"; exit 2;; esac
[ ! -e "$1" ] && [ ! -L "$1" ] || { echo "$1 exists: name a new folder"; exit 2; }
D=$1
case "/$D" in */.|*/..) echo "$1: the last part of the name may not be . or .."; exit 2;; esac
# <absolute path of a directory>: walks it from the top as the system does, through each symbolic
# link. Returns 0 at the first place where another account may be able to swap what lies below: one
# whose owner is not this account, root or, under sudo, the account that ran sudo; a directory that
# its group or others can write in without the sticky bit; or one that is missing, is no directory
# or cannot be examined. It then sets place, why and cure for the message. Else it returns 1 and
# sets real, the path without links that it has looked at: the path to use. Keep it in step with
# others_can_change() in run_suite.py.
open_above() {
  local at=/ here=/ part target owner mode kind me links=0 parts=()
  place='' why='' cure='' real=''
  # A here-string needs a writable TMPDIR before bash 5.1. Up to a NUL: names can have newlines.
  # id, not $EUID: bash takes a value for that name from the caller's environment.
  if ! me=$(id -u 2>/dev/null) || [[ ${1:-} != /* ]] ||
    ! IFS=/ read -r -d '' -a parts < <(printf '%s\0' "$1"); then
    printf -v place %q "${1:-}"
    why='it could not be examined'
    return 0
  fi
  set -- ${parts[@]+"${parts[@]}"}
  while :; do
    if ! IFS=' ' read -r owner mode kind < <(
      LC_ALL=C stat -c '%u %a %F' -- "$here" 2>/dev/null
    ); then
      why='it could not be examined'
      if [ ! -e "$here" ] && [ ! -L "$here" ] && [ -x "$at" ]; then why='it does not exist'; fi
    elif [ "$owner" != 0 ] && [ "$owner" != "$me" ] &&
      { [ "$me" != 0 ] || [ "$owner" != "${SUDO_UID:-}" ]; }; then
      why='owned by another account'
      [ "$kind" != 'symbolic link' ] || why='a symbolic link that belongs to another account'
    elif [ "$kind" = directory ]; then
      if ((8#$mode & 8#22)) && ! ((8#$mode & 8#1000)); then
        why='its group or others can write in it, and it has no sticky bit'
        [ "$owner" != "$me" ] || [ "$me" = 0 ] || cure=yes # root's may be shared on purpose
      fi
    elif [ "$kind" != 'symbolic link' ]; then
      why='it is not a directory'
    else # the link's parts come before the rest of the path
      links=$((links + 1))
      if [ "$links" -gt 40 ] ||
        ! IFS= read -r -d '' target < <(readlink -z -- "$here" 2>/dev/null) ||
        ! IFS=/ read -r -d '' -a parts < <(printf '%s\0' "$target"); then
        why='it could not be examined'
      fi
    fi
    if [ -n "$why" ]; then
      printf -v place %q "$here"
      cure=${cure:+"; if $place is yours, close it with chmod go-w"}
      return 0
    fi
    if [ "$kind" = directory ]; then
      at=$here
    else
      set -- ${parts[@]+"${parts[@]}"} "$@"
      [[ $target != /* ]] || at=/
    fi
    while :; do # the next name; ".." as the system takes it, from the directory reached
      if ! (($#)); then
        real=$at
        return 1
      fi
      part=$1
      shift
      case $part in
        '' | .) ;;
        ..) at=${at%/*} at=${at:-/} ;;
        *) break ;;
      esac
    done
    here=${at%/}/$part
  done
}
# Before it creates anything, the script rejects the new folder if another account may be able to
# change a directory above it or a symbolic link on the way. It creates only the path's last part.
[[ $D == /* ]] || D=$PWD/$D
while [[ $D == */ && $D != / ]]; do D=${D%/}; done
if open_above "${D%/*}/"; then
  case $why in
  'it does not exist') echo "first_look.sh: $place, above the new folder, does not exist: make it first; this script makes only the new folder; nothing was run" ;;
  'it is not a directory' | 'it could not be examined') echo "first_look.sh: $place, above the new folder: $why; nothing was run" ;;
  *)
    printf -v named %q "$D"
    echo "first_look.sh: the new folder $named lies below $place, which another account may be able to change ($why): it could put another directory in the new folder's place while" \
      "the script runs and lead what it writes into other files of yours. Name a new folder below directories that only you or root can change (such as one in your home directory," \
      "or in /tmp, whose sticky bit keeps others from replacing it)$cure; nothing was run"
    ;;
  esac
  exit 2
fi
D=${real%/}/${D##*/}
case "$D" in *[!A-Za-z0-9_./-]*) echo "$D: only letters, digits and _ . - / in the folder's whole path: make and its shell would read other signs"; exit 2;; esac
case "$D/" in "$TOP"/*) echo "$D lies inside the source tree: name a folder outside it"; exit 2;; esac
tool=build/f2h1/frame/handover_proof.bin   # from inside the folder
run() { echo "+ $*"; "$@"; }

run mkdir -m 700 "$D" || { echo "$D was not made: name a new folder"; exit 2; }   # without -p, mkdir fails if the name has been taken since the test above: nothing that another user put there is built in or run
# From here on the script works inside the folder, by relative names. A process keeps its current
# directory even if a directory above it is renamed or replaced, so the files that the script itself
# writes, and the program built here that it itself starts, are those in this folder. The check
# after cd requires that the current directory has the expected path (by `realpath .`, which asks
# the system), belongs to this account, has mode 700 (2700 below a set-group-ID directory) and is
# empty.
run cd -P -- "$D" || exit 2
[ "$(realpath .)" = "$D" ] && [ -O . ] && [[ $(stat -c %a .) == *700 ]] && list=$(ls -A) && [ -z "$list" ] ||
  { echo "$D is not the folder that was just made: another was put in its place: name a new folder below folders in which only you can write"; exit 2; }
run mkdir -p proofs/on-the-first-table proofs/on-the-copy || exit 2
# The program (blocks/frame/handover_proof.cpp) commits to a synthetic matrix of 100 rows as two
# tables: the anchor, a commitment's first table (`on-the-first-table` in paths), and a leaf table
# of the same rows (`on-the-copy`). It writes a key file for the anchor (`key.file`), a leaf file
# for the leaf table (`entry.file`) and three proofs on each table, and checks them with
# check_key(), check_entry() and check_proof(), which cp_verify runs too. About fifty altered proofs
# that it rejected stay in the folder (`0.altered-NN.proof`).
# Only make gets absolute paths: it runs in the source directory. `SETTINGS=f2h1` is
# blocks/Makefile's default: the field p = 2^64 - 2^32 + 1 (Goldilocks).
echo "+ make -C $HERE OUT=$D/build SETTINGS=f2h1 $D/$tool > make.log 2>&1"
make -C "$HERE" OUT="$D/build" SETTINGS=f2h1 "$D/$tool" > make.log 2>&1 || { echo "the build failed (is there a C++17 compiler?): see $D/make.log"; exit 2; }
run "./$tool" ./proofs || { echo "the toy's files were not made or not accepted"; exit 2; }
run cp proofs/on-the-copy/0.proof honest.proof || exit 2
turn='import sys; b = bytearray(open(sys.argv[1], "rb").read()); b[len(b) // 2] ^= 1; open(sys.argv[2], "xb").write(b)'   # one bit of the middle byte
echo "+ python3 -I -c '$turn' honest.proof bent.proof"   # -I: python3 loads no code from the current directory
python3 -I -c "$turn" honest.proof bent.proof || exit 2
# A judging call (`judge`) regenerates the toy's key file and leaf file in memory and reads only the
# given proof file. It prints its verdict in its last line and writes it next to that file
# (honest.txt, bent.txt). Exit code: 0 if the proof is accepted, 1 if rejected, 2 if not
# checked. `ACCEPTED` also covers a proof whose bits reach the target only once its work is counted.
for which in honest bent; do
  echo "+ ./$tool ./proofs judge on-the-copy 0 ./$which.proof > $which.log 2>&1"
  "./$tool" ./proofs judge on-the-copy 0 "./$which.proof" > "$which.log" 2>&1
  echo $? > "$which.status"
done
[ -s honest.txt ] && [ -s bent.txt ] || { echo "a file was not judged: see $D/honest.log, $D/bent.log"; exit 2; }
if grep -qx ACCEPTED honest.txt && [ "$(cat honest.status)" = 0 ] && ! grep -qx ACCEPTED bent.txt && [ "$(cat bent.status)" = 1 ]; then
  echo "AS EXPECTED: the honest copy was accepted, and the copy with one bit flipped was rejected. The last line of each check:"; status=0
else
  echo "NOT AS EXPECTED. The last line of each check:"; status=1
fi
for which in honest bent; do   # the logs name the file from inside the folder; here it is printed with the folder's name
  last=$(tail -n 1 "$which.log")
  case $last in ./*) echo "$D/${last#./}";; *) echo "$last";; esac
done
exit $status
