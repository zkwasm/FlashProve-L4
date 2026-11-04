#!/bin/bash
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.

# Shows, on the short program verifier/tests/compiler_pass_control.cpp, why verifier/build.sh,
# prover/build.sh and weight_sums/build.sh pass -fno-ipa-modref. The fault: g++ 13.3.0's pass
# ipa-modref can conclude that a function does not read through a reference parameter although the
# function passes the object on by value. The caller's stores into the object are then deleted, and
# the program computes with what was on the stack. A CPU prover that is not in this repository
# (`reference_prover` in the last message), built with -O1 -fno-inline, then stops at its
# self-check internal-cross-sign. The script builds the program four times and runs each build:
#   -O2                              must be right
#   -O2 -fno-ipa-modref              what the build scripts pass to g++: must be right
#   -O2 -fno-inline -fno-ipa-modref  must be right
#   -O2 -fno-inline                  built by g++ 13.3.0, it prints three numbers left on the stack.
#                                    Either result passes; the output says if the fault is present.
# usage: compiler_pass_control.sh <new folder>. The script makes the new folder, which must not
# exist yet, in a directory that must exist, and deletes nothing. A few seconds. Exit status 0: the
# first three are right; 1: any of them is wrong; 2: a wrong call, or set-up or a build failed.
# Exit status 0 shows nothing about other programs, such as cp_verify.
set -uo pipefail
unset CDPATH
here=$(cd -- "$(dirname "$0")" && pwd) || exit 2
if [[ $# -ne 1 || -z $1 || $1 == - ]]; then
  echo "usage: compiler_pass_control.sh <new folder>" >&2
  exit 2
fi
if [[ -e $1 ]]; then
  printf -v named %q "$1"
  echo "compiler_pass_control.sh: $named exists already; name a directory that does not exist yet" >&2
  exit 2
fi
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
new=$1
# The script exits, with nothing created, if open_above() finds that another account may be able to
# change a directory or a symbolic link above the new folder. Otherwise it uses `real`, the path
# without links.
[[ $new == /* ]] || new=$PWD/$new
while [[ $new == */ && $new != / ]]; do new=${new%/}; done
if open_above "${new%/*}/"; then
  case $why in
  'it does not exist') echo "compiler_pass_control.sh: $place, above the new folder, does not exist: make it first; this script makes only the new folder; nothing was run" >&2 ;;
  'it is not a directory' | 'it could not be examined') echo "compiler_pass_control.sh: $place, above the new folder: $why; nothing was run" >&2 ;;
  *)
    printf -v named %q "$new"
    echo "compiler_pass_control.sh: the new folder $named lies below $place, which another account may be able to change ($why): it could put another directory in the new folder's place while" \
      "the script runs and lead what it writes into other files of yours. Name a new folder below directories that only you or root can change (such as one in your home directory," \
      "or in /tmp, whose sticky bit keeps others from replacing it)$cure; nothing was run" >&2
    ;;
  esac
  exit 2
fi
new=${real%/}/${new##*/}
mkdir -m 700 -- "$new" || exit 2 # a plain mkdir fails if anything has appeared at the name since the -e test on it above
# The script changes into the new folder and gives the programs that it builds and starts relative
# names. A process keeps its current directory even if a directory above it is renamed or replaced,
# so these programs are the ones in this folder. Before it builds, it checks that the folder
# belongs to this account, has mode 700 (2700 below a set-group-ID directory) and is empty.
cd -P -- "$new" && [ -O . ] && [[ $(stat -c %a .) == *700 ]] && list=$(ls -A) && [ -z "$list" ] ||
  { echo "compiler_pass_control.sh: cd does not lead to the folder just made: name another" >&2; exit 2; }
bad=0
one() { # <name> <must be right: 1 or 0> <flags ..>
  local name=$1 must=$2 said status=0
  shift 2
  g++ -std=c++17 -Wall -Wextra "$@" -o "$name" "$here/compiler_pass_control.cpp" || exit 2
  said=$("./$name") || status=$?
  if [[ $status -eq 0 && $said == "(1011, 2022, 3033)" ]]; then
    echo "g++ $*: the sum is right"
  elif [[ $must -eq 1 ]]; then
    echo "NOT AS EXPECTED: g++ $*: printed '$said' with status $status, wanted (1011, 2022, 3033) with status 0"
    bad=1
  else
    echo "g++ $*: the sum is wrong, $said. This compiler ($(g++ --version | head -n 1)) has the ipa-modref fault. The build" \
      "scripts switch that pass off: build with them, or switch that pass off yourself. This script shows nothing about other programs, such as cp_verify: see SO-37 in docs/limits-further.md"
  fi
}
one plain 1 -O2
one as_built 1 -O2 -fno-ipa-modref
one no_inline_pass_off 1 -O2 -fno-inline -fno-ipa-modref
one no_inline 0 -O2 -fno-inline
if [[ $bad -eq 0 ]]; then
  echo "compiler_pass_control: as expected"
else
  echo "compiler_pass_control: NOT AS EXPECTED: this compiler gets the sum wrong with flags that must be safe; do not trust its" \
    "builds of cp_verify and reference_prover"
fi
exit $bad
