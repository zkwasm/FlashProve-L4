#!/bin/bash
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.
# Turns a key file, a leaf file and a proof file into parsed files (tools/parser/PARSED-FORM.md).
# What the parser is, and what must not go through it: tools/parser/README.md.
#
# usage: tools/parser/parse_files.sh CONFIG_DIR PUBLIC_KEY_DIR REQUEST_FILE PROOF_FILE NEW_WORK_DIR
#   The first four are the arguments of the same names of cp_verify (AGENTS.md).
#   NEW_WORK_DIR must not exist. The script creates it, writes only there (the compiler also uses
#   the system's temporary directory) and removes nothing.
# It compiles the verifier's source and tools/audit/pubtool.cpp, writes the plans of the files' hash
# chains into NEW_WORK_DIR/tables and the parsed files into NEW_WORK_DIR/parsed, and reads each
# parsed file back with read_parsed.py.
# Needs: bash, GNU coreutils, g++ 13 or later, python3.
# Exit code 0: the last line begins "PARSED AND READ BACK:". 1: a file was refused. 2: a wrong
# call, among them a CONFIG_DIR that is rejected and a NEW_WORK_DIR where another account could
# change it. 3: a tool is missing, the build failed (NEW_WORK_DIR/build.log), or parse_files.py met
# an error of its own.
set -u
umask 077 # what the script makes is this account's alone; parse_files.py refuses a directory that a group can write in
if [ $# -ne 5 ]; then
  echo "usage: tools/parser/parse_files.sh CONFIG_DIR PUBLIC_KEY_DIR REQUEST_FILE PROOF_FILE NEW_WORK_DIR" >&2
  exit 2
fi
unset CDPATH # with it set, a "cd" below could enter another directory than the one named
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P) || exit 2
tree=$(cd -- "$here/../.." && pwd -P) || exit 2
config=$(realpath -e -- "$1") && keys=$(realpath -e -- "$2") && request=$(realpath -e -- "$3") && proof=$(realpath -e -- "$4") ||
  { echo "parse_files.sh: one of the first four arguments does not exist" >&2; exit 2; }
command -v python3 > /dev/null || { echo "parse_files.sh: python3 is needed" >&2; exit 3; }
# pubtool reads the three public files to their end, whatever they are: a FIFO would block it.
for f in policy.txt layout.txt KL.bin; do
  [ -f "$config/$f" ] && [ "$(stat -L -c %s -- "$config/$f")" -le 268435456 ] ||
    { echo "parse_files.sh: $config/$f is not a regular file of at most 2^28 bytes" >&2; exit 2; }
done
work=$5
[[ $work == /* ]] || work=$PWD/$work
while [[ $work == */ && $work != / ]]; do work=${work%/}; done
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
# Before anything is created: an account that can change a place above NEW_WORK_DIR could put another
# directory in its place. From here on `work` is the path without links that open_above() walked,
# plus the last part as typed.
if open_above "${work%/*}/"; then
  echo "parse_files.sh: NEW_WORK_DIR lies below $place: $why$cure. Name a directory below directories that exist and that only you or root can change" >&2
  exit 2
fi
work=${real%/}/${work##*/}
mkdir -m 700 -- "$work" || { echo "parse_files.sh: $work cannot be created: name a directory that does not exist yet" >&2; exit 2; }
cd -- "$work" && mkdir tables parsed || exit 2

echo "building the tool that writes the plans (a minute or two) .."
flags=(-std=c++17 -O2 -fno-ipa-modref -w -pthread "-I$tree/verifier/src" "-I$tree/blocks" "-I$tree/prover" -DBLK_FIELD=2 -DDIGEST_WORDS=8)
zeros='"0000000000000000000000000000000000000000000000000000000000000000"'
g++ "${flags[@]}" "-DVF_SOURCE_DIGEST=$zeros" -DCP_VERIFY_AS_A_UNIT -c "$tree/verifier/src/cp_verify.cpp" -o unit.o >> build.log 2>&1 &&
  g++ "${flags[@]}" "$tree/tools/audit/pubtool.cpp" unit.o -o pubtool >> build.log 2>&1 ||
  { echo "parse_files.sh: the build failed: see $work/build.log" >&2; exit 3; }
# 128: pubtool rejects a configuration whose parameter file asks for fewer bits.
./pubtool "$config" 128 tables >> tables.log 2>&1 || { echo "parse_files.sh: pubtool rejects CONFIG_DIR: see $work/tables.log" >&2; exit 2; }

python3 -B "$here/parse_files.py" tables "$config" "$keys" "$request" "$proof" parsed || exit $?
k=$(od -A n -t u4 --endian=little -j 8 -N 4 -- "$proof" | tr -d ' ') # the leaf table's number, in the proof file's head (R14.2)
python3 -B "$here/read_parsed.py" parsed/key.parsed.bin "--config=$config" "--file=$keys/key.bin" &&
  python3 -B "$here/read_parsed.py" "parsed/leaf-$k.parsed.bin" "--config=$config" "--file=$keys/leaf/$k.bin" &&
  python3 -B "$here/read_parsed.py" parsed/proof.parsed.bin "--config=$config" "--file=$proof" || exit $?
echo "PARSED AND READ BACK: $work/parsed"
