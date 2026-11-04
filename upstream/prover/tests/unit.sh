#!/bin/bash
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.

# Builds one unit test from one source file and runs it: a .cpp with g++, for the CPU; a .cu with
# nvcc (minimum version: prover/check_nvcc_version.sh), for an H100 (sm_90a).
#   prover/tests/unit.sh build|run|both <source file, from prover/> <folder for the program and its logs> [<one run's arguments> ...]
#   prover/tests/unit.sh explain-sigill <name>
# The second form is for a program built with the same switches elsewhere that died with SIGILL: it
# exits 77 with a "not run: <name>: ..." line if this CPU cannot run the SHA instructions, 1 if it can
# or that could not be tried, and 2 for a usage error, a folder that could not be made, or a TMPDIR
# that another account may be able to change (the same look as for the output directory below). It
# makes its small program in a new folder in TMPDIR (a relative TMPDIR counts as unset: /tmp) and
# removes it again.
# "build" only builds; "run" runs what an earlier call built into the same output directory
# (`folder` in the usage text). Each further argument contains one run's arguments, separated by
# white space; with none, the test runs once, without arguments. It runs from prover/: some tests
# open files by paths relative to it. A relative output directory starts at the caller's working
# directory. The script removes nothing that it did not make in the same call. In the output
# directory it overwrites <test>, <test>.build.log and <test>.log, which are named after the source
# file alone, so builds with and without --slot-count need separate directories. When a .cpp test
# dies with SIGILL, it makes a small test program in a new folder there (sha_probe.<random>), runs
# it and removes that folder again; see "The SHA instructions" below. It rejects a directory that it finds open to
# another account: see "The output directory" below.
# Options, anywhere among the arguments:
#   --field 2     changes nothing, as in prover/build.sh: there is one field, p = 2^64 - 2^32 + 1.
#                 Any other value, or no value after --field, is a usage error.
#   --slot-count  builds with -DSLOT_COUNT=1 (prover/common/setting.h), which adds checks to some
#                 tests: prover/tests/forest_lookup_bench.cu then checks that each slot of the
#                 prover's message was written exactly once.
#   --dw 8        changes nothing: a digest has eight 32-bit words (-DDIGEST_WORDS=8). In any other
#                 form, --dw counts as an ordinary argument.
# Exit status: 0: the build and every run succeeded; 1: a run failed; 2: no test ran, for example
# after a usage error, a rejected output directory, a failed build, or "run" without a built
# program; 77: no run failed, but at least one reported that it cannot run on this machine
# (status 2, and the last line of its output begins with "not run: " or "<test>: not run: "), or
# a .cpp test died with SIGILL on a CPU that cannot run the SHA instructions that it is built to
# use. On such a CPU every SIGILL of a .cpp test counts so, whatever its cause, and is never a pass.
set -uo pipefail
unset CDPATH
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
callers_directory=$PWD
cd "$(dirname "$0")/.." || exit 2
dw=8 field=2 words=() counting=()
while [[ $# -gt 0 ]]; do
  if [[ $1 == --dw && ${2:-} == 8 ]]; then
    dw=$2
    shift 2
  elif [[ $1 == --field ]]; then
    if [[ ${2:-} != 2 ]]; then
      echo "unit.sh: there is one field, number 2 (p = 2^64 - 2^32 + 1): --field takes 2 and nothing else" >&2
      exit 2
    fi
    field=$2
    shift 2
  elif [[ $1 == --slot-count ]]; then
    counting=(-DSLOT_COUNT=1)
    shift
  else
    words+=("$1")
    shift
  fi
done
set -- ${words[@]+"${words[@]}"}
# The SHA instructions. A .cpp test is built with -msha -msse4.1 -mssse3 (below). On a CPU that cannot
# run those instructions, a test that uses them dies with SIGILL (status 132); a test that does not
# use them passes. When a run of a .cpp test dies with SIGILL, a one-instruction program built with
# the same switches tells whether the CPU lacks them: it runs one sha256rnds2 on a value that the
# compiler cannot know (the number of its arguments). It is built in a new folder of its own, made
# with mktemp, so two calls that share an output directory never use the same files.
# cpu_lacks_sha <folder>: makes that program's folder in <folder>, builds and runs the program,
# removes what it made, and sets sha_probe_status: 132 (SIGILL), 0 or 1 (the instruction ran),
# -1 (no g++ here builds it), -2 (no folder could be made), or another status (it could not start,
# for example 126 on a file system that does not let programs start). Returns 0 only for 132.
cpu_lacks_sha() {
  local probe
  sha_probe_status=-2
  probe=$(mktemp -d "$1/sha_probe.XXXXXX") || return 1
  printf '%s\n' '#include <immintrin.h>' \
    'int main(int argc, char **) { __m128i a = _mm_set1_epi32(argc); a = _mm_sha256rnds2_epu32(a, a, a); return _mm_cvtsi128_si32(a) == 12345; }' \
    >"$probe/probe.cpp"
  if TMPDIR=$probe g++ -O1 -fno-ipa-modref -msha -msse4.1 -mssse3 -o "$probe/probe" "$probe/probe.cpp" >"$probe/build.log" 2>&1; then
    sha_probe_status=0
    # The outer 2>/dev/null also hides bash's own report that the program was killed; no core file is written.
    { (ulimit -c 0 && exec "$probe/probe" </dev/null >/dev/null 2>&1); } 2>/dev/null || sha_probe_status=$?
  else
    sha_probe_status=-1
  fi
  rm -f -- "$probe/probe" "$probe/probe.cpp" "$probe/build.log" # the files made just above, by this call
  rmdir -- "$probe" 2>/dev/null # and its folder; it stays if the compiler left a file of its own in it
  [[ $sha_probe_status -eq 132 ]] # 128 + SIGILL
}
act=${1:-} source=${2:-} out=${3:-}
if [[ $act == explain-sigill ]]; then # unit.sh explain-sigill <name>: why did <name> die with SIGILL here?
  if [[ ! ${2:-} =~ ^[A-Za-z0-9_.-]+$ || $# -ne 2 ]]; then
    echo "usage: unit.sh explain-sigill <name of the program that died with SIGILL>" >&2
    exit 2
  fi
  [[ ${TMPDIR:-} == /* ]] && where=$TMPDIR || where=/tmp # a relative TMPDIR would now lead elsewhere, after the cd above
  if open_above "${where%/}/"; then # the same look as for an output directory: the small program is built and run there
    echo "unit.sh: explain-sigill: $(printf %q "$where") (TMPDIR, or /tmp) is or lies below $place, which another account may be able to" \
      "change ($why): it could choose what runs as you. Set TMPDIR to a directory that only you or root can change$cure; nothing was run" >&2
    exit 2
  fi
  base=$(mktemp -d "$where/unit_sh_explain.XXXXXX") || exit 2
  lacks=1
  cpu_lacks_sha "$base" && lacks=0
  rmdir -- "$base" 2>/dev/null # made just above, by this call
  if [[ $lacks -eq 0 ]]; then
    echo "not run: $2: this CPU cannot run the SHA instructions: the program died with SIGILL, and so did a one-instruction" \
      "program that uses sha256rnds2; run it on a CPU that can"
    exit 77
  elif [[ $sha_probe_status -eq 0 || $sha_probe_status -eq 1 ]]; then
    echo "$2 died with SIGILL, but the SHA instructions run here, so that is not why"
  elif [[ $sha_probe_status -eq -1 ]]; then
    echo "$2 died with SIGILL; whether this CPU can run the SHA instructions could not be tried: the compiler did not build the small program"
  else
    echo "$2 died with SIGILL; whether this CPU can run the SHA instructions could not be tried: the small program ended with status $sha_probe_status"
  fi
  exit 1
fi
case "$source" in
  *.cpp) kind=host ;;
  *.cu) kind=gpu ;;
  *) kind= ;;
esac
if [[ ! $act =~ ^(build|run|both)$ || -z $kind || -z $out || ! -f $source ]]; then
  echo "usage: unit.sh build|run|both <source file, from prover/> <folder for the program and its logs>" \
    "[<one run's arguments> ...], or unit.sh explain-sigill <name>" >&2
  exit 2
fi
shift 3
export PATH=$PATH:/usr/local/cuda/bin
[[ $kind != gpu || $act == run ]] || ./check_nvcc_version.sh || exit 2 # refuses an nvcc older than 13.2.86
[[ $out == /* ]] || out=$callers_directory/$out
while [[ $out == */ || $out == */. ]]; do out=${out%/*}; done # After a trailing "/" or "/." stat below would look at a symbolic link's target.
case ${out##*/} in
  '' | ..)
    echo "unit.sh: ${out:-/}: the output directory's path must end in a name of its own, not in \"..\", and cannot be \"/\"" >&2
    exit 2
    ;;
esac
# The output directory. Programs are started from it, and files in it are overwritten by name, so
# it must meet two conditions. If it does not, the script exits with status 2 and has written
# nothing into it and started nothing from it.
# - The directory above it exists, and open_above() finds neither that directory nor one above it
#   open to change by another account. The script creates only the last part of the path
#   and then uses the path that open_above() has checked, which has no symbolic links.
# - If the name is free, the script creates the directory with mode 700. Otherwise it is the
#   caller's own directory, not a symbolic link, not writable by its group or by others, and
#   nothing in it belongs to another account.
# The shell follows a symbolic link at a log's name, and at the program's name when it starts the
# program: in a directory that meets the conditions, the link is the caller's own. Calls may share
# one directory.
if open_above "${out%/*}/"; then
  if [ "$why" = 'it does not exist' ]; then
    echo "unit.sh: $place, above the output directory, does not exist: make it first; this script makes only the output directory. Nothing was created." >&2
    exit 2
  fi
  echo "unit.sh: the output directory $(printf %q "$out") lies below $place, which another account may be able to change ($why): it could put another" \
    "directory in the output directory's place while the test is built and run and choose what runs as you. Name an output directory below directories" \
    "that only you or root can change (such as one in your home directory, or in /tmp, whose sticky bit keeps others from replacing it)$cure;" \
    "nothing was run" >&2
  exit 2
fi
out=${real%/}/${out##*/}
# The name is not tested before mkdir. A failed mkdir does not end the script: if something has
# the name (another call may have just created the directory), the check below decides.
mkdir_said=$(mkdir -m 700 -- "$out" 2>&1)
# One call of stat reads the owner, the type and the permissions together: between two calls,
# another account could replace what is at the path. stat does not follow a symbolic link, so the
# "d" excludes one. The two "-" in the pattern: no write permission for the group, none for others.
# What another account put into the directory while it could write there is still there after a
# chmod: hence the search with find.
# "-uid +N" or "-uid -N" (a higher or a lower number) and not "! -uid N": if a file is removed
# before find asks for its owner (another call's temporary file), each test of the owner is false,
# and "!" would make that a match.
if [[ $(stat -c '%u %A' -- "$out" 2>/dev/null) != "$(id -u) "d????-??-? ||
  -n $(find "$out" -mindepth 1 \( -uid "+$(id -u)" -o -uid "-$(id -u)" \) -print -quit 2>/dev/null) ]]; then
  [[ -z $mkdir_said || -e $out || -L $out ]] || echo "$mkdir_said" >&2 # why nothing could be created at a free name
  echo "unit.sh: ${out:-/}: name an output directory that does not exist yet, or your own directory (not a symbolic link) that neither its group nor" \
    "others can write and in which nothing belongs to another account" >&2
  exit 2
fi
t=$(basename "${source%.*}")
# The compilers' temporary files go into the output directory, not under the caller's TMPDIR or
# /tmp, where other accounts can create files too. The test's runs get the caller's TMPDIR back.
# -fno-ipa-modref, in both compile commands below, works around a bug in g++ 13.3.0, which
# verifier/tests/compiler_pass_control.sh reproduces.
callers_tmpdir=${TMPDIR-} had_tmpdir=${TMPDIR+yes}
export TMPDIR=$out
if [[ $act == run ]]; then
  [[ -x $out/$t ]] # what follows takes a missing program for one that did not build
elif [[ $kind == host ]]; then
  g++ -O2 -fno-ipa-modref -std=c++17 -DDIGEST_WORDS=$dw -DBLK_FIELD=$field ${counting[@]+"${counting[@]}"} -Wall -Wextra -pthread -msha -msse4.1 -mssse3 -I. -I../blocks -o "$out/$t" "$source" \
    >"$out/$t.build.log" 2>&1
else
  nvcc -O3 -std=c++17 -arch=sm_90a --extended-lambda --default-stream per-thread -Xcompiler -pthread,-fno-ipa-modref -I. -I../blocks -DDIGEST_WORDS=$dw -DBLK_FIELD=$field \
    ${counting[@]+"${counting[@]}"} -o "$out/$t" "$source" -lcublas -lcublasLt -lcuda >"$out/$t.build.log" 2>&1
fi
# shellcheck disable=SC2181 # $? is the status of the branch that ran
if [ $? -ne 0 ]; then # the log stays
  echo "$t: does not build: see $out/$t.build.log"
  exit 2
fi
if [[ $act == build ]]; then
  echo "$t: built"
  exit 0
fi
[[ $# -gt 0 ]] || set -- ""
if [[ -n $had_tmpdir ]]; then TMPDIR=$callers_tmpdir; else unset TMPDIR; fi
failed=0 not_here=0 no_sha=0 judged=
: >"$out/$t.log"
for arguments in "$@"; do
  status=0
  # Only the last line that this run appends counts. A run that ends with status 2 and has the
  # words "not run" elsewhere in its output is a failure.
  before=$(wc -c <"$out/$t.log")
  # shellcheck disable=SC2086 # split on purpose: the arguments of one run
  "$out/$t" $arguments >>"$out/$t.log" 2>&1 </dev/null || status=$?
  if [[ $status -eq 132 && $kind == host && -z $judged ]]; then # judged once per call
    cpu_lacks_sha "$out" && judged=lacks || judged=has
  fi
  if [[ $status -eq 132 && $judged == lacks ]]; then
    no_sha=1
  elif [[ $status -eq 2 && $(tail -c "+$((before + 1))" "$out/$t.log" | tail -n 1) =~ ^("$t: ")?"not run: " ]]; then
    not_here=1
  elif [[ $status -ne 0 ]]; then
    failed=1
    echo "$t $arguments: status $status"
  fi
done
if [[ $failed -ne 0 ]]; then
  echo "$t: fails: see $out/$t.log"
  exit 1
fi
if [[ $no_sha -ne 0 ]]; then
  echo "not run: $t: this CPU cannot run the SHA instructions: the test died with SIGILL, and so did a one-instruction" \
    "program that uses sha256rnds2; run it on a CPU that can"
  exit 77
fi
if [[ $not_here -ne 0 ]]; then
  echo "$t: cannot be run on this machine: $(tail -n 1 "$out/$t.log")"
  exit 77
fi
echo "$t: passes"
