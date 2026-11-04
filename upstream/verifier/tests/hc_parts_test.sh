#!/bin/bash
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.

# Builds one test program, verifier/tests/hc_*_test.cpp, and runs it.
# The script makes <new folder> (mode 700) and any missing parents (under umask 077, to close them
# to other accounts too), builds there, and removes the built program after the run. The folders
# stay; <new folder> holds the output, <program's name>.txt, unless compilation failed.
# "--field 2" may come first and changes nothing: the only field is number 2 (p = 2^64 - 2^32 + 1).
# usage: hc_parts_test.sh [--field 2] <program's name> <new folder> [the program's arguments]
# Exit status 0: the program exited with status 0; 1: it did not; 2: wrong arguments,
# <new folder> exists, its absolute path has a character or part that the script rejects,
# mkdir failed, made_here() rejected the new folder, cd did not reach it, or compilation failed.
# -fno-ipa-modref avoids a bug in g++ 13.3.0, shown by verifier/tests/compiler_pass_control.sh.
# verifier/src/cp_verify.cpp, the verifier's source, which each hc_*_test.cpp includes, requires
# VF_SOURCE_DIGEST: verifier/build.sh passes a digest of the sources, this script 64 zeros.
# Started by the runner; by hand: below a directory of your own: the script does not examine the directories above the new folder. (Runner: run_suite.py. The work folder goes there.)
set -euo pipefail
unset CDPATH
made_here() { # <a folder that this script has just made>: it must be this account's, closed to others (mode 700; 2700 below a set-group-ID directory) and empty
  # One cd, then three tests of ".", so that all three look at the same folder. If stat or ls fails,
  # the folder is rejected too. With -P, cd resolves ".." as mkdir does. The trailing slash keeps
  # cd from taking a folder named "-" for the previous directory.
  (cd -P -- "$1/" && [ -O . ] && [[ $(stat -c %a .) == *700 ]] && list=$(ls -A) && [ -z "$list" ]) && return
  echo "hc_parts_test.sh: $1 is not the folder that was just made: another was put in its place: name a new folder below folders in which only you can write" >&2
  return 1
}
export LC_ALL=C
here=$(cd -- "$(dirname "$0")" && pwd) || exit 2
dw=8 field=2
if [[ ${1:-} == --field && ${2:-} == 2 ]]; then
  field=$2
  shift 2
fi
if [[ $# -lt 2 || $1 != hc_*_test || ! -f $here/$1.cpp || -z $2 ]]; then
  echo "usage: hc_parts_test.sh [--field 2] <an hc_*_test.cpp of this folder, without .cpp> <new folder> [arguments ..]" >&2
  exit 2
fi
if [[ -e $2 ]]; then
  echo "hc_parts_test.sh: $2 exists already; name a directory that does not exist yet" >&2
  exit 2
fi
name=$1
case $2 in /*) out=$2 ;; *) out=$PWD/$2 ;; esac; case /$out/ in */-*|*/../*|*[!A-Za-z0-9_./-]*) echo "hc_parts_test.sh: the work folder's whole path (the current directory included) may hold letters, digits" \
  "and _ . / - only, and no part of it may be \"..\" or begin with \"-\": name another" >&2; exit 2 ;; esac
(umask 077 && mkdir -p -- "$(dirname -- "$2")") && mkdir -m 700 -- "$2" && made_here "$2" || exit 2 # the mkdir without -p fails if anything has appeared at the name since the -e test on it above
out=$(cd -- "$2" && pwd) && [ "$out" -ef "$2" ] || { echo "hc_parts_test.sh: cd does not lead to the folder just made: name another" >&2; exit 2; } # -ef: a second look that cd reached the directory just made; the rule above refuses the names that lead cd elsewhere
shift 2
word=$(printf '0%.0s' {1..64})
if ! g++ -O2 -fno-ipa-modref -std=c++17 -pthread -I "$here/../src" -I "$here/../../blocks" -I "$here/../../prover" -DBLK_FIELD=$field -DDIGEST_WORDS=$dw -DVF_SOURCE_DIGEST="\"$word\"" -o "$out/$name" "$here/$name.cpp"; then
  echo "hc_parts_test.sh: $name does not compile" >&2
  exit 2
fi
status=0
"$out/$name" "$@" >"$out/$name.txt" 2>&1 || status=$?
rm -f "$out/$name"
# "|| true": grep exits with status 1 if every line ends in " ok", which is no failure.
{ grep -v ' ok$' "$out/$name.txt" || true; } | tail -n 12
if [[ $status -eq 0 ]]; then
  echo "hc_parts_test.sh: $name: all as expected; the output is in $out/$name.txt"
else
  echo "hc_parts_test.sh: $name: status $status: not as expected; the output is in $out/$name.txt"
  exit 1
fi
