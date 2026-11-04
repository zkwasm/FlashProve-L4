#!/bin/bash
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

#
# Terms: docs/glossary.md.
#
# Loops too long for a test's command in SUITE.md. They build the checkers of two GPU tests,
# psc_layer_test (PSC, the sumcheck with a hiding polynomial) and forest_lookup_test (a group of
# lookup calls), and run malicious provers ("cheats") of forest_lookup_test. A checker is a CPU
# program that runs the verifier's functions on its test's files. Exit code 0 whatever the printed
# counts: SUITE.md has the line expected of `checks`; no test in SUITE.md runs `data_cheats`.
# This script takes no GPU lock: run_suite.py holds one for tests of kind gpu.
#   checks <checks>                          builds <checks>/psc_layer_check.f2 and
#                                            <checks>/forest_lookup_check.f2 from verifier/tests
#   data_cheats <programs> <checks> <work>   forest_lookup_test's cheats on the data: crossed, key,
#                                            multiplicity
# <programs>: forest_lookup_test's directory. <work> and, for `checks`, <checks> must not exist yet.
# Elsewhere <checks> is the directory that `checks` created. A rejected directory gives exit
# code 2. When you run this script by hand, name directories below a directory of your own.
# PSC_ROWS_FIELD, if set, must be 2 (not checked): it replaces the field's number, 2, in the name
# forest_lookup_check.f2 of the checker that is run, and `checks` builds for no other number.
# Started by the runner; by hand: below a directory of your own. (Runner: run_suite.py. The work folder goes there.)
set -u
unset CDPATH
# A plain mkdir fails if the name is taken, so no existing file is overwritten.
new_directory() {
  [ ! -e "$1" ] || { echo "psc_rows.sh: $1 exists already; name a directory that does not exist yet" >&2; exit 2; }
  local whole
  case $1 in /*) whole=$1 ;; *) whole=$PWD/$1 ;; esac
  case /$whole/ in */-*|*/../*|*[!A-Za-z0-9_./-]*) echo "psc_rows.sh: the new folder's whole path (the current directory included) may hold letters, digits and _ . / - only, and no part of it may be \"..\" or begin with \"-\": name another: $whole" >&2; exit 2 ;; esac # before anything is made: a path with a blank, "~", "*" or "$" would name another place when somebody retypes it unquoted
  (umask 077 && mkdir -p -- "$(dirname -- "$1")") && mkdir -m 700 -- "$1" || exit 2
}
# forest_lookup_check is run from <checks>, so <checks> must be the caller's own directory and
# closed to other accounts, as "psc_rows.sh checks" creates it. Not checked: the directories above
# <checks>, and <programs>, from which forest_lookup_test is run.
made_by_checks() {
  local d=$1
  while [[ $d == */ || $d == */. ]]; do d=${d%/*}; done # Behind a closing "/" or "/." the test for a link would look at the link's target.
  [ -d "$d" ] && [ ! -L "$d" ] && [ -O "$d" ] && [[ $(stat -c %a -- "$d") == *700 ]] && return
  echo "psc_rows.sh: $1 is not a directory of your own with mode 700; name the directory that \"psc_rows.sh checks\" created" >&2; exit 2
}
word="${1:?a word}"
field="${PSC_ROWS_FIELD:-2}" # which field's checker plays: that of the programs' build
case "$word" in
checks)
  case ${2:?folder} in /*) out=$2 ;; *) out=$PWD/$2 ;; esac # Made absolute before the cd below, so that a relative name counts from the caller's directory.
  new_directory "$out"
  cd "$(dirname "$0")/../../verifier/src" || exit 2 # The sources are found from this script's own place, not from the caller's directory.
  built=0
  for f in 2; do
    for t in psc_layer_check forest_lookup_check; do
      g++ -std=c++17 -O2 -fno-ipa-modref -pthread -DDIGEST_WORDS=8 -DBLK_FIELD=$f -DBLK_HASH=1 -I . -I ../../blocks -I ../../prover ../tests/$t.cpp -o "$out/$t.f$f" && built=$((built + 1))
    done
  done
  echo "psc_rows.sh: $built of 2 checkers built"
  ;;
data_cheats)
  programs="${2:?}" checks="${3:?}" work="${4:?}"
  made_by_checks "$checks"
  new_directory "$work"
  for c in crossed key multiplicity; do
    mkdir -- "$work/$c"
    "$programs/forest_lookup_test" "$work/$c" $c > "$work/$c.txt"
    echo "$c: $(tail -n 1 "$work/$c.txt"); $("$checks/forest_lookup_check.f$field" "$work/$c" refuse | tail -n 1)"
  done
  ;;
*)
  echo "psc_rows.sh: no such word: $word"
  exit 2
  ;;
esac
