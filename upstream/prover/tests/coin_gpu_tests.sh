#!/bin/bash
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.

# Builds and runs seven GPU tests of the private randomness (`coins`). Then checks that
# coin_renew_gpu_test exits with an error in eight misuse cases (`scene`) of the generator's key,
# counter or stream, and coin_fill_gpu_test when the driver rejects the launch in hz::fill()
# (prover/hiding/trees.cuh).
#   prover/tests/coin_gpu_tests.sh <new directory for the programs and their logs>
# Started by the runner; by hand: below a directory of your own (runner: run_suite.py), since the
# script itself does not check the directories above the new one.
# The directory is not removed. Last line: "coin_gpu_tests: all hold" (exit status 0),
# "coin_gpu_tests: <n> FAIL" (1) or, if each of the 16 runs reports that the machine has no GPU,
# "coin_gpu_tests: not run: this machine has no GPU" (77). Exit status 2: nothing was run.
set -uo pipefail
unset CDPATH
made_here() { # <a folder that this script has just made>: it must be this account's, closed to others (mode 700; 2700 below a set-group-ID directory) and empty
  # One cd, then three tests of ".", so that all three see the same folder; if stat or ls fails, that refuses too. -P: cd reads ".." as mkdir does. The slash: a folder
  # named "-" is not taken for the previous directory.
  (cd -P -- "$1/" && [ -O . ] && [[ $(stat -c %a .) == *700 ]] && list=$(ls -A) && [ -z "$list" ]) && return
  echo "coin_gpu_tests.sh: $1 is not the folder that was just made: another was put in its place: name a new folder below folders in which only you can write" >&2
  return 1
}
[ $# -eq 1 ] && [ -n "$1" ] || { echo "usage: prover/tests/coin_gpu_tests.sh <new folder>" >&2; exit 2; }
[ ! -e "$1" ] || { echo "coin_gpu_tests.sh: $1 exists already; name a directory that does not exist yet" >&2; exit 2; }
case $1 in /*) out=$1 ;; *) out=$PWD/$1 ;; esac # Made absolute before the cd below, so that a relative name counts from the caller's directory.
case /$out/ in */-*|*/../*|*[!A-Za-z0-9_./-]*) echo "coin_gpu_tests.sh: the new folder's whole path (the current directory included) may hold letters, digits and _ . / - only, and no part of it may be \"..\" or begin with \"-\": name another: $out" >&2; exit 2 ;; esac # before anything is made: a path with a blank, "~", "*" or "$" would name another place when somebody retypes it unquoted
(umask 077 && mkdir -p -- "$(dirname -- "$out")") && mkdir -m 700 -- "$out" && made_here "$out" || exit 2 # A plain mkdir fails if the name is taken, so only programs that this run builds are run from this directory.
cd "$(dirname "$0")/.." || exit 2
bad=0 not_run=0 runs=0
for name in coin_chacha_gpu_test coin_fill_gpu_test coin_contexts_gpu_test coin_renew_gpu_test coins_by_value_gpu_test slot_pads_agree_gpu_test coins_own_gpu_test; do
  tests/unit.sh both "tests/$name.cu" "$out/$name" >"$out/$name.log" 2>&1
  status=$?
  echo "$name: status $status: $(tail -n 1 "$out/$name.log")"
  runs=$((runs + 1)); if [ $status -eq 77 ]; then not_run=$((not_run + 1)); elif [ $status -ne 0 ]; then bad=$((bad + 1)); fi
done
for scene in given-twice renew-while-set renew-while-parked counter-not-above coin-after-wipe low-stream given-zero renew-zero; do
  "$out/coin_renew_gpu_test/coin_renew_gpu_test" "$scene" >"$out/scene-$scene.log" 2>&1
  status=$?
  said=$(grep -c '^coins: ' "$out/scene-$scene.log")
  echo "coin_renew_gpu_test $scene: status $status (wanted 2), lines that begin with 'coins: ': $said (wanted 1): $(tail -n 1 "$out/scene-$scene.log")"
  runs=$((runs + 1)); if [ $status -eq 2 ] && grep -q 'not run' "$out/scene-$scene.log"; then not_run=$((not_run + 1)); elif [ $status -ne 2 ] || [ "$said" -ne 1 ]; then bad=$((bad + 1)); fi
done
"$out/coin_fill_gpu_test/coin_fill_gpu_test" refused-launch >"$out/scene-refused-launch.log" 2>&1
status=$?
said=$(grep -c 'hiding/trees\.cuh' "$out/scene-refused-launch.log")
echo "coin_fill_gpu_test refused-launch: status $status (wanted 1), lines that name hiding/trees.cuh: $said (wanted 1): $(tail -n 1 "$out/scene-refused-launch.log")"
runs=$((runs + 1)); if [ $status -eq 2 ] && grep -q 'not run' "$out/scene-refused-launch.log"; then not_run=$((not_run + 1)); elif [ $status -ne 1 ] || [ "$said" -ne 1 ] || grep -q 'not run\|has returned' "$out/scene-refused-launch.log"; then bad=$((bad + 1)); fi
# A run that reports no GPU (exit status 2 and the words "not run"; 77 from tests/unit.sh) never
# counts as passed. If only some runs report it, something is wrong, and they count as failures.
if [ $not_run -eq $runs ]; then echo "coin_gpu_tests: not run: this machine has no GPU"; exit 77; fi
bad=$((bad + not_run))
if [ $bad -eq 0 ]; then echo "coin_gpu_tests: all hold"; else echo "coin_gpu_tests: $bad FAIL"; fi
[ $bad -eq 0 ]
