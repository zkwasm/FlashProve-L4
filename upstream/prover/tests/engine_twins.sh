#!/bin/bash
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.
#
# Compares toy-size proofs (short synthetic vectors, no model) made on two engines, implementations
# of the prover's heavy steps (blocks/engine/engine.h). blocks/toy/toy_run.cpp and
# blocks/toy/segments_run.cpp use the CPU engine; prover/engine/tests/toy_run_gpu.cu and
# prover/engine/tests/segments_run_gpu.cu use the GPU engine. Each program's second argument, 5,
# fixes the witness and the prover's private randomness: a program and its counterpart must print
# the same proof digest, and every run's verifier must accept. The comparison matters because the
# library's exact tests of hiding (blocks/tests/hiding/) run on the CPU engine.
#   prover/tests/engine_twins.sh build 1|2 <folder>   compiles the four programs; needs no GPU
#   prover/tests/engine_twins.sh run   1|2 <folder>   each pair at 0 and 20 bits of proof of work
# `build` passes 1|2 as BLK_FIELD; `run` only prints it. Only 2, the Goldilocks field, compiles:
# blocks/base.h rejects every other value, so `build 1` fails.
# `run` needs a GPU. It executes the programs in <folder>, so no other account may write there:
# `build` creates <folder> with mode 700 and rejects a name that exists; `run` accepts only the
# caller's own directory with that mode and the file that `build` leaves in it.
# Exit code 0: built, or 4 pairs equal and 8 runs accepted; 1: not; 2: bad usage, unusable nvcc or
# <folder>. run_suite.py checks the last line printed against a pattern in SUITE.md.
# If SIGHUP, SIGINT, SIGKILL or SIGTERM ends a GPU program, `run` prints "NO END: ..." and exits at
# once with code 1: it starts no further program on that GPU. It sets no time limit of its own.
# The test shows nothing beyond toy sizes, or for a witness or randomness other than argument 5's.
# Started by the runner; by hand: below a directory of your own, since the script does not check
# the directories above <folder>. (Runner: run_suite.py.)
set -uo pipefail
unset CDPATH
[ $# -eq 3 ] && { [ "$1" = build ] || [ "$1" = run ]; } && { [ "$2" = 1 ] || [ "$2" = 2 ]; } && [ -n "$3" ] || { echo "usage: tests/engine_twins.sh build|run 1|2 <folder>" >&2; exit 2; }
word=$1 field=$2
case $3 in /*) out=$3 ;; *) out=$PWD/$3 ;; esac # Made absolute before the cd below, so that a relative name counts from the caller's directory.
while [[ $out == */ || $out == */. ]]; do out=${out%/*}; done # Behind a closing "/" or "/." the test for a link below would look at the link's target.
case /$out/ in */-*|*/../*|*[!A-Za-z0-9_./-]*) echo "engine_twins.sh: the new folder's whole path (the current directory included) may hold letters, digits and _ . / - only, and no part of it may be \"..\" or begin with \"-\": name another: $out" >&2; exit 2 ;; esac # before anything is made: a path with a blank, "~", "*" or "$" would name another place when somebody retypes it unquoted
cd "$(dirname "$0")/../.." || exit 2
# The second mkdir fails if the name is taken, so `build` never reuses a directory. `build` leaves
# an empty file, made-by-engine_twins-build, in the new directory, and `run` requires that file, so
# that naming some other private directory by mistake runs nothing from it. The file guards against
# mistakes only: anyone who can write to a directory can create that file.
if [ "$word" = build ]; then
  prover/check_nvcc_version.sh || exit 2 # refuses an nvcc older than 13.2.86, before the directory is created, so that this refusal leaves nothing behind
  [ ! -e "$out" ] || { echo "engine_twins.sh: $out exists already; name a directory that does not exist yet" >&2; exit 2; }
  (umask 077 && mkdir -p -- "$(dirname -- "$out")") && mkdir -m 700 -- "$out" && : >"$out/made-by-engine_twins-build" || exit 2
elif [ ! -d "$out" ] || [ -L "$out" ] || [ ! -O "$out" ] || [[ $(stat -c %a -- "$out") != *700 ]] || [ ! -f "$out/made-by-engine_twins-build" ] || [ -L "$out/made-by-engine_twins-build" ]; then
  echo "engine_twins.sh: $out is not a directory that \"engine_twins.sh build\" made (a directory of your own, mode 700, holding the file made-by-engine_twins-build); name the directory that build made, or run build again" >&2; exit 2
fi
if [ "$word" = build ]; then
  mkdir -m 700 -- "$out/tmp" || exit 2
  export TMPDIR=$out/tmp # the compilers' temporary files stay in the new, closed directory, not in a shared one
  bad=0
  for p in toy_run segments_run; do
    g++ -std=c++17 -O2 -fno-ipa-modref -I blocks -DBLK_FIELD="$field" -DBLK_HASH=1 "blocks/toy/$p.cpp" -o "$out/$p" >"$out/build-$p.log" 2>&1 || bad=$((bad + 1))
    nvcc -std=c++17 -O2 -arch=sm_90 -Xcompiler -fno-ipa-modref -I blocks -I prover -DBLK_FIELD="$field" -DBLK_HASH=1 "prover/engine/tests/${p}_gpu.cu" -lcublas -o "$out/${p}_gpu" >"$out/build-${p}_gpu.log" 2>&1
    status=$?
    if [ $status -ne 0 ]; then rm -f "$out/${p}_gpu"; bad=$((bad + 1)); fi
  done
  if [ $bad -eq 0 ]; then echo "engine_twins: field $field: built"; else echo "engine_twins: field $field: $bad of 4 programs did not build (see $out/build-*.log)"; fi
  [ $bad -eq 0 ]; exit
fi
equal=0 yes=0
for p in toy_run segments_run; do
  for work in 0 20; do
    "$out/$p" $work 5 >"$out/$p.$work.cpu.log" 2>&1
    "$out/${p}_gpu" $work 5 >"$out/$p.$work.gpu.log" 2>&1; ended=$? # no timeout, on purpose: ending a program while its GPU kernel runs can take the whole machine out of use (SUITE.md, "Run the tests")
    case $ended in 129 | 130 | 137 | 143) # 128 + the signal: ended by SIGHUP, SIGINT, SIGKILL or SIGTERM. What such an end leaves on the GPU is not known, so no further program is started there
      echo "NO END: a signal ended ${p}_gpu, work $work (status $ended): nothing is shown on this run or on those behind it; what it printed: $out/$p.$work.gpu.log"
      exit 1 ;;
    esac # These four come from outside, mostly to a program that hangs. The program's own crash (134: abort(), 139: SIGSEGV) does not stop the run: the lines below judge it by what it printed
    c=$(grep -m 1 '^proof digest' "$out/$p.$work.cpu.log") g=$(grep -m 1 '^proof digest' "$out/$p.$work.gpu.log")
    yes=$((yes + $(grep -c '^verdict: ACCEPT' "$out/$p.$work.cpu.log") + $(grep -c '^verdict: ACCEPT' "$out/$p.$work.gpu.log")))
    if [ -n "$c" ] && [ "$c" = "$g" ]; then equal=$((equal + 1)); echo "$p, work $work: equal: $c"; else echo "$p, work $work: DIFFER: cpu «$c», gpu «$g»"; fi
  done
done
echo "engine_twins: field $field: $equal of 4 pairs of proofs equal, $yes of 8 runs accepted"
[ $equal -eq 4 ] && [ $yes -eq 8 ]
