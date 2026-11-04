#!/bin/bash
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.

# Started by the runner; by hand: below a directory of your own. (Runner: run_suite.py. The work folder goes there.)
# Plays the test program frame_as_the_verifier_reads (prover/tests/frame_as_the_verifier_reads.cu;
# target frame_as_the_verifier_reads of prover/build.sh) on each given configuration: cp_prove holds
# no verifier code, so this is where what the prover's code derives from a commitment's public files
# meets what the verifier's code derives from them.
# Each configuration gets a commitment on made-up weights, then two calls of the tool. Without a
# switch it must end with status 0 and print the line "frame_as_the_verifier_reads: the same". With
# -turn-one-bound it must end with status 1 and print the line that names the first range check as
# differing, which shows that the comparison of the range checks sees one changed bound. A status
# alone says nothing: a failed set-up and files that the verifier's code refuses also end with 1.
# Not tested here: that the comparison of the parts compared byte for byte can fail; the other
# numbers of a range check; that frame.range_checks() is the list on which cp_prove tested its
# conditions of hiding.
# usage: frame_readings.sh <folder with cp_commit and cp_verify> <folder with frame_as_the_verifier_reads>
#        <new work folder> <configuration: a folder with policy.txt, KL.bin, layout.txt> ...
# Needs a test build (test seed, made-up weights) and one otherwise idle H100.
# Last line and exit status: "frame_readings: <n> configurations, all as expected", 0; else
# "frame_readings: FAILED", 1; status 2 for a wrong call. An existing work folder is left untouched:
# status 2. Status 2 also unless the folder's absolute path has only letters, digits and _ . / - and
# no part that is ".." or begins with "-".
set -u
unset CDPATH
B=${1:?the folder of cp_commit} F=${2:?the folder of frame_as_the_verifier_reads} W=${3:?a new work folder}
shift 3
[ $# -ge 1 ] || { echo "frame_readings: name at least one configuration"; exit 2; }
case $W in /*) full=$W ;; *) full=$PWD/$W ;; esac; case /$full/ in */-*|*/../*|*[!A-Za-z0-9_./-]*) echo "frame_readings.sh: the work folder's whole path (the current directory included) may hold letters, digits" \
  "and _ . / - only, and no part of it may be \"..\" or begin with \"-\": name another" >&2; exit 2 ;; esac
if [ -e "$W" ]; then echo "frame_readings: the work folder $W exists: name a new one"; exit 2; fi
mkdir -p -- "$(dirname -- "$W")" && mkdir -- "$W" || exit 2 # the mkdir without -p fails if anything has appeared at the name since the -e test on it above
full=$(cd -- "$W" && pwd) && [ "$full" -ef "$W" ] && W=$full || { echo "frame_readings.sh: cd does not lead to the folder just made: name another" >&2; exit 2; }
TOOL=$F/frame_as_the_verifier_reads
SAME='frame_as_the_verifier_reads: the same'
TURNED='^frame_as_the_verifier_reads: the two readings of the public files differ: the range checks handed to the conditions of hiding: check 0 of .* here; .* in the verifier$'
bad=0 n=0
no() { echo "NOT AS EXPECTED: $1: $2"; bad=1; }
for K in "$@"; do
  n=$((n + 1))
  D=$W/$n
  mkdir -p -- "$D/config" && cp -- "$K/policy.txt" "$K/KL.bin" "$K/layout.txt" "$D/config/" || { no "$K" "its files cannot be copied"; continue; }
  bits=$(awk '$1 == "sec" { print $2 }' "$D/config/policy.txt")
  [ -n "$bits" ] || { no "$K" "its policy.txt has no line sec"; continue; }
  fewer=()
  if [ "$bits" != 128 ]; then fewer=(-test-bits "$bits"); fi
  head -c 32 /dev/urandom >"$D/seed.test" || exit 2
  ${GPU_RUN:-} "$B/cp_commit" -weights made-up-blocks -config "$D/config" -M 1 -ivk "$D/ivk" -isk "$D/isk" -test-seed "$D/seed.test" ${fewer[@]+"${fewer[@]}"} >"$D/cp_commit.log" 2>&1 || { no "$K" "cp_commit: status $?"; continue; }
  mkdir -m 700 -- "$D/plain" "$D/turned" || exit 2
  ${GPU_RUN:-} "$TOOL" "$bits" -isk "$D/isk" -spool "$D/plain" >"$D/plain.log" 2>&1; got=$?
  [ "$got" = 0 ] && grep -a -q -x -F -- "$SAME" "$D/plain.log" || no "$K" "the plain call: status $got, last line: $(tail -n 1 "$D/plain.log" | cut -c1-200)"
  ${GPU_RUN:-} "$TOOL" "$bits" -turn-one-bound -isk "$D/isk" -spool "$D/turned" >"$D/turned.log" 2>&1; got=$?
  [ "$got" = 1 ] && grep -a -q -E -- "$TURNED" "$D/turned.log" || no "$K" "the call that turns one bound: status $got, last line: $(tail -n 1 "$D/turned.log" | cut -c1-200)"
done
if [ $bad = 0 ]; then echo "frame_readings: $n configurations, all as expected"; else echo "frame_readings: FAILED"; fi
exit $bad
