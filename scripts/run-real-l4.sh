#!/usr/bin/env bash
set -euo pipefail
umask 077
test "$(id -u)" != 0
root=$(pwd -P)
build="$root/upstream/prover/build/hc"
config=${1:-"$root/runs/real-config-tu9cL1"}
work=${2:-"$root/real-l4-$(date -u +%Y%m%dT%H%M%SZ)"}
mkdir "$work"
mkdir "$work/in" "$work/home" "$work/tmp"
cp "$config/example.req" "$work/in/example.req"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "$work/hardware.txt"
set +e
"$build/cp_commit" -weights "$root/models/stories260K/flashprove32" \
  -config "$config" -M 1 -ivk "$work/ivk" -isk "$work/isk" \
  -worst-case-table "$work/worst-case.txt" > "$work/commit.log" 2>&1
status=$?
echo "$status" > "$work/commit.exit"
set -e
test "$status" = 0
id=$(sed -n 's/^commitment ID: //p' "$work/commit.log" | tr -d ' ')
[[ "$id" =~ ^[0-9a-f]{64}$ ]]
printf '%s\n' "$id" > "$work/trusted-ID.txt"
set +e
env HOME="$work/home" TMPDIR="$work/tmp" "$build/cp_prove" \
  -isk "$work/isk" -in "$work/in" -out "$work/out" > "$work/prove.log" 2>&1
status=$?
echo "$status" > "$work/prove.exit"
set -e
test "$status" = 0
test -f "$work/out/example.proof"
set +e
"$build/cp_verify" "$config" "$work/ivk" "$work/in/example.req" \
  "$work/out/example.y" "$work/out/example.proof" -id "$id" > "$work/verify.log" 2>&1
status=$?
echo "$status" > "$work/verify.exit"
set -e
test "$status" = 0
test "$(tail -1 "$work/verify.log")" = YES
echo 'FULL REAL MODEL PROOF VERIFIED; NEGATIVE TESTS STILL REQUIRED'
