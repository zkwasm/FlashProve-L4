#!/bin/bash
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.

# Checks prover/hiding/coins_known_f2.h ("the table") against prover/tests/coins_known_maker.cpp
# ("the maker"), which generates it. When the programs cp_prove and cp_commit start,
# hz::coins_own_test() (prover/hiding/coins_own.cuh) tests a generator of private randomness
# ("coins" in identifiers) against the table's known answers; a failure is reported as
# `own-test-coins`. CPU only; needs g++ and python3.
# The option -fno-ipa-modref in the g++ call avoids a bug in g++ 13.3.0, reproduced by
# verifier/tests/compiler_pass_control.sh.
#  1. The maker's output must equal the table byte for byte. This catches a table edited by hand, or
#     not regenerated after a change to the headers included from prover/tests/coins_second_text/,
#     a second implementation of the generator.
#  2. Python's BLAKE2b-512 of four of the maker's lists of values (`cells`) must match their digests
#     in the table: this checks the maker's BLAKE2b code.
#  3. Negative controls: no match with one bit of a list flipped, or with another list's digest.
# Nothing here runs the programs' generator, or shows that the maker's keys and values are right.
# usage: coins_known_test.sh <new folder for the program> [2]    (2, the field's number: optional)
# The script creates this directory, which must not exist yet, and, under umask 077, any missing
# directory above it; it replaces and deletes nothing.
# Started by the runner; by hand: below a directory of your own, since the script does not check
# who can write in the directories above the new one. (Runner: run_suite.py.)
# Exit code 0: all checks pass; 1: a check fails; 2: bad usage, or the set-up or the build fails.
set -uo pipefail
unset CDPATH
made_here() { # <a folder that this script has just made>: it must be this account's, closed to others (mode 700; 2700 below a set-group-ID directory) and empty
  # One cd, then three tests of ".", so that all three see the same directory; the check also fails
  # if stat or ls fails. With -P, cd resolves symbolic links before "..", as mkdir does. The slash
  # after $1 keeps a directory named "-" from being read as the previous directory.
  (cd -P -- "$1/" && [ -O . ] && [[ $(stat -c %a .) == *700 ]] && list=$(ls -A) && [ -z "$list" ]) && return
  echo "coins_known_test.sh: $1 is not the folder that was just made: another was put in its place: name a new folder below folders in which only you can write" >&2
  return 1
}
export LC_ALL=C
if [[ $# -lt 1 || $# -gt 2 || -z $1 || ${2:-2} != 2 ]]; then
  echo "usage: coins_known_test.sh <new folder for the program> [2]     (2 is the field's number, the only one; it may be left out)" >&2
  exit 2
fi
if [[ -e $1 ]]; then
  echo "coins_known_test.sh: $1 exists already; name a directory that does not exist yet" >&2
  exit 2
fi
case $1 in /*) out=$1 ;; *) out=$PWD/$1 ;; esac # Made absolute before the cd below, so that a relative name counts from the caller's directory.
case /$out/ in */-*|*/../*|*[!A-Za-z0-9_./-]*) echo "coins_known_test.sh: the new folder's whole path (the current directory included) may hold letters, digits and _ . / - only, and no part of it may be \"..\" or begin with \"-\": name another: $out" >&2; exit 2 ;; esac # before anything is made: a path with a blank, "~", "*" or "$" would name another place when somebody retypes it unquoted
(umask 077 && mkdir -p -- "$(dirname -- "$out")") && mkdir -m 700 -- "$out" && made_here "$out" || exit 2 # A plain mkdir fails if the name is taken, so the table that is printed into this directory cannot replace hiding/coins_known_f2.h or any other file.
cd "$(dirname "$0")/.." || exit 2
field=2 table=hiding/coins_known_f2.h
export table
if ! g++ -std=c++17 -O2 -fno-ipa-modref -Wall -Wextra -DBLK_FIELD=$field -o "$out/coins_known_maker" tests/coins_known_maker.cpp >"$out/coins_known_maker.build.log" 2>&1; then
  echo "coins_known_test: the maker does not build: see $out/coins_known_maker.build.log"
  exit 2
fi
failed=0
if ! "$out/coins_known_maker" >"$out/coins_known.h"; then
  echo "FAILED: the maker prints no table"
  failed=1
elif ! cmp -s "$out/coins_known.h" "$table"; then
  echo "FAILED: $table is not what tests/coins_known_maker.cpp prints: compare it with $out/coins_known.h"
  failed=1
fi
# look <fill|long> <context> <group> <0, or 1 to flip a bit> <context2> <group2>: prints "found" if
# the digest of list [context][group] is entry [context2][group2] of the table's COINS_KNOWN_FILL
# or COINS_KNOWN_LONG, otherwise "missing" (nothing on an error). A digest: the first 32 bytes of
# BLAKE2b-512 of the values, 8 bytes each, little-endian.
look() {
  "$out/coins_known_maker" cells "$1" "$2" "$3" | python3 -I -B -c '
import hashlib, os, re, struct, sys
name, groups = ("COINS_KNOWN_FILL", 3) if sys.argv[1] == "fill" else ("COINS_KNOWN_LONG", 5)
cells = [int(line, 16) for line in sys.stdin]
cells[len(cells) // 2] ^= int(sys.argv[2]) << 17
digest = hashlib.blake2b(b"".join(struct.pack("<Q", cell) for cell in cells)).digest()[:32]
row = ", ".join("0x%08xu" % word for word in struct.unpack("<8I", digest))
array = open(os.environ["table"]).read().split("constexpr uint32_t " + name + "[")[1].split("\n};")[0]
rows = re.findall(r"\{(0x[0-9a-f]{8}u(?:, 0x[0-9a-f]{8}u){7})\}", array)
print("found" if cells and rows[groups * int(sys.argv[3]) + int(sys.argv[4])] == row else "missing")
' "$1" "$4" "$5" "$6"
}
for list in "fill 0 0" "fill 11 2" "long 0 3" "long 3 4"; do
  place=${list#* } # the list's own context and group
  # shellcheck disable=SC2086 # three words and two
  if [[ $(look $list 0 $place) != found ]]; then
    echo "FAILED: python's BLAKE2b-512 of the list \"$list\" is not at the list's place in the table"
    failed=1
  fi
  # shellcheck disable=SC2086
  if [[ $(look $list 1 $place) != missing ]]; then
    echo "FAILED: the control: the list \"$list\" with one bit flipped still has the digest at its place in the table"
    failed=1
  fi
  # shellcheck disable=SC2086
  if [[ $(look $list 0 1 1) != missing ]]; then
    echo "FAILED: the control: the digest of the list \"$list\" is taken at another list's place (context 1, group 1)"
    failed=1
  fi
done
if [[ $failed -ne 0 ]]; then
  echo "coins_known_test: fails"
  exit 1
fi
echo "coins_known_test: passes"
