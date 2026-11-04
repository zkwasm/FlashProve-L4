#!/bin/bash
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.
# Makes the test fixtures of other_parameters_test.py and other tests. A fixture, worlds/<name>, is
# one honest run (commit, one request, prove): worlds/true under the given configuration, and one
# under each of its variants, which other_parameters_kits.py writes into kits/ ("sisters" in the
# output). All have the same synthetic weights, request and test seed, so nothing in them is hidden.
# usage: other_parameters_worlds.sh <directory of cp_commit, cp_prove, cp_verify>
#        <configuration (`KIT`): policy.txt, KL.bin> <directory to make>
# cp_commit must be a test build (`prover/build.sh tests`), for `-test-seed` and `-test-bits`, and
# cp_prove of the same build: a production cp_prove refuses a commitment under a test seed.
# <directory to make> ("folder" in messages) must not exist; the script removes
# nothing.
# CARD, if set: a command prefix for each program that uses the GPU (a lock, a timeout).
# ONLY, if set: the variants to make besides worlds/true, separated by blanks.
# A configuration under which a program fails is left out, worlds/true too: the script prints the
# program and status, renames worlds/<name> to worlds/.left-<name>, and still exits with
# status 0. It exits with 1 if any fixture's cp_commit.log does not have exactly one commitment ID
# line or one of worlds/true's further steps fails, and with 2 if it rejects a path or a variant's
# name or other_parameters_kits.py fails.
# Started by the runner; by hand: below a directory of your own. (Runner: run_suite.py. The work folder goes there.)
set -u
unset CDPATH
made_here() { # <a folder that this script has just made>: it must be this account's, closed to others (mode 700; 2700 below a set-group-ID directory) and empty
  # One cd, then three tests of ".", so that all three see the same directory; the check also
  # fails if stat or ls fails. With -P, cd resolves ".." as mkdir does. The trailing slash keeps
  # cd from reading a directory named "-" as the previous directory.
  (cd -P -- "$1/" && [ -O . ] && [[ $(stat -c %a .) == *700 ]] && list=$(ls -A) && [ -z "$list" ]) && return
  echo "other_parameters_worlds.sh: $1 is not the folder that was just made: another was put in its place: name a new folder below folders in which only you can write" >&2
  return 1
}
B=${1:?programs}; KIT=${2:?kit}; O=${3:?folder to make}; HERE=$(cd -- "$(dirname -- "$0")" && pwd) || exit 2; CARD=${CARD:-}
whole=; for word in "$B" "$KIT" "$O"; do [[ $word == /* ]] && whole+=$word || whole+=$PWD/$word; done
case "$whole$HERE" in *[!A-Za-z0-9_./-]*) # these names go unquoted, and into program text: other signs would send its calls elsewhere
  echo "other_parameters_worlds.sh: a folder's path may hold letters, digits and / . _ - only" >&2
  exit 2 ;;
esac
[ -e "$O" ] && { echo "other_parameters_worlds: $O exists"; exit 2; }
[[ $O == /* ]] || O=$PWD/$O # made absolute, so that no command reads a name that begins with a dash as an option
mkdir -p -- "$(dirname -- "$O")" && mkdir -m 700 -- "$O" && made_here "$O" && mkdir -p $O/kits/true $O/worlds || exit 2 # the second mkdir has no -p, so it fails if something has taken the name since the test above
# TMPDIR becomes a directory inside the one just made, which is closed to other accounts: bash may
# write a here-document to a file in TMPDIR and open it again by name (before version 5.1 it always
# does), and an account that can replace files there could give python3 another program.
export TMPDIR=$O/tmp; mkdir -- "$TMPDIR" || exit 2
(umask 077 && head -c 32 /dev/urandom > $O/seed.test)
python3 -B $HERE/other_parameters_kits.py $KIT $O/kits > $O/names.txt || exit 2
cp $KIT/policy.txt $KIT/KL.bin $O/kits/true/
for n in true $(cat $O/names.txt); do
  [[ $n != *[!abcdefghijklmnopqrstuvwxyz-]* ]] || { echo "other_parameters_worlds: a name in $O/names.txt holds more than small letters and dashes: $n"; exit 2; } # it becomes part of paths below $O
  [ -z "${ONLY:-}" ] || [ $n = true ] || case " $ONLY " in *" $n "*) ;; *) continue ;; esac
  W=$O/worlds/$n; mkdir -p $W/config $W/in; cp $O/kits/$n/policy.txt $O/kits/$n/KL.bin $W/config/
  message=$(awk '$1 == "group" && $2 == "link" { print $5 }' $W/config/policy.txt); target=$(awk '$1 == "sec" { print $2 }' $W/config/policy.txt)
  left() { echo "$n | LEFT OUT: $1 | $(grep -a -h '^NO:\|^key:\|^entry:\|refused\|internal' $2 | sort -u | head -n 4 | tr '\n' ' ' | cut -c1-260)"; mv $W $O/worlds/.left-$n; }
  $B/cp_verify -plan arrays $W/config $target > $W/arrays.txt 2>&1; s=$?; [ $s = 4 ] || { left "the verifier's planning call: status $s" $W/arrays.txt; continue; }
  python3 -B $HERE/../../prover/tools/pack_layout.py --arrays $W/arrays.txt $message 4 > $W/config/layout.txt 2> $W/pack.err; s=$?; [ $s = 0 ] || { left "the layout tool: status $s" $W/pack.err; continue; }
  copies=1; [ $n = true ] && copies=2 # worlds/true: two leaf tables, for entries_and_parts_test.py
  $CARD $B/cp_commit -weights made-up-blocks -config $W/config -M $copies -ivk $W/ivk -isk $W/isk -test-bits 0 -test-seed $O/seed.test > $W/cp_commit.log 2>&1; s=$?; [ $s = 0 ] || { left "cp_commit: status $s" $W/cp_commit.log; continue; }
  sed -n "s/^commitment ID: //p" $W/cp_commit.log | tr -d ' ' > $W/id.txt
  # A request file is T token ids, then r and e; its proof is about tokens r + 1 to e, where token T
  # is the next token.
  python3 -I -B - $W/config/policy.txt $W/in/example.req <<'END'
import struct, sys
word = dict(line.split() for line in open(sys.argv[1]) if len(line.split()) == 2)
T, V = 1 << int(word["lt"]), int(word["V"])
open(sys.argv[2], "wb").write(struct.pack("<%dI" % (T + 2), *[(2654435761 * (t + 1) >> 7) % V for t in range(T)], T - 1, T))
END
  id=$(cat $W/id.txt); [[ ${#id} = 64 && $id != *[!0123456789abcdef]* ]] || { echo "other_parameters_worlds: $n: $W/cp_commit.log does not hold exactly one line with the commitment's ID (64 hex digits)"; exit 1; } # SUITE.md's rows put id.txt's text on a command line, unquoted: it must be one word
  $CARD $B/cp_prove -isk $W/isk -in $W/in -out $W/out > $W/cp_prove.log 2>&1; s=$?; [ $s = 0 ] || { left "cp_prove: status $s" $W/cp_prove.log; continue; }
  # 14: NOT-A-VERIFICATION, the best ending at 0 requested bits: no check that ran has failed
  $B/cp_verify $W/config $W/ivk $W/in/example.req $W/out/example.y $W/out/example.proof -id $id -bits 0 > $W/cp_verify.log 2>&1; s=$?; [ $s = 14 ] || { left "cp_verify: status $s" $W/cp_verify.log; continue; }
  if [ $n = true ]; then # the test's control: the same configuration and commitment, a proof from a second run of the prover
    C=$O/worlds/same-configuration; mkdir -p $C; cp -r $W/config $W/ivk $W/isk $W/in $W/id.txt $C/
    $CARD $B/cp_prove -isk $C/isk -in $C/in -out $C/out > $C/cp_prove.log 2>&1 || { echo "other_parameters_worlds: the control's proof is not made"; exit 1; }
    # For what_a_yes_covers_test.py, two requests about token T - 1: in-two/two.req is example.req's
    # first T - 1 tokens, then the integer model's top choice c, with (r, e) = (T - 2, T);
    # in-one/one.req has the same tokens with (T - 2, T - 1). To find c, cp_prove tries the token
    # ids in c's place, 64 to a batch, until one yields a proof: under the exact-match policy only
    # the top choice gets one. The search, on a copy of the secret key directory (`isk`), stays in
    # search/.
    X=$O/search; mkdir -p $X; cp -r $W/isk $X/isk; found=; asked=0; V=$(awk '$1 == "V" { print $2 }' $W/config/policy.txt)
    [[ $V =~ ^[0-9]{1,9}$ ]] || { echo "other_parameters_worlds: the line V of $W/config/policy.txt holds no number"; exit 1; } # bash runs a command that stands in a word with which it computes (x[$(command)]): only a number gets that far
    for ((b = 0; b < V; b += 64)); do
      [ -z "$found" ] || break
      mkdir -p $X/in.$b
      python3 -I -B - $W/in/example.req $X/in.$b $b $V <<'END'
import struct, sys
w = open(sys.argv[1], "rb").read(); T = len(w) // 4 - 2; tokens = list(struct.unpack("<%dI" % T, w[: 4 * T])); b, V = int(sys.argv[3]), int(sys.argv[4])
for c in range(b, min(b + 64, V)): open("%s/c%04d.req" % (sys.argv[2], c), "wb").write(struct.pack("<%dI" % (T + 2), *tokens[: T - 1], c, T - 2, T))
END
      $CARD $B/cp_prove -isk $X/isk -in $X/in.$b -out $X/out.$b > $X/cp_prove.$b.log 2>&1; s=$?
      asked=$((asked + $(ls $X/in.$b | wc -l))); found=$(cd $X/out.$b 2>/dev/null && ls | sed -n 's/^c0*\([0-9][0-9]*\)\.proof$/\1/p' | tr '\n' ' ')
      echo "true | the search for a covered token: candidates from $b on: cp_prove's status $s; proofs: $found; marks failed: $(ls $X/out.$b 2>/dev/null | grep -c '\.failed$')"
    done
    set -- $found; [ $# = 1 ] || { echo "other_parameters_worlds: of $asked candidates for the token at T - 1 not exactly one got a proof: $found"; exit 1; }
    mkdir -p $W/in-two $W/in-one
    python3 -I -B - $W/in/example.req $1 $W/in-two/two.req $W/in-one/one.req <<'END'
import struct, sys
w = open(sys.argv[1], "rb").read(); T = len(w) // 4 - 2; tokens = list(struct.unpack("<%dI" % T, w[: 4 * T]))[: T - 1] + [int(sys.argv[2])]
open(sys.argv[3], "wb").write(struct.pack("<%dI" % (T + 2), *tokens, T - 2, T)); open(sys.argv[4], "wb").write(struct.pack("<%dI" % (T + 2), *tokens, T - 2, T - 1))
END
    for k in two one; do
      $CARD $B/cp_prove -isk $W/isk -in $W/in-$k -out $W/out-$k > $W/cp_prove.$k.log 2>&1 && [ -s $W/out-$k/$k.proof ] || { echo "other_parameters_worlds: the proof of $k.req is not made"; exit 1; }
    done
    echo "true | two requests that cover the token at T - 1 (the model's choice: $1, found among $asked asked) | proofs $(stat -c%s $W/out-two/two.proof) and $(stat -c%s $W/out-one/one.proof) bytes | files in out-one: $(ls $W/out-one | tr '\n' ' ')"
  fi
  echo "$n | made | key $(stat -c%s $W/ivk/key.bin) entry $(stat -c%s $W/ivk/leaf/0.bin) proof $(stat -c%s $W/out/example.proof) bytes | reply $(od -An -tu4 $W/out/example.y | tr -d ' ') | ID $(cut -c1-16 $W/id.txt).."
done
echo "other_parameters_worlds: $(ls $O/worlds | wc -l) worlds made, $(ls -A $O/worlds | grep -c '^\.left-') sisters left out"
