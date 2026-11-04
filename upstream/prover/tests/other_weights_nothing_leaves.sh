#!/bin/bash
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

#
# Terms: docs/glossary.md.
#
# Checks that cp_prove, started on other weights with a leaf table's true secret files, releases
# only an empty <name>.failed for each request.
# A leaf table is a masked encoding of all the weights under a Merkle tree. Its directory
# isk/leaf/<k>/ has the nodes of one level of the tree (hcnodes.bin) and 32 bytes (fresh.bin) from
# which, with the seed, the masks are derived. So a run on other weights computes the true masks
# over other data. That harms only if something of it leaves the process. What must stop it: every
# column to be opened is recomputed and checked against the stored nodes before any of it is
# written [hc-own-node]; and a batch that fails releases only empty `.failed` files.
#
# The script copies a commitment's secret key directory (isk/) twice. In the copy `other` it changes
# only isk.txt's line `weights`, which no public configuration file has, to the other source of
# synthetic weights (`generated` or `made-up-blocks`). It runs cp_prove on each copy, on the same
# requests. `same` must succeed, with each request's <name>.proof and token file <name>.y, which
# cp_verify must accept (on a finished run's directory, `<run>` below, nothing verifies them).
# `other` must fail with a line that begins "hc-own-node:" and leave no staging directory out/.tmp
# and no new file but an empty <name>.failed for each request. Of the copied files only the column
# count isk/leaf/<k>/count.bin may change (how is not checked): a batch is counted when it begins.
#
# The script looks only at the files that a run leaves below `same/` or `other/`: not at memory, at
# what else cp_prove prints (test builds print more than production builds), or at unlinked data on
# the disk. While the table is opened, cp_prove stages each request's token file, proof header and
# first hash chain under out/.tmp; a batch that fails unlinks them and does not overwrite them. The
# first chain uses the request's own fresh private randomness.
# Everything here runs under a test seed and on synthetic weights: nothing is hidden.
# Started by the runner; by hand: below a directory of your own. (Runner: run_suite.py.)
# usage: other_weights_nothing_leaves.sh <programs> <run> <new work directory> [cp_prove arguments]
#   <programs>: the directory of a test build's cp_prove and cp_verify
#   <run>: a finished run's directory, with isk/ and in/; or `kit:<directory>`, an example
#          configuration's directory: the script then makes the commitment
#          (cp_commit must be in <programs>) and one request itself
#   <new work directory>: must not exist. Its absolute path may have only letters, digits and
#          _ . / -, and no part of it may be ".." or begin with "-". The script makes it with mode
#          700, and missing directories above it under umask 077. It does not check the existing
#          directories above it: name one below directories in which only you can write
#   [cp_prove arguments]: by default none
# Exit status 0: all checks passed; 1: a check failed; 2: nothing was tested.
# GPU_RUN in the environment: a command prefix for the programs that use the GPU.
# Messages say `folder` for directory and `made-up` for synthetic.
set -u
unset CDPATH
made_here() { # <a folder that this script has just made>: it must be this account's, closed to others (mode 700; 2700 below a set-group-ID directory) and empty
  # One cd, then three tests of ".", so that all three look at the same directory. If stat or ls
  # fails, the check fails too. With -P, cd follows symbolic links before it applies "..", as mkdir
  # does. The trailing slash keeps cd from taking a directory named "-" for the previous directory.
  (cd -P -- "$1/" && [ -O . ] && [[ $(stat -c %a .) == *700 ]] && list=$(ls -A) && [ -z "$list" ]) && return
  echo "other_weights_nothing_leaves.sh: $1 is not the folder that was just made: another was put in its place: name a new folder below folders in which only you can write" >&2
  return 1
}
B=$1; W0=$2; W=$3; shift 3; MORE=$*; made=no
[ -e "$W" ] && { echo "the work folder $W exists: name a new one"; exit 2; }
case $W in /*) full=$W ;; *) full=$PWD/$W ;; esac
case /$full/ in */-*|*/../*|*[!A-Za-z0-9_./-]*) echo "other_weights_nothing_leaves.sh: the new folder's whole path (the current directory included) may hold letters, digits and _ . / - only, and no part of it may be \"..\" or begin with \"-\": name another: $full" >&2; exit 2 ;; esac # before anything is made: a path with a blank, "~", "*" or "$" would name another place when somebody retypes it unquoted
case $W0 in kit:*)
  kit=${W0#kit:}; W0=$W/made; made=yes
  [ -x "$B/cp_commit" ] && [ -f "$kit/policy.txt" ] && (umask 077 && mkdir -p -- "$(dirname -- "$W")") && mkdir -m 700 -- "$W" && made_here "$W" && mkdir -p -- "$W0/in" || { echo "usage: see the head"; exit 2; } # the mkdir without -p fails if anything has appeared at the name since the -e test on it above
  # TMPDIR becomes a directory inside the one just made, which is closed to other accounts: bash
  # may write a here-document to a file in TMPDIR and open it again by name (before version 5.1 it
  # always does), and an account that can replace files there could give python3 another program.
  export TMPDIR=$full/tmp; mkdir -- "$TMPDIR" || exit 2
  python3 -I -c "import sys; sys.stdout.buffer.write(bytes(range(1, 33)))" > "$W0/seed"
  python3 -I - "$kit/policy.txt" "$W0/in/example.req" <<'END'
import struct, sys
word = dict(line.split() for line in open(sys.argv[1]) if len(line.split()) == 2)
T, V = 1 << int(word["lt"]), int(word["V"])
open(sys.argv[2], "wb").write(struct.pack("<%dI" % (T + 2), *[(2654435761 * (t + 1) >> 7) % V for t in range(T)], T - 1, T))
END
  ${GPU_RUN:-} "$B/cp_commit" -weights made-up-blocks -config "$kit" -M 1 -ivk "$W0/ivk" -isk "$W0/isk" -test-seed "$W0/seed" > "$W0/cp_commit.log" 2>&1 || { echo "cp_commit failed: $W0/cp_commit.log"; exit 2; } ;;
esac
[ -x "$B/cp_prove" ] && [ -x "$B/cp_verify" ] && [ -f "$W0/isk/isk.txt" ] && [ -d "$W0/in" ] || { echo "usage: see the head"; exit 2; }
grep -q '^seed test$' "$W0/isk/isk.txt" || { echo "this test runs on a commitment under a test's seed alone (isk.txt's line: seed test)"; exit 2; }
was=$(sed -n 's/^weights //p' "$W0/isk/isk.txt")
case $was in made-up-blocks) now=generated ;; generated) now=made-up-blocks ;; *) echo "this test runs on made-up weights alone (isk.txt's line weights: $was)"; exit 2 ;; esac
[ $made = yes ] || { (umask 077 && mkdir -p -- "$(dirname -- "$W")") && mkdir -m 700 -- "$W" && made_here "$W"; } || exit 2 # the work folder: made above if an example configuration was named, else here, its last part again by a plain mkdir
for c in same other; do mkdir "$W/$c" && cp -a "$W0/isk" "$W/$c/isk" && cp -a "$W0/in" "$W/$c/in" || exit 2; done
sed -i "s/^weights $was\$/weights $now/" "$W/other/isk/isk.txt"
[ "$(diff "$W/same/isk/isk.txt" "$W/other/isk/isk.txt" | grep -c '^[<>]')" = 2 ] && [ "$(diff -rq "$W/same/isk" "$W/other/isk" | wc -l)" = 1 ] || { echo "the two copies do not differ in that one line alone"; exit 2; }
bad=0; no() { echo "OTHERWISE: $*"; bad=$((bad + 1)); }
for c in same other; do
  (cd -- "$W/$c" && find . -type f | LC_ALL=C sort) > "$W/$c.before"
  ${GPU_RUN:-} "$B/cp_prove" -isk "$W/$c/isk" -in "$W/$c/in" -out "$W/$c/out" $MORE > "$W/$c.log" 2>&1; rc=$?
  (cd -- "$W/$c" && find . -type f | LC_ALL=C sort) > "$W/$c.after"
  new=$(comm -13 "$W/$c.before" "$W/$c.after" | tr '\n' ' '); gone=$(comm -23 "$W/$c.before" "$W/$c.after" | tr '\n' ' ')
  moved=$(for f in $(comm -12 "$W/$c.before" "$W/$c.after"); do cmp -s "$W/$c/$f" "$W0/$f" || printf '%s ' "$f"; done)
  echo "$c: status $rc; new files: ${new:-none}; gone: ${gone:-none}; moved against the commitment's: ${moved:-none}"
  names=$(cd -- "$W/$c/in" && ls | sed 's/\.req$//')
  if [ $c = same ]; then
    [ $rc = 0 ] || no "same: status $rc"
    for n in $names; do [ -s "$W/$c/out/$n.proof" ] && [ -s "$W/$c/out/$n.y" ] || no "same: no proof or no reply for $n"; done
    if [ $made = yes ]; then for n in $names; do "$B/cp_verify" "$kit" "$W0/ivk" "$W/$c/in/$n.req" "$W/$c/out/$n.y" "$W/$c/out/$n.proof" -id "$(sed -n "s/^commitment ID: //p" "$W0/cp_commit.log" | tr -d ' ')" >"$W/$c.verify.log" 2>&1 || no "same: cp_verify does not accept the response to $n: $(tail -n 1 "$W/$c.verify.log" | cut -c1-160)"; done; fi
    grep -q '^hc-own-node:' "$W/$c.log" && no "same: the own test has spoken"
  else
    [ $rc != 0 ] || no "other: status 0"
    grep -q '^hc-own-node:' "$W/$c.log" || no "other: no line of the opening's own test: the run ended for another reason: $(grep -a -E '^refused|internal error|^ended' "$W/$c.log" | head -2 | cut -c1-160)"
    want=$(for n in $names; do printf './out/%s.failed ' "$n"; done)
    [ "$new" = "$want" ] || no "other: the new files are not the requests' empty marks alone"
    for n in $names; do [ -f "$W/$c/out/$n.failed" ] && [ ! -s "$W/$c/out/$n.failed" ] || no "other: $n.failed is missing or holds bytes"; done
    [ -e "$W/$c/out/.tmp" ] && no "other: the staging folder is still there"
  fi
  [ -z "$gone" ] || no "$c: files are gone"
  for f in $moved; do case $f in ./isk/leaf/*/count.bin) ;; ./isk/isk.txt) [ $c = other ] || no "same: isk.txt moved" ;; *) no "$c: $f has moved" ;; esac; done
done
[ $bad = 0 ] && echo "other_weights_nothing_leaves.sh: as foretold" || echo "other_weights_nothing_leaves.sh: $bad things otherwise"
exit $((bad != 0))
