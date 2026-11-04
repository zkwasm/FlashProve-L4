#!/bin/bash
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.
#
# Tests the audit (tools/audit/README.md), which recomputes random values in published files,
# on files that this script makes: a small test configuration (`kit`), synthetic weights
# (`made-up-blocks`), its own test seed and test randomness (`coins` in names and options).
# usage: play.sh <folder with cp_commit, cp_prove, cp_verify> <field: 2> <new work folder>
# <field>: must be 2 (p = 2^64 - 2^32 + 1).
# <new work folder>: its parent must exist; no other account may be able to change a place above
# it. The programs must be a test build (`prover/build.sh tests`). About ten minutes, three of them
# on the GPU, not counting the second of step 2's two builds of the audit's helper programs (two to
# three minutes). Each result line ends in "as expected" or says what happened instead.
# Needs one H100, g++ and python3 with numpy (tools/audit/README.md, "Try it"), and `flock`.
# Destroyed, all in the work directory (`folder` in usage and messages): in steps 2 and 8 an audit
# that passes with --erase-kept-files destroys the files kept by -keep-for-the-audit (three, then
# two); in step 9 every verdict (YES, NO, NOT WHOLE) with that word erases `audit/batch.fresh`,
# which this script wrote, and nothing else; if that file has a second hard link, it stays.
# The script itself removes only files that it made, each by name. It sends no signal; cp_prove,
# which it starts, kills only the process groups of its own cp_verify runs.
# Exit status 0: all as expected; 1: something else, or a missing argument; 2: bad usage, or the
# set-up failed. The work directory holds a test's secrets: nothing in it is for serving.
# The script never removes the work directory: after success, failure or Ctrl-C it stays, with
# whatever is in it, for the caller to remove; the same name is rejected for another run.
# Ctrl-C does not always end the run: cp_prove catches it (arm_ends(), prover/prover/ends.h) and
# exits by itself, after which the script goes on: press Ctrl-C again.
set -u
umask 077   # what this script makes without -m is closed to group and others, whatever the caller's umask: the scripts under test reject a secret folder below a folder that its group can write
HC=${1:?the folder of the programs} F=${2:?field} WK=${3:?a new work folder}
case $F in 2) ;; *) echo "the second word is the field's number: 2 (this version builds no other)"; exit 2;; esac
# A quoted "~" or "*" in the path would create a directory of that name, which is dangerous to
# remove. Commands read a part that begins with "-" as an option. Where the path crosses a symbolic
# link, the shell's `cd` and the `mkdir` command resolve ".." differently, so the script could work
# in a directory other than the one that it created.
case /$WK/ in */-*|*/../*|*[!A-Za-z0-9/._-]*)
  echo "the work directory's path may contain only letters, digits, \"/\", \".\", \"_\" and \"-\", and no part of it may be \"..\" or begin with \"-\": name another new directory"; exit 2;; esac
{ [ -e "$WK" ] || [ -L "$WK" ]; } && { echo "$WK exists already: name a folder that does not exist"; exit 2; }
for p in cp_commit cp_prove cp_verify; do [ -x "$HC/$p" ] || { echo "no $HC/$p"; exit 2; }; done
unset CDPATH # with it set, "cd" below could enter another directory than the one created here
[[ $WK == /* ]] || WK=$PWD/$WK
while [[ $WK == */ && $WK != / ]]; do WK=${WK%/}; done
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
# This check runs before anything is created: an account that can change a place above the work
# directory could put another directory in its place. From here on `WK` is the path without links
# that open_above() walked, plus the last part as typed.
if open_above "${WK%/*}/"; then
  case $why in
  'it does not exist') { echo "$place, above the work folder, does not exist: make it first; this script makes only the work folder"; exit 2; } ;;
  'it is not a directory' | 'it could not be examined') { echo "$place, above the work folder: $why: name a work folder below directories that you can read"; exit 2; } ;;
  *)
    printf -v named %q "$WK"
    text="the work folder $named lies below $place, which another account may be able to change ($why): it could put another folder in the work folder's place and choose what runs"
    text+=" beside the secrets. Name a work folder below directories that only you or root can change (such as one in your home directory, or in /tmp, whose sticky bit keeps others"
    text+=" from replacing it)$cure"
    { echo "$text"; exit 2; }
    ;;
  esac
fi
WK=${real%/}/${WK##*/}
mkdir -m 700 -- "$WK" || exit 2 # a plain mkdir fails if something has taken the name since the test above; mode 0700, because the directory holds secrets
here=$(cd -- "$(dirname "$0")" && pwd) || exit 2
T=$(cd -- "$here/../.." && pwd) || exit 2
HC=$(cd -- "$HC" && pwd) || exit 2
kit=$T/prover/tests/configurations/pinned_no_proof_of_work
mkdir -p "$WK/world/config" "$WK/world/in" "$WK/aside" || exit 2
WK=$(cd -- "$WK" && pwd) && mkdir -m 700 -- "$WK/tmp" && export TMPDIR=$WK/tmp || exit 2   # its own temporary directory, closed to other accounts: bash before 5.1 writes a here-document to a file in TMPDIR and opens it again by name, and the two below give python3 its program
w=$WK/world; k=$w/isk/audit; of=world; bad=0
said() { if [ "$2" = "as expected" ]; then echo "$1: $2"; else echo "$1: $2 (see ${3:-the logs in $WK})"; bad=1; fi; }
commitment() { # <name of the scene> <status wanted> [--rehearsal; without it the call gets --erase-kept-files]; its work folder: $tools; the commitment: $WK/$of
  ( cd -- "$WK/elsewhere" && bash "$here/audit_commitment.sh" "../$of/ivk" "../$of/isk" "$tools" ${3:---erase-kept-files} ) > "$WK/commitment.$1.log" 2>&1; s=$?
  said "the commitment's audit, $1 (status $s)" "$([ $s = $2 ] && echo as expected || echo "NOT STATUS $2")" "$WK/commitment.$1.log"
}
# 1. cp_commit under the test seed, two leaf tables (further weight tables for the same weights).
# In `isk/audit` (`isk`: the secret key directory) -keep-for-the-audit keeps 32 bytes for the key
# file (`ivk/key.bin`) and for each leaf file (`ivk/leaf/<k>.bin`; `entry` in names and messages):
# the kept files.
cp "$kit/policy.txt" "$kit/KL.bin" "$kit/layout.txt" "$w/config/" || exit 2
(umask 077 && head -c 32 /dev/urandom > "$w/seed.test" && head -c 32 /dev/urandom > "$w/coins.test")
COINS_CONTEXTS=$w/commit.contexts.txt "$HC/cp_commit" -weights made-up-blocks -config "$w/config" -M 2 -ivk "$w/ivk" -isk "$w/isk" -test-seed "$w/seed.test" -keep-for-the-audit > "$w/cp_commit.log" 2>&1 < /dev/null
said "cp_commit" "$([ $? = 0 ] && echo as expected || echo "STATUS OTHER THAN 0")" "$w/cp_commit.log"
[ $bad = 0 ] || { echo "audit play: field $F: cp_commit did not end well (was it built with the test options: prover/build.sh tests? is a GPU free?): nothing further is tried"; exit 1; }
id=$(sed -n "s/^commitment ID: //p" "$w/cp_commit.log" | tr -d ' ')
said "three kept files" "$([ "$(ls "$k" 2>/dev/null | tr '\n' ' ')" = "entry.0.fresh entry.1.fresh key.fresh " ] && echo as expected || echo OTHERWISE)"
# 2. audit_commitment.sh. Expected exit statuses: 0 YES, 1 NO, 2 rejected (`refused` in names and
# messages), 4 NOT WHOLE, 5 NOTHING KEPT HERE, 6 rehearsal passed. `CRY:` lines in the logs report
# failed checks. `$tools/f$F` is where audit_commitment.sh builds its helper programs.
# Every call of audit_commitment.sh or audit_batch.sh starts in a directory that holds a decoy
# numpy.py and libgcc_s.so.1, and below it a decoy docs/format/format_rules.py: none may be imported
# or linked.
mkdir "$WK/elsewhere" "$WK/elsewhere/docs" "$WK/elsewhere/docs/format" && echo 'open("WAS-RUN", "w")' > "$WK/elsewhere/numpy.py" && echo 'open("WAS-RUN", "w")' > "$WK/elsewhere/docs/format/format_rules.py" && echo "no library" > "$WK/elsewhere/libgcc_s.so.1" || exit 2   # libgcc_s.so.1: where libgcc_s.so is a linker's text, ld looks for this name in the current folder first
refused() { # <name of the scene> <words of the rejection's line>; nothing may be written into $tools/f$F, which may already contain a file «theirs»
  commitment "$1" 2 --rehearsal
  said "refused for that, and nothing was laid there" "$(grep -q "$2" "$WK/commitment.$1.log" && [ -z "$(ls -A "$tools/f$F/" 2> /dev/null | grep -v '^theirs$')" ] && echo as expected || echo OTHERWISE)" "$WK/commitment.$1.log"
}
closed() { tools=$WK/$1; mkdir -m 0700 "$tools" || exit 2; } # a work folder of mode 0700, so that what a scene makes writable by group or others inside it stays out of their reach
closed open.group;  mkdir -m 0770 "$tools/f$F"; refused its-group-can-write-in-the-helpers-folder 'its group or others can write it'
closed open.others; mkdir -m 0707 "$tools/f$F"; refused others-can-write-in-the-helpers-folder 'its group or others can write it'
closed open.file;   mkdir -m 0700 "$tools/f$F"; install -m 0660 /dev/null "$tools/f$F/theirs"; refused its-group-can-write-a-file-in-the-helpers-folder 'its group or others can write it'
closed linked;      mkdir -m 0700 "$WK/aside/own"; ln -s "$WK/aside/own" "$tools/f$F"; refused the-helpers-folder-is-a-link 'is a link'
tools=$w/isk/tools; refused the-work-folder-lies-in-the-secret-folder 'lies inside the public or the secret folder'
said "and no folder was made in the secret one" "$([ ! -e "$w/isk/tools" ] && echo as expected || echo OTHERWISE)"
tools=$WK; refused the-work-folder-holds-the-two-other-folders 'is inside the work folder'
# If audit_commitment.sh ever lost its rejection of "/", it would create /f<field>, which an
# ordinary account cannot. So this call is skipped for root, and where /f<field> exists.
# The call after it has --rehearsal in the work directory's place. An unquoted empty variable
# vanishes from a command line: with the work directory's name gone, --rehearsal would be read as
# that name. So the call must be rejected.
# A secret key directory of whose path a part is missing must be rejected when the path is checked:
# what is created there a moment later would never be checked. One that its group can read must be
# rejected, as cp_prove rejects it.
# A call without a fourth word must be rejected, and so must audit.py itself if it gets neither
# word; a symbolic link in the helpers' directory must be rejected before anything is written there.
# Then a path through `loose` must be rejected for the secret key directory: an account that can
# write in a directory on the path could lead the audit, and what it destroys, to another
# directory. For the public key directory (`ivk`), which is only read, the same path passes.
# `built-from.sum` holds audit_commitment.sh's sum over its helpers' sources (64 hexadecimal digits
# and a newline). While the sum matches, a call must build nothing; with the file emptied, it must
# build the helpers again.
# An empty fourth argument must be rejected.
if [ "$(id -u)" != 0 ] && [ ! -e "/f$F" ]; then tools=/; refused the-work-folder-is-the-root 'the work folder is «/»'; fi
( cd -- "$WK/elsewhere" && bash "$here/audit_commitment.sh" "../$of/ivk" "../$of/isk" --rehearsal ) > "$WK/commitment.no-work-folder.log" 2>&1; s=$?
said "the commitment's audit with --rehearsal where the work folder must be named (status $s): refused for that, no folder of that name is made, and the kept files are there" "$([ $s = 2 ] && grep -q 'stands where a folder must be named' "$WK/commitment.no-work-folder.log" && [ ! -e "$WK/elsewhere/--rehearsal" ] && [ "$(ls "$k" 2>/dev/null | tr '\n' ' ')" = "entry.0.fresh entry.1.fresh key.fresh " ] && echo as expected || echo OTHERWISE)" "$WK/commitment.no-work-folder.log"
( cd -- "$WK/elsewhere" && bash "$here/audit_commitment.sh" "../$of/ivk" "../$of/isk" --erase-kept-files ) > "$WK/commitment.refused.log" 2>&1; s=$?
said "the commitment's audit with --erase-kept-files where the work folder must be named (status $s): refused for that, no folder of that name is made, and the kept files are there" "$([ $s = 2 ] && grep -q 'stands where a folder must be named' "$WK/commitment.refused.log" && [ ! -e "$WK/elsewhere/--erase-kept-files" ] && [ "$(ls "$k" 2>/dev/null | tr '\n' ' ')" = "entry.0.fresh entry.1.fresh key.fresh " ] && echo as expected || echo OTHERWISE)" "$WK/commitment.refused.log"
mkdir -m 0770 "$WK/loose" && ln -s "$w" "$WK/loose/up" || exit 2   # a folder of the caller's own that its group can write, and in it a link to the commitment's folder ($WK itself is closed)
( cd -- "$WK/elsewhere" && bash "$here/audit_commitment.sh" "../$of/ivk" "$WK/loose/up/isk" "$WK/unused" --rehearsal ) > "$WK/commitment.loose-way.log" 2>&1; s=$?
said "the commitment's audit with a folder that its group can write on the way to the secret folder (status $s): refused for that, by name, no work folder is made, and the kept files are there" "$([ $s = 2 ] && grep -q 'on the way to the secret folder .*/loose: its group or others can write in it' "$WK/commitment.loose-way.log" && [ ! -e "$WK/unused" ] && [ "$(ls "$k" 2>/dev/null | tr '\n' ' ')" = "entry.0.fresh entry.1.fresh key.fresh " ] && echo as expected || echo OTHERWISE)" "$WK/commitment.loose-way.log"
( cd -- "$WK/elsewhere" && bash "$here/audit_commitment.sh" "../$of/ivk" "$WK/none/isk" "$WK/unused" --rehearsal ) > "$WK/commitment.refused.log" 2>&1; s=$?
said "the commitment's audit with a secret folder of whose path a part is missing (status $s): refused at the look along the way, and no work folder is made" "$([ $s = 2 ] && grep -q 'on the way to the secret folder .*: it does not exist' "$WK/commitment.refused.log" && [ ! -e "$WK/unused" ] && echo as expected || echo OTHERWISE)" "$WK/commitment.refused.log"
chmod 750 "$w/isk"; ( cd -- "$WK/elsewhere" && bash "$here/audit_commitment.sh" "../$of/ivk" "../$of/isk" "$WK/unused" --rehearsal ) > "$WK/commitment.refused.log" 2>&1; s=$?; chmod 700 "$w/isk"
said "the commitment's audit with a secret folder that its group can read (status $s): refused for that, no work folder is made, and the kept files are there" "$([ $s = 2 ] && grep -q 'its group or others have a permission on it' "$WK/commitment.refused.log" && [ ! -e "$WK/unused" ] && [ "$(ls "$k" 2>/dev/null | tr '\n' ' ')" = "entry.0.fresh entry.1.fresh key.fresh " ] && echo as expected || echo OTHERWISE)" "$WK/commitment.refused.log"
tools=$WK/tools; mkdir -m 0775 "$tools" && echo "no library" > "$tools/libfield.so" || exit 2   # the work folder itself may be open to its group; a file of the library's name at its top is not looked at
commitment a-rehearsal 6 --rehearsal
( cd -- "$WK/elsewhere" && bash "$here/audit_commitment.sh" "../$of/ivk" "../$of/isk" "$tools" ) > "$WK/commitment.no-word-at-all.log" 2>&1; s=$?
said "the commitment's audit without a fourth word (status $s): refused, both words named, and the kept files are there" "$([ $s = 2 ] && grep -q -e '--rehearsal .*--erase-kept-files' "$WK/commitment.no-word-at-all.log" && [ "$(ls "$k" | wc -l)" = 3 ] && echo as expected || echo OTHERWISE)" "$WK/commitment.no-word-at-all.log"
( cd -- "$tools/f$F" && AUDIT_LIB=./libfield.so python3 -B "$here/audit.py" "$T" . - $F --level=1 --commitment --ivk="$w/ivk" --isk="$w/isk" ) > "$WK/commitment.refused.log" 2>&1; s=$?
said "audit.py itself, called for the commitment with neither word (status $s): refused for that, and the kept files are there" "$([ $s = 2 ] && grep -q 'want exactly one of --rehearsal' "$WK/commitment.refused.log" && [ "$(ls "$k" | wc -l)" = 3 ] && echo as expected || echo OTHERWISE)" "$WK/commitment.refused.log"
mkdir -m 700 "$WK/link-inside" "$WK/link-inside/f$F" && echo "not the helpers' sum" > "$WK/aside/outside" && ln -s "$WK/aside/outside" "$WK/link-inside/f$F/built-from.sum" || exit 2
kept_tools=$tools; tools=$WK/link-inside; commitment a-link-in-the-helpers-folder 2 --rehearsal; tools=$kept_tools
said "refused for that: the link is the one thing in that folder, and the file that it leads to is as it was" "$(grep -q 'is anything other than a plain file or a folder' "$WK/commitment.a-link-in-the-helpers-folder.log" && [ -L "$WK/link-inside/f$F/built-from.sum" ] && [ "$(ls -A "$WK/link-inside/f$F")" = built-from.sum ] && [ "$(cat "$WK/aside/outside")" = "not the helpers' sum" ] && echo as expected || echo OTHERWISE)" "$WK/commitment.a-link-in-the-helpers-folder.log"
( cd -- "$WK/elsewhere" && bash "$here/audit_commitment.sh" "$WK/loose/up/ivk" "../$of/isk" "$tools" --rehearsal ) > "$WK/commitment.loose-way-to-the-public-folder.log" 2>&1; s=$?
said "the commitment's audit, a rehearsal with that folder on the way to the public folder, which is only read (status $s)" "$([ $s = 6 ] && echo as expected || echo "NOT STATUS 6")" "$WK/commitment.loose-way-to-the-public-folder.log"
said "that second call in the work folder has built nothing anew" "$(grep -q '^building' "$WK/commitment.a-rehearsal.log" && [ -s "$WK/commitment.loose-way-to-the-public-folder.log" ] && ! grep -q '^building' "$WK/commitment.loose-way-to-the-public-folder.log" && echo as expected || echo OTHERWISE)" "$WK/commitment.loose-way-to-the-public-folder.log"
: > "$tools/f$F/built-from.sum"; commitment the-helpers-sum-is-emptied 6 --rehearsal
said "the helpers are built anew, and their sum is written again" "$(grep -q '^building' "$WK/commitment.the-helpers-sum-is-emptied.log" && [ "$(wc -c < "$tools/f$F/built-from.sum")" = 65 ] && echo as expected || echo OTHERWISE)" "$WK/commitment.the-helpers-sum-is-emptied.log"
exec 8< "$tools/f$F" && flock -n 8 || exit 2   # this script itself holds the lock on the helpers' folder
commitment the-helpers-folder-is-locked 2 --rehearsal
exec 8<&-
said "refused for that" "$(grep -q 'another audit is at work' "$WK/commitment.the-helpers-folder-is-locked.log" && echo as expected || echo OTHERWISE)" "$WK/commitment.the-helpers-folder-is-locked.log"
( cd -- "$WK/elsewhere" && bash "$here/audit_commitment.sh" "../$of/ivk" "../$of/isk" "$tools" "" ) > "$WK/commitment.an-empty-last-word.log" 2>&1; s=$?
said "the commitment's audit with an empty last word (status $s): refused for that, and the kept files are there" "$([ $s = 2 ] && grep -q 'the fourth word is empty' "$WK/commitment.an-empty-last-word.log" && [ "$(ls "$k" | wc -l)" = 3 ] && echo as expected || echo OTHERWISE)" "$WK/commitment.an-empty-last-word.log"
mv "$k/entry.1.fresh" "$WK/aside/"; commitment one-kept-file-moved-away 4
(umask 077 && head -c 32 /dev/zero | tr '\0' '\1' > "$k/entry.1.fresh"); commitment one-kept-file-false 1
said "its reports name the second entry alone" "$([ "$(grep -c '^CRY: ' "$WK/commitment.one-kept-file-false.log")" -gt 0 ] && ! grep '^CRY: ' "$WK/commitment.one-kept-file-false.log" | grep -qv 'leaf/1.bin\|entry 1' && echo as expected || echo OTHERWISE)" "$WK/commitment.one-kept-file-false.log"
mv -f "$WK/aside/entry.1.fresh" "$k/entry.1.fresh"; commitment in-earnest 0
said "the kept files are gone, their folder stands" "$([ -d "$k" ] && [ -z "$(ls -A "$k")" ] && echo as expected || echo OTHERWISE)"
commitment once-more 5
# 3. The audit's self-tests. Step 2 built the library `AUDIT_LIB` and the public inputs in `pub`.
export AUDIT_LIB=$WK/tools/f$F/libfield.so
pub=$(find "$WK/tools/f$F" -maxdepth 1 -type d -name 'pub.*' | head -n 1)
said "the random generator against the tree's known answers" "$(python3 -B "$here/coins_known.py" "$T" > "$WK/coins_known.log" 2>&1 && echo as expected || echo OTHERWISE)" "$WK/coins_known.log"
said "the code of field 2 against the format's own functions" "$(python3 -B "$here/test_field.py" "$T" > "$WK/test_field.log" 2>&1 && echo as expected || echo OTHERWISE)" "$WK/test_field.log"
# The three programs that import docs/format/format_rules.py take it only from the repository that
# they are in: started with "." for the repository, in the directory with step 2's decoy of that
# name below it, each must stop before it imports anything from there.
for p in "audit.py . $pub $w $F --level=1" "coins_known.py ." "test_field.py ."; do
  ( cd -- "$WK/elsewhere" && python3 -B "$here"/$p ) > "$WK/another-repository.log" 2>&1; s=$?
  said "${p%% *} with «.» for the repository, in another folder (status $s): stopped for that" "$([ $s != 0 ] && grep -q 'must be the repository that this' "$WK/another-repository.log" && echo as expected || echo OTHERWISE)" "$WK/another-repository.log"
done
# 4. Four requests, cp_prove, cp_verify on each proof.
( cd -- "$here" && python3 -B - "$w/config/policy.txt" "$w/in" ) << 'END' || exit 2   # from this folder: a text from standard input finds its modules in the current folder first
import struct, sys
word = dict(l.split() for l in open(sys.argv[1]) if len(l.split()) == 2)
T, V = 1 << int(word["lt"]), int(word["V"])
for j in range(4):  # the requests of prover/tests/coin_contexts_listed.py
    open("%s/r%d.req" % (sys.argv[2], j), "wb").write(struct.pack("<%dI" % (T + 2), *[(2654435761 * (t + 1 + 1000 * j) >> 7) % V for t in range(T)], T - 1, T))
END
COINS_CONTEXTS=$w/batch.contexts.txt "$HC/cp_prove" -isk "$w/isk" -in "$w/in" -out "$w/out" -in-step 2 -test-coins "$w/coins.test" > "$w/cp_prove.log" 2>&1 < /dev/null
said "cp_prove" "$([ $? = 0 ] && echo as expected || echo "STATUS OTHER THAN 0")" "$w/cp_prove.log"
for j in 0 1 2 3; do
  "$HC/cp_verify" "$w/config" "$w/ivk" "$w/in/r$j.req" "$w/out/r$j.y" "$w/out/r$j.proof" -id "$id" > "$w/cp_verify.r$j.log" 2>&1 < /dev/null; s=$?
  said "cp_verify on r$j (status $s; a pass is status 0 with the last line YES)" "$([ $s = 0 ] && [ "$(tail -n 1 "$w/cp_verify.r$j.log")" = YES ] && echo as expected || echo NO YES)" "$w/cp_verify.r$j.log"
done
# 5. audit.py --level=1 on all seven files must report nothing. It must also print that it
# recomputed (`made again`) all 9 trees that consist only of random values and compared (`held`) 10
# answers to samples: this configuration's numbers. How many roots and given digests it compared
# depends on this run's seed and randomness.
# 6. audit.py --flip: every flipped value that a check covers must be reported, and nothing else.
python3 -B "$here/audit.py" "$T" "$pub" "$w" $F --level=1 > "$WK/level1.log" 2>&1
said "all seven files at level 1" "$(grep -q "^audit.py: field $F: LEVEL 1 over 1 key, 2 entries, 4 proofs: NOTHING CRIED$" "$WK/level1.log" && echo as expected || echo SOMETHING WAS REPORTED OR NOTHING RAN)" "$WK/level1.log"
python3 -B "$here/audit.py" "$T" "$pub" "$w" $F --level=1 --flip > "$WK/flip.log" 2>&1
said "what it made again and held is said: all 9 trees of random values alone, 10 answers to samples, some roots and given digests" "$(grep -c -E '^\(H0\) different trees of random values alone: 9, made again whole: 9; their roots and given digests held, in all files: [1-9][0-9]*; answers to samples of such messages held: 10$' "$WK/level1.log" | grep -qx 1 && echo as expected || echo OTHERWISE)" "$WK/level1.log"
said "flipped bits" "$(grep -q '^FLIP: EVERY HELD CLASS IS ALIVE, AND NOTHING ELSE MOVED$' "$WK/flip.log" && echo as expected || echo A CHECK IS DEAD OR SOMETHING ELSE MOVED)" "$WK/flip.log"
# 7. audit_commitment.sh with the kept files written again by hand (32 zeros, as under a test
# seed): NO, only because cp_prove has run on this commitment.
for n in key entry.0 entry.1; do (umask 077 && head -c 32 /dev/zero > "$k/$n.fresh"); done
commitment behind-the-proofs 1
said "its reports name the counts alone" "$([ "$(grep -c '^CRY: ' "$WK/commitment.behind-the-proofs.log")" = "$(grep -c '^CRY: .*a prover has run on this folder' "$WK/commitment.behind-the-proofs.log")" ] && echo as expected || echo OTHERWISE)" "$WK/commitment.behind-the-proofs.log"
# 8. A second commitment, one leaf table, without -test-seed: cp_commit draws the seed and the kept
# bytes, as for an operator, though in the test build. Under a test seed the kept bytes are 32
# zeros, the value that the audit uses where nothing is kept. So with 32 zeros for the key file's
# kept bytes the audit must answer NO: it would pass if cp_commit had not used the bytes that it
# drew.
# The `_roomy` configuration's synthetic scales pass cp_commit's worst-case check, which -test-seed
# skips in step 1.
d=$WK/drawn; k=$d/isk/audit; of=drawn
mkdir -p "$d/config" && cp "${kit}_roomy/policy.txt" "${kit}_roomy/KL.bin" "${kit}_roomy/layout.txt" "$d/config/" || exit 2
"$HC/cp_commit" -weights made-up-blocks -config "$d/config" -M 1 -ivk "$d/ivk" -isk "$d/isk" -keep-for-the-audit > "$d/cp_commit.log" 2>&1 < /dev/null
said "cp_commit under a drawn seed" "$([ $? = 0 ] && echo as expected || echo "STATUS OTHER THAN 0")" "$d/cp_commit.log"
said "its two kept files hold 32 bytes that are not all zero" "$(for n in key entry.0; do [ "$(wc -c < "$k/$n.fresh")" = 32 ] && ! cmp -s "$k/$n.fresh" <(head -c 32 /dev/zero) || echo "$n"; done | grep -q . && echo OTHERWISE || echo as expected)"
commitment drawn-a-rehearsal 6 --rehearsal
mv "$k/key.fresh" "$WK/aside/"; commitment drawn-the-keys-kept-file-moved-away 4
(umask 077 && head -c 32 /dev/zero > "$k/key.fresh"); commitment drawn-the-keys-kept-file-false 1
said "its reports name the key file alone" "$([ "$(grep -c '^CRY: ' "$WK/commitment.drawn-the-keys-kept-file-false.log")" -gt 0 ] && ! grep '^CRY: ' "$WK/commitment.drawn-the-keys-kept-file-false.log" | grep -qv 'key\.bin\|key file' && echo as expected || echo OTHERWISE)" "$WK/commitment.drawn-the-keys-kept-file-false.log"
mv -f "$WK/aside/key.fresh" "$k/key.fresh"; commitment drawn-in-earnest 0
said "the kept files are gone, their folder stands" "$([ -d "$k" ] && [ -z "$(ls -A "$k")" ] && echo as expected || echo OTHERWISE)"
# 9. audit_batch.sh on a second batch of the first commitment. Exit statuses as in step 2.
# keep() writes the batch's kept file, `audit/batch.fresh`, from `coins2.test`: under -test-coins
# those are the 32 bytes that cp_prove -keep-for-the-audit would keep.
# The first call finds only step 7's three files in `audit`: NOTHING KEPT HERE, and they stay.
# cp_prove rejects an `isk` while anything is in its `audit`, so the script then removes them.
# The batch has four files, one slot each, ordered by name without ".req": `0` (wrong length), `a`,
# `a-b` (after `a`, though "a-b.req" sorts before "a.req"), `d` (a token outside the vocabulary).
# The two proofs are in slots 1 and 2, counted from 0.
# Five names must be rejected for the secret key directory: "", "/", ".", ".." and a symbolic link
# to it. These calls, and the one after them (a work directory that contains the other four), have
# --erase-kept-files: a verdict would erase the kept file, so each must exit with 2 and leave it.
# So must a call with --erase-kept-files and a mistyped directory of responses, and so must
# audit.py itself, called with these two. A call without a sixth word, or with an empty one, must be
# rejected. A missing part of the secret key directory's path, and a secret key directory that its
# group can read: as in step 2, and so are `loose` and --rehearsal in the work directory's place.
k=$w/isk/audit
keep() { (umask 077 && head -c ${1:-32} "$w/coins2.test" > "$k/batch.fresh"); }
batch() { # <name of the scene> <status wanted> <what audit/ must hold afterwards> [--rehearsal; without it the call gets --erase-kept-files]
  ( cd "$WK/elsewhere" && bash "$here/audit_batch.sh" ../world/ivk ../world/isk "../world/${requests:-in2}" "../world/${responses:-out2}" "$tools" ${4:---erase-kept-files} ) > "$WK/batch.$1.log" 2>&1; s=$?
  said "the batch's audit, $1 (status $s)" "$([ $s = $2 ] && echo as expected || echo "NOT STATUS $2")" "$WK/batch.$1.log"
  said "audit/ then holds: $3" "$([ "$(LC_ALL=C ls -A "$k" | tr '\n' ' ')" = "$3" ] && echo as expected || echo OTHERWISE)"
}
requests=in responses=out batch a-commitments-kept-files-alone 5 "entry.0.fresh entry.1.fresh key.fresh "   # step 4's folders: in2 and out2 are not made yet, and a missing folder is rejected
rm -f "$k/key.fresh" "$k/entry.0.fresh" "$k/entry.1.fresh"   # the three files that 7 has written, by their names
mkdir "$w/in2" && (umask 077 && head -c 32 /dev/urandom > "$w/coins2.test") || exit 2
( cd "$here" && python3 -B - "$w/config/policy.txt" "$w/in2" ) << 'END' || exit 2
import struct, sys
word = dict(l.split() for l in open(sys.argv[1]) if len(l.split()) == 2)
T, V = 1 << int(word["lt"]), int(word["V"])
for j, n in enumerate(("0", "a", "a-b", "d")):
    w = [(2654435761 * (t + 1 + 1000 * (j + 4)) >> 7) % V for t in range(T)] + [T - 1, T]
    if n == "d": w[5] = V
    open("%s/%s.req" % (sys.argv[2], n), "wb").write(struct.pack("<%dI" % (T + 2), *w)[:4 * (T + 1) if n == "0" else None])
END
"$HC/cp_prove" -isk "$w/isk" -in "$w/in2" -out "$w/out2" -test-coins "$w/coins2.test" > "$w/cp_prove2.log" 2>&1 < /dev/null
said "cp_prove on four files, two of them no requests (status $?; 0 or 3)" "$([ "$(LC_ALL=C ls "$w/out2" | tr '\n' ' ')" = "0.failed a-b.proof a-b.y a.proof a.y d.failed " ] && echo as expected || echo "OTHER FILES THAN TWO RESPONSES AND TWO «.failed»")" "$w/cp_prove2.log"
keep; batch with-rehearsal 6 "batch.fresh " --rehearsal
( cd "$WK/elsewhere" && bash "$here/audit_batch.sh" ../world/ivk ../world/isk ../world/in2 ../world/out2 "$tools" ) > "$WK/batch.no-word-at-all.log" 2>&1; s=$?
said "the batch's audit without a sixth word (status $s): refused, both words named, and the kept file is there" "$([ $s = 2 ] && grep -q -e '--rehearsal .*--erase-kept-files' "$WK/batch.no-word-at-all.log" && [ "$(ls "$k" | tr '\n' ' ')" = "batch.fresh " ] && echo as expected || echo OTHERWISE)" "$WK/batch.no-word-at-all.log"
( cd "$WK/elsewhere" && bash "$here/audit_batch.sh" ../world/ivk ../world/isk ../world/in2 ../world/out2-mistyped "$tools" --erase-kept-files ) > "$WK/batch.a-mistyped-folder.log" 2>&1; s=$?
said "the batch's audit with a mistyped folder of responses, --erase-kept-files (status $s): refused, and the kept file is there" "$([ $s = 2 ] && grep -q 'on the way to the folder of -out .*: it does not exist' "$WK/batch.a-mistyped-folder.log" && [ "$(ls "$k" | tr '\n' ' ')" = "batch.fresh " ] && echo as expected || echo OTHERWISE)" "$WK/batch.a-mistyped-folder.log"
said "it has found four slots, two of them without a request" "$(grep -q '^the batch: 4 slot(s): 2 with a reply and a proof, 2 failed (2 of these files are not well-formed requests); ' "$WK/batch.with-rehearsal.log" && echo as expected || echo OTHERWISE)" "$WK/batch.with-rehearsal.log"
(umask 077 && { head -c 1 "$w/coins2.test" | tr '\000-\377' '\001-\377\000'; tail -c +2 "$w/coins2.test"; } > "$k/batch.fresh"); batch one-kept-byte-wrong 1 ""
said "its reports name both proofs" "$(grep -q '^CRY: .*a\.proof' "$WK/batch.one-kept-byte-wrong.log" && grep -q '^CRY: .*a-b\.proof' "$WK/batch.one-kept-byte-wrong.log" && echo as expected || echo OTHERWISE)" "$WK/batch.one-kept-byte-wrong.log"
keep 20; batch a-short-kept-file 4 ""
keep; (umask 077 && echo x > "$k/other"); batch another-name-beside-it 4 "other "; rm -f "$k/other"
keep; ln "$k/batch.fresh" "$WK/aside/second-name"; batch a-hard-link 4 "batch.fresh "
said "the bytes under the second name are as they were" "$(cmp -s "$WK/aside/second-name" "$w/coins2.test" && echo as expected || echo OTHERWISE)"; rm -f "$k/batch.fresh" "$WK/aside/second-name"
keep; echo x > "$w/in2/x.req.tmp"; batch an-extra-file-among-the-requests 4 ""; rm -f "$w/in2/x.req.tmp"
keep; mv "$w/in2/a.req" "$w/in2/b.req"; batch a-request-renamed 4 ""; mv "$w/in2/b.req" "$w/in2/a.req"
keep; mv "$w/in2/a-b.req" "$w/in2/A.req"; for e in y proof; do mv "$w/out2/a-b.$e" "$w/out2/A.$e"; done; batch renamed-in-both-folders-so-that-the-order-changes 1 ""
mv "$w/in2/A.req" "$w/in2/a-b.req"; for e in y proof; do mv "$w/out2/A.$e" "$w/out2/a-b.$e"; done
keep; ln -s "$w/isk" "$WK/aside/link-to-the-secret-folder"
for a in "" / . .. "$WK/aside/link-to-the-secret-folder"; do
  ( cd "$WK/elsewhere" && bash "$here/audit_batch.sh" ../world/ivk "$a" ../world/in2 ../world/out2 "$tools" --erase-kept-files ) > "$WK/batch.refused.log" 2>&1; s=$?
  said "the batch's audit with «$a» for the secret folder (status $s): refused, and the kept file is there" "$([ $s = 2 ] && [ -f "$k/batch.fresh" ] && echo as expected || echo OTHERWISE)" "$WK/batch.refused.log"
done
( cd "$WK/elsewhere" && bash "$here/audit_batch.sh" ../world/ivk ../world/isk ../world/in2 ../world/out2 "$WK" --erase-kept-files ) > "$WK/batch.refused.log" 2>&1; s=$?
said "the batch's audit with a work folder that holds the four other folders (status $s): refused for that, nothing written there, and the kept file is there" "$([ $s = 2 ] && grep -q 'is inside the work folder' "$WK/batch.refused.log" && [ ! -e "$WK/f$F" ] && [ -f "$k/batch.fresh" ] && echo as expected || echo OTHERWISE)" "$WK/batch.refused.log"
( cd "$WK/elsewhere" && bash "$here/audit_batch.sh" ../world/ivk ../world/isk ../world/in2 ../world/out2 --rehearsal ) > "$WK/batch.refused.log" 2>&1; s=$?
said "the batch's audit with --rehearsal where the work folder must be named (status $s): refused for that, no folder of that name is made, and the kept file is there" "$([ $s = 2 ] && grep -q 'stands where a folder must be named' "$WK/batch.refused.log" && [ ! -e "$WK/elsewhere/--rehearsal" ] && [ -f "$k/batch.fresh" ] && echo as expected || echo OTHERWISE)" "$WK/batch.refused.log"
( cd "$WK/elsewhere" && bash "$here/audit_batch.sh" ../world/ivk "$WK/loose/up/isk" ../world/in2 ../world/out2 "$tools" --rehearsal ) > "$WK/batch.refused.log" 2>&1; s=$?
said "the batch's audit with a folder that its group can write on the way to the secret folder (status $s): refused for that, by name, and the kept file is there" "$([ $s = 2 ] && grep -q 'on the way to the secret folder .*/loose: its group or others can write in it' "$WK/batch.refused.log" && [ -f "$k/batch.fresh" ] && echo as expected || echo OTHERWISE)" "$WK/batch.refused.log"
( cd "$WK/elsewhere" && bash "$here/audit_batch.sh" ../world/ivk "$WK/none/isk" ../world/in2 ../world/out2 "$tools" --erase-kept-files ) > "$WK/batch.refused.log" 2>&1; s=$?
said "the batch's audit with a secret folder of whose path a part is missing (status $s): refused at the look along the way" "$([ $s = 2 ] && grep -q 'on the way to the secret folder .*: it does not exist' "$WK/batch.refused.log" && echo as expected || echo OTHERWISE)" "$WK/batch.refused.log"
chmod 750 "$w/isk"; ( cd "$WK/elsewhere" && bash "$here/audit_batch.sh" ../world/ivk ../world/isk ../world/in2 ../world/out2 "$tools" --erase-kept-files ) > "$WK/batch.refused.log" 2>&1; s=$?; chmod 700 "$w/isk"
said "the batch's audit with a secret folder that its group can read (status $s): refused for that, and the kept file is there" "$([ $s = 2 ] && grep -q 'its group or others have a permission on it' "$WK/batch.refused.log" && [ -f "$k/batch.fresh" ] && echo as expected || echo OTHERWISE)" "$WK/batch.refused.log"
( cd "$WK/elsewhere" && bash "$here/audit_batch.sh" ../world/ivk ../world/isk ../world/in2 ../world/out2 --erase-kept-files ) > "$WK/batch.refused.log" 2>&1; s=$?
said "the batch's audit with --erase-kept-files where the work folder must be named (status $s): refused for that, no folder of that name is made, and the kept file is there" "$([ $s = 2 ] && grep -q 'stands where a folder must be named' "$WK/batch.refused.log" && [ ! -e "$WK/elsewhere/--erase-kept-files" ] && [ -f "$k/batch.fresh" ] && echo as expected || echo OTHERWISE)" "$WK/batch.refused.log"
( cd "$WK/elsewhere" && bash "$here/audit_batch.sh" ../world/ivk ../world/isk ../world/in2 ../world/out2 "$tools" "" ) > "$WK/batch.refused.log" 2>&1; s=$?
said "the batch's audit with an empty last word (status $s): refused for that, and the kept file is there" "$([ $s = 2 ] && grep -q 'the sixth word is empty' "$WK/batch.refused.log" && [ -f "$k/batch.fresh" ] && echo as expected || echo OTHERWISE)" "$WK/batch.refused.log"
python3 -B "$here/audit.py" "$T" "$pub" - $F --level=1 --batch --erase-kept-files --ivk="$w/ivk" --isk="$w/isk" --in="$w/in2" --out="$w/out2-mistyped" > "$WK/batch.refused.log" 2>&1; s=$?
said "audit.py itself, called for the batch with a mistyped folder of responses, --erase-kept-files (status $s): refused for that, and the kept file is there" "$([ $s = 2 ] && grep -q -e '--out=.*/out2-mistyped is not there, or is no folder' "$WK/batch.refused.log" && [ -f "$k/batch.fresh" ] && echo as expected || echo OTHERWISE)" "$WK/batch.refused.log"
batch without-rehearsal 0 ""
batch once-more 5 ""
said "the numpy.py in the current folder and the format_rules.py below it were not run" "$([ ! -e "$WK/elsewhere/WAS-RUN" ] && echo as expected || echo "IT WAS RUN")"
echo "audit play: field $F: $([ $bad = 0 ] && echo ALL AS EXPECTED || echo "NOT ALL AS EXPECTED: see the lines above; logs in $WK")"
exit $bad
