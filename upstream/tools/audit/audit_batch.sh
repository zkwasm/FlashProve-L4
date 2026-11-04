#!/bin/bash
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.

# Audits one batch made with `cp_prove -keep-for-the-audit`. Run it before any of the batch's
# responses is handed out: it cannot tell whether one was. audit.py recomputes random values in the
# proofs from the commitment's seed and from the 32 bytes kept in <secret folder>/audit/batch.fresh.
# audit.py accepts synthetic weights only.
# Steps: tools/audit/README.md, "Audit one batch of proofs".
# What a YES leaves unchecked: tools/audit/README.md, "What a batch's YES does not cover".
# usage: audit_batch.sh <public folder: cp_commit's -ivk> <secret folder: -isk> <cp_prove's -in> <cp_prove's -out> <work folder> --rehearsal | --erase-kept-files
# --rehearsal: all the checks, and nothing is erased. --erase-kept-files: the audit itself, which
# erases the kept file. A call without a sixth word is refused.
# Builds its helpers in <work folder>/f<field> (always f2: Goldilocks is the only field), where
# nothing secret is written, and runs audit.py on the other four folders in place, by their names as
# typed. Never copy a secret folder for an audit: the original's batch.fresh would remain, and each
# folder would keep its own count of opened columns.
# Needs Linux with GNU tools, flock, g++ and python3 with numpy; no GPU. The first call builds the
# helpers in two to three minutes. It compiles verifier/src/cp_verify.cpp, for which
# verifier/build.sh gives g++ 13 or later and about 1.6 GB of memory.
# The runs so far: tools/audit/README.md, "What has been run".
# All that this script makes is the work folder and what is in it: the helpers (pubtool,
# libfield.so), an object file from each build, the public inputs and their logs. It all stays after
# every ending, an interrupted one too.
# The helpers can read the secrets, so the work folder is checked as in audit_commitment.sh, which
# can share it. Not checked: who can write to the repository, which supplies audit.py and the
# helpers' sources; the caller's environment: PATH, LD_*, BASH_ENV and others in it can name code.
# Before anything in the other four folders is read, the path to each is examined from the top.
# Exit code 2: a place on it belongs to another account, is missing, is no directory or cannot be
# examined; or the secret folder is, or lies below, a place that another account may be able to
# change, or its group or others have a permission on it. More at owners_above().
# At every verdict (YES, NO, NOT WHOLE) audit.py erases batch.fresh and nothing else: it overwrites,
# syncs and unlinks the file (KeptBatch.erase() in kept.py); a journal, snapshot or backup may still
# contain the bytes. It erases only a regular file of at most 32 bytes with one hard link that the
# caller owns, and only if the file was not moved, replaced or changed during the audit. Otherwise
# the last line says that the file was not erased, and there is no YES.
# An uncaught exception in audit.py, once it has seen batch.fresh, ends as NOT WHOLE and erases it
# too. All this only with --erase-kept-files. A batch whose file is erased cannot be audited again,
# so rehearse first (--rehearsal): then audit.py erases nothing.
# At every other ending (refused, NOTHING KEPT HERE, no verdict, Ctrl-C, a kill) nothing is erased,
# unless the erasure had begun. While batch.fresh exists, hiding stays void for its batch, and
# cp_prove rejects the secret folder (`cp-kept-for-the-audit`). Responses may be handed out only
# after a YES. Where audit.py prints MEMORY NOT LOCKED, the seed and the 32 bytes may reach swap.
# This script's own lines remove nothing and send no signal. `mv` puts new helpers in the earlier
# ones' place, and the compiler removes its temporary files. If the script alone is killed, the
# compiler or pubtool that it started runs on to its end. The last line replaces the shell with
# audit.py (`exec`), which starts no other program.
# Keep the folder of -in as cp_prove read it: the batch's random values depend on every request file
# and on the order of their names. Use a batch in which no well-formed request fails: a failed one's
# leaf of the binding tree (a Merkle tree over the batch) is in no file, so less can be compared.
# Exit code: audit.py's 0 YES, 1 NO, 2 refused, 4 NOT WHOLE, 5 NOTHING KEPT HERE, 6 rehearsal
# passed, 7 no verdict; this script's 2 refused, 3 a tool is missing, or the helpers or the public
# inputs could not be made.
set -u
umask 077
unset CDPATH
no() { echo "audit_batch.sh: $2. Nothing was looked at or erased: if <secret folder>/audit/batch.fresh is there, it still is, hiding stays void for its batch, and none of its responses may be handed out"; exit "$1"; }   # <status> <why>
[ $# = 5 ] || [ $# = 6 ] || no 2 "usage: audit_batch.sh <public folder> <secret folder> <folder of -in> <folder of -out> <work folder> --rehearsal | --erase-kept-files"
MORE=${6:-}
[ $# = 5 ] || [ -n "$MORE" ] || no 2 "the sixth word is empty (a variable that was not set?): write --rehearsal or --erase-kept-files there"
case $MORE in ''|--rehearsal|--erase-kept-files) ;; *) no 2 "the sixth word can only be --rehearsal or --erase-kept-files";; esac
# Both words are looked for among all the words before anything is run: one of them in a folder's
# place is refused.
for p in "$@"; do [ "$p" != --rehearsal ] && [ "$p" != --erase-kept-files ] || [ -n "$MORE" ] || no 2 "«$p» stands where a folder must be named (a variable before it that was empty or not set?): name all five folders"; done
[ -n "$MORE" ] || no 2 "the sixth word is missing: say which call this is: --rehearsal (all the checks; nothing is erased) or --erase-kept-files (the audit itself: at its verdict, YES, NO or NOT WHOLE, it erases <secret folder>/audit/batch.fresh, if its last line says so, and the batch cannot be audited again: rehearse first)"
# An unquoted variable that is empty vanishes from the call, and the next word takes its place: if
# the work folder's word vanished, the last word would be taken for its name.
for p in "${@:1:5}"; do case $p in -*) no 2 "«$p» stands where a folder must be named (a variable that was not set?): name all five folders, and write ./ before a name that begins with «-»";; esac; done
for p in python3 g++ sha256sum find id realpath flock sort xargs head cut; do command -v $p > /dev/null || no 3 "$p is not to be found"; done
whole() { # <which folder> <its name as given>: prints the name as given, made whole without following a link: the current folder changes below
  b=${2%"${2##*[!/]}"}   # without the slashes at its end
  case $b in ''|.|..) no 2 "the name of the $1 is empty, or is «/», «.» or «..» (a variable that was not set?): name the folder in full";; esac
  [ ! -L "$b" ] || no 2 "the name of the $1 is a link: name the folder itself"
  r=$(realpath -m -- "$b") && [ "$r" != / ] || no 2 "the name of the $1 cannot be read as a path, or leads to «/»"
  [[ $b == /* ]] || b=$PWD/$b
  printf '%s\n' "$b"
}
IVK=$(whole "public folder" "$1") || { echo "$IVK"; exit 2; }
ISK=$(whole "secret folder" "$2") || { echo "$ISK"; exit 2; }
IN=$(whole "folder of -in" "$3") || { echo "$IN"; exit 2; }
OUT=$(whole "folder of -out" "$4") || { echo "$OUT"; exit 2; }
WK=$(whole "work folder" "$5") || { echo "$WK"; exit 2; }   # for its refusals; the path itself comes from the walk below
ivk_real=$(realpath -m -- "$IVK") && isk_real=$(realpath -m -- "$ISK") && in_real=$(realpath -m -- "$IN") && out_real=$(realpath -m -- "$OUT") ||
  no 2 "a folder's name cannot be read as a path"   # every link followed, as the kernel reads them: only for the two comparisons of names below
WK=$5   # the work folder: made whole here, and walked below
[[ $WK == /* ]] || WK=$PWD/$WK
while [[ $WK == */ && $WK != / ]]; do WK=${WK%/}; done
case ${WK##*/} in .|..) no 2 "name the work folder by a name of its own, not «.» or «..»" ;; esac
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
# Runs before anything is created: an account that can change a directory or link above the work
# folder could put another folder in its place. Only what is above is examined here, not the work
# folder. WK becomes the link-free path that open_above() examined plus the last part as typed.
if open_above "${WK%/*}/"; then
  case $why in
  'it does not exist') no 2 "$place, above the work folder, does not exist: make it first; this script makes only the work folder and f<field> in it" ;;
  'it is not a directory' | 'it could not be examined') no 2 "$place, above the work folder: $why: name a work folder below directories that you can read" ;;
  *)
    printf -v named %q "$WK"
    text="the work folder $named lies below $place, which another account may be able to change ($why): it could put another folder in the work folder's place and choose what runs"
    text+=" beside the secrets. Name a work folder below directories that only you or root can change (such as one in your home directory, or in /tmp, whose sticky bit keeps others"
    text+=" from replacing it)$cure"
    no 2 "$text"
    ;;
  esac
fi
WK=${real%/}/${WK##*/}
here=$(cd "$(dirname "$0")" && pwd) && T=$(cd "$here/../.." && pwd) || no 2 "the folder of this script cannot be entered"
# The path to each of the four folders, as typed, is examined from the top before anything there is
# read: a link or directory of another account's on it could lead the audit, and with it the
# erasure, to another folder. The folder itself is examined too. A link earlier on the path is
# followed if it belongs to an owner whom open_above() trusts.
# The secret folder must pass all of open_above()'s tests: a missing part of its path is refused
# too.
# cp_prove examines the path of -isk likewise, and follows no link (Prover::read_isk() in
# prover/prover/isk.cuh).
# The other three are only read, so less is asked: a directory that its group or others can write in
# passes. A part that is missing or is no directory is refused here, and again by the [ -d ] test
# below; so is a public folder without key.bin or leaf/. A verdict NOT WHOLE over one of these
# would, with --erase-kept-files, erase batch.fresh. audit.py itself rejects them too.
# owners_above <absolute path>/: open_above() without the test of modes. It calls open_above(), so
# that both follow links and ".." alike and trust the same owners. open_above() learns a place's
# owner, mode and type from `stat -c '%u %a %F'` alone. During the call a shell function `stat`
# shadows that command: it runs the real stat and prints 700 in place of every mode, so no directory
# is found writable and the walk goes on past it. `unset -f` removes the function right after the
# call, so the other calls of open_above() use the real stat.
owners_above() {
  local found
  stat() { local u m k; IFS=' ' read -r u m k < <(command stat "$@") && printf '%s 700 %s\n' "$u" "$k"; }
  open_above "$1"; found=$?
  unset -f stat
  return "$found"
}
away() { # <which folder> <its path> <what is asked of the way>: says what the walk has found, and ends
  printf -v named %q "$isk_real" # the walk names a place by the path that it has walked, without links
  [ "$place" != "$named" ] || cure='' # no «close it with chmod» for the secret folder itself: what was put into it while it was open would still be there
  printf -v named %q "$2"
  case $why in
  *'another account' | *'no sticky bit')
    no 2 "on the way to the $1 $named, $place: $why: another account may be able to lead this audit to another folder. Name the folder by a path on which $3$cure"
    ;;
  *) no 2 "on the way to the $1 $named, $place: $why. Name a folder that is there and that you can read, in full (a mistyped name?)" ;;
  esac
}
if open_above "$ISK/"; then away "secret folder" "$ISK" "every part is yours or root's and closed to others"; fi
# open_above() asks who can change the folder. cp_prove also asks that nobody else can read it.
case $(command stat -c %a -- "$real") in 0|*00) ;; *) no 2 "the secret folder $ISK: its group or others have a permission on it (mode $(command stat -c %a -- "$real")): cp_prove refuses such a secret folder too: find out who changed the mode and who may have read it before you close it (chmod go-rwx)";; esac
if owners_above "$IVK/"; then away "public folder" "$IVK" "every part is yours or root's"; fi
if owners_above "$IN/"; then away "folder of -in" "$IN" "every part is yours or root's"; fi
if owners_above "$OUT/"; then away "folder of -out" "$OUT" "every part is yours or root's"; fi
for p in "$IVK" "$IN" "$OUT"; do [ -d "$p" ] || no 2 "$p is not there, or is no folder (a mistyped name?): name the folder as cp_commit or cp_prove got it"; done
[ -f "$IVK/key.bin" ] && [ -d "$IVK/leaf" ] || no 2 "$IVK holds no key.bin, or no folder leaf: this is not a public folder of cp_commit's (a mistyped name?)"
[ -s "$ISK/config/policy.txt" ] || no 2 "no $ISK/config/policy.txt: this is not a secret folder of cp_commit's"
case $(head -n 1 "$ISK/config/policy.txt") in
  'version H821') F=2;;
  'version H811') no 2 "$ISK/config/policy.txt: its first line is «version H811», the format of field 1, which this version does not build (blocks/base.h): this tool audits files of version H821 alone";;
  *) no 2 "$ISK/config/policy.txt: its first line names no field that this tool knows";;
esac
case $WK/ in "$ivk_real"/*|"$isk_real"/*|"$in_real"/*|"$out_real"/*) no 2 "$WK is inside one of the four other folders: name a work folder outside them";; esac
for d in "$ivk_real" "$isk_real" "$in_real" "$out_real"; do   # nor the other way round: the helpers are written into the work folder, and a file more among the requests changes the batch
  case $d/ in "$WK"/*) no 2 "$d is inside the work folder: name a work folder that holds none of the four other folders";; esac
done
tree_real=$(cd -- "$T" && pwd -P) || no 2 "the tree's folder cannot be entered"   # nor below the sources from which the helpers are built: what is built there would count as a change of the tree (below)
case $WK/ in "$tree_real"/verifier/src/*|"$tree_real"/blocks/*|"$tree_real"/prover/*) no 2 "$WK lies below verifier/src, blocks or prover of this tree, the sources from which the helpers are built: name a work folder outside them";; esac
# Makes the work folder, then f<field> from inside it, each with mode 700. An existing work folder
# is accepted if it is the caller's own and no link (its mode is not checked); an existing f<field>
# must pass the `find` below. `realpath .` asks the system; `pwd -P` only resolves bash's stored
# path.
mkdir -m 700 -- "$WK" 2> /dev/null || [ -d "$WK" ] || no 2 "$WK cannot be made (is something else there under that name?): name a work folder that does not exist yet"
cd -P -- "$WK" && [ "$(realpath .)" = "$WK" ] && [ -O . ] || no 2 "$WK is a link, or is not your own: name a work folder that does not exist yet"
mkdir -m 700 -- "f$F" 2> /dev/null || [ -d "f$F" ] || no 2 "$WK/f$F cannot be made (is something else there under that name?): name a work folder that does not exist yet"
cd -P -- "f$F" || no 2 "$WK/f$F cannot be entered"
[ "$(realpath .)" = "$WK/f$F" ] && [ -z "$(find . \( ! -uid "$(id -u)" -o -perm /022 -o ! \( -type f -o -type d \) \) -print -quit 2>&1)" ] ||   # «.» is the first that find looks at; a complaint of find's counts as a find
  no 2 "$WK/f$F is a link, or it or something in it is not your own, or is anything other than a plain file or a folder, or its group or others can write it: the helpers there would run beside the secrets, and this script's own writing would follow a link. Name a work folder that does not exist yet"
python3 -B -E -s -c 'import numpy' 2> /dev/null || no 3 "$(command -v python3) cannot import numpy: PYTHONPATH and your user site-packages (pip --user) are ignored here, so numpy must come from the system's packages or from a virtual environment"   # -E -s, here and in the last line: no PYTHON* variable and no user site-packages decide where python3 looks for modules
cannot() { no "${2:-3}" "$1"; }
# The helpers must match the present sources: pubtool writes the plans (each hash chain's rounds and
# messages) by which audit.py parses the files, so helpers built from other sources could end an
# honest audit without a YES.
# Lock: one call at a time uses this folder, from before the sum is computed until audit.py exits; a
# second call is refused. The lock is a `flock` on the folder itself, which the kernel releases when
# the process ends.
# gxx(): the compiler gets PATH and nothing else of the caller's environment (it would search CPATH
# for headers and write a file named in DEPENDENCIES_OUTPUT). Its temporary files go into this
# folder, named as `.` and not by its path: `.` stays the directory that was entered and checked
# above, but an account that can write in the work folder could put another folder at the path.
# Whoever replaces a temporary file chooses code that runs beside the secrets.
# tree_sum(): one SHA-256 over the compiler's version line and over the name and bytes of every file
# below the three -I folders (verifier/src, blocks, prover), of the helpers' two sources and of the
# two scripts that build them. Left out: paths that begin with prover/build, and __pycache__.
# System headers and libraries are not in the sum: a change in them shows only if the compiler's
# version line changes. A symbolic link below the three folders stops the script: the repository has
# none there, and a link's target could change without changing the sum.
# Each call computes the sum. If built-from.sum has the same sum and both helpers are there, they
# are used as they are. Otherwise they are built under temporary names, the sum is computed once
# more, and only if it has not changed are they renamed; built-from.sum is written last. No other
# account except root can put a matching built-from.sum here: this folder is the caller's own, its
# group and others cannot write in it, and it was checked from inside (above).
# The public inputs' folder has the sum's first 16 digits in its name, so inputs made by an earlier
# pubtool are never used. The 128 passed to pubtool is the requested bits: pubtool rejects a
# configuration whose target (`sec` in policy.txt) is lower.
# The script removes nothing: unit.$$.o, a failed build's files and earlier pub.* folders stay. `mv`
# replaces the earlier helpers and built-from.sum; a failed pubtool run's log is overwritten.
exec 9< . || cannot "the helpers' folder $WK/f$F could not be opened for its lock" 2
flock -n 9; case $? in
  0) ;;
  1) cannot "another audit is at work in $WK/f$F: wait until it has ended, or name another work folder" 2;;
  *) cannot "the helpers' folder $WK/f$F could not be locked: name a work folder on a local disk" 2;;
esac
gxx() { env -i "PATH=$PATH" TMPDIR=. g++ "$@" 9<&-; }   # (the lock's descriptor goes to audit.py alone: a helper's child that outlives a kill must not hold the lock)
tree_sum() {
  { gxx --version | head -n 1
    cd -- "$T" && find verifier/src blocks prover \( -path 'prover/build*' -o -name __pycache__ \) -prune -o -type l -printf 'a link: %p\n' &&
      find verifier/src blocks prover tools/audit/pubtool.cpp tools/audit/fieldlib.cpp tools/audit/audit_commitment.sh tools/audit/audit_batch.sh \( -path 'prover/build*' -o -name __pycache__ \) -prune -o -type f -print0 |
      LC_ALL=C sort -z | xargs -0 sha256sum
  } 2>&1 | sha256sum | cut -c1-64
}
[ -z "$(cd -- "$T" && find verifier/src blocks prover \( -path 'prover/build*' -o -name __pycache__ \) -prune -o -type l -print -quit 2>&1)" ] || cannot "there is a link below verifier/src, blocks or prover of this tree (or these folders cannot be read): the helpers are not built from such a tree"
sum=$(tree_sum); [[ $sum =~ ^[0-9a-f]{64}$ ]] || cannot "the sum over the tree's sources could not be made (sha256sum, find, sort, xargs, head and cut are needed)"
if [ ! -x pubtool ] || [ ! -s libfield.so ] || [ "$(cat built-from.sum 2> /dev/null)" != "$sum" ]; then
  [ ! -e pubtool ] || echo "the helpers in this work folder were built from another state of the tree or by another compiler, or nothing here says from which: they are built anew"
  echo "building the verifier as a unit, the tool for the public inputs and the small library (two to three minutes) .."
  : > built-from.sum || cannot "built-from.sum cannot be written in the work folder"   # no sum while the helpers are replaced: a build that is cut short leaves none that fits
  gxx -std=c++17 -O2 -fno-ipa-modref -w -pthread -I"$T/verifier/src" -I"$T/blocks" -I"$T/prover" -DBLK_FIELD=$F -DDIGEST_WORDS=8 -DVF_SOURCE_DIGEST='"0000000000000000000000000000000000000000000000000000000000000000"' -DCP_VERIFY_AS_A_UNIT -c "$T/verifier/src/cp_verify.cpp" -o unit.$$.o &&
    gxx -std=c++17 -O2 -fno-ipa-modref -w -pthread -I"$T/verifier/src" -I"$T/blocks" -I"$T/prover" -DBLK_FIELD=$F -DDIGEST_WORDS=8 "$here/pubtool.cpp" unit.$$.o -o pubtool.$$ &&
    gxx -O2 -fno-ipa-modref -shared -fPIC -o libfield.$$.so "$here/fieldlib.cpp" || cannot "the helpers could not be built (is the C++ compiler there?)"
  [ "$(tree_sum)" = "$sum" ] || cannot "the tree has changed while the helpers were built: nothing of that build is used: run it again when nobody works in the tree"
  mv -T pubtool.$$ pubtool && mv -T libfield.$$.so libfield.so && printf '%s\n' "$sum" > built-from.$$.sum && mv -T built-from.$$.sum built-from.sum || cannot "the helpers could not be put in place in the work folder"
fi
pub=pub.$(cat "$ISK/config/policy.txt" "$ISK/config/layout.txt" "$ISK/config/KL.bin" | sha256sum | cut -c1-16).$(printf %s "$sum" | cut -c1-16)
if [ ! -d "$pub" ]; then   # made under another name and renamed when pubtool succeeds; pubtool does not check its writes to the .tsv lists (128: pubtool rejects a configuration whose `sec` is lower)
  mkdir "$pub.$$" && ./pubtool "$ISK/config" 128 "$pub.$$" > "$pub.log" 2>&1 9<&- && mv -T "$pub.$$" "$pub" || no 3 "the public inputs could not be made (see $WK/f$F/$pub.log)"
fi
AUDIT_LIB=./libfield.so exec python3 -B -E -s "$here/audit.py" "$T" "$pub" - $F --level=1 --batch --ivk="$IVK" --isk="$ISK" --in="$IN" --out="$OUT" ${MORE:+"$MORE"}
