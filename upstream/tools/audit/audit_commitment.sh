#!/bin/bash
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.
#
# Audits one commitment made with `cp_commit -keep-for-the-audit`: audit.py recomputes random values
# in the public files from the secrets and compares them. Run it before any of these files is given
# out. audit.py exits with code 2 unless <secret folder>/isk.txt records synthetic weights.
# Procedure, and what a YES does not cover: README.md, "Audit a commitment before you hand it out".
# usage: audit_commitment.sh <public folder> <secret folder> <work folder> --rehearsal | --erase-kept-files
# --rehearsal: all the checks, and nothing is destroyed. --erase-kept-files: the audit itself, which
# destroys the kept files if every check passes. A call without a fourth word is refused.
# The first two folders are cp_commit's -ivk and -isk. audit.py reads them in place, with the
# configuration in <secret folder>/config. Never copy a secret folder for an audit: the original's
# kept files would remain, and each folder would keep its own count of opened columns
# (`count.bin`), which has a limit.
# When every check has passed, audit.py destroys (overwrites, syncs, unlinks) the kept files
# <secret folder>/audit/key.fresh and entry.<k>.fresh, and nothing else, and then prints YES; a
# journal, a snapshot, flash memory or a backup may still contain their bytes. After exit code 4 or
# 7, some or all of these files may be gone or overwritten. So too if audit.py is killed while it
# destroys them, as by SIGTERM, SIGHUP or SIGKILL (Ctrl-C is caught: exit code 7), or if the
# machine stops then. Without --erase-kept-files it destroys nothing. This script writes only in
# <work folder>/f<field>, after making these two directories if they are missing. It removes
# nothing, not even its own intermediate files, and sends no signal; a rebuild replaces the helpers
# and built-from.sum, and a run of the helper pubtool overwrites its log.
# The script builds audit.py's helpers in f<field> (f2: the sources build one field, number 2,
# Goldilocks): two to three minutes the first time; nothing secret is written there. It rebuilds
# them when their sources or the compiler's version line have changed (see tree_sum()).
# Needs Linux with GNU tools, flock, g++ and python3 with numpy; no GPU and no network. Each build
# compiles verifier/src/cp_verify.cpp, for which verifier/build.sh gives g++ 13 or later and about
# 1.6 GB of memory.
# A run creates nothing outside the work folder. What stays in f<field>, after every ending: the
# helpers (pubtool, libfield.so), built-from.sum, one unit.<pid>.o from each build, and pubtool's
# output directory and log (see `pub`), a new pair for each configuration and each value of
# tree_sum(); after a run that ended early, unfinished files as well. The work folder is the
# caller's to remove.
# The helpers run under the account that reads the secrets, and libfield.so inside audit.py's
# process. So the script exits with code 2 unless: every directory above the work folder exists, and
# open_above() finds none that another account may be able to change; the work folder is outside the
# other two folders and outside verifier/src, blocks and prover of this repository, contains neither
# of the two folders, and is new or the caller's own and no link; f<field> is no link and, with all
# that it contains, is the caller's own and not writable by its group or others, and contains
# nothing but plain files and folders (this script's own writing would follow a link).
# Not checked: who can write to the repository, which supplies audit.py and the helpers' sources;
# the caller's environment, in which PATH, LD_*, BASH_ENV and other variables can name code.
# The public and the secret folder are used by their paths as typed. Before anything in them is
# read, each path is walked from the top down to the folder itself. The script exits with code 2 and
# names the place if it finds: for the secret folder, a directory or link that another account may
# be able to change (open_above()), or a permission of its group or others on the folder itself; for
# the public folder, which is only read, a place that belongs to another account; for either, a
# place that is missing, is no directory or cannot be examined. A link of the caller's own, or
# root's, is followed if the directory that contains it has passed.
# Exit code: audit.py's (0 YES, 1 NO, 2 refused, 4 NOT WHOLE, 5 NOTHING KEPT HERE, 6 a rehearsal
# passed, 7 no verdict); 2 also from this script; 3: a program or numpy is missing, the helpers'
# sources contain a link or changed during the build, or a helper could not be built or run.
# Any other code: no verdict: a signal ended the run (a shell reports 128 plus its number), or
# python3 could not be started.
set -u
umask 077
unset CDPATH
[ $# = 3 ] || [ $# = 4 ] || { echo "usage: audit_commitment.sh <public folder> <secret folder> <work folder> --rehearsal | --erase-kept-files"; exit 2; }
MORE=${4:-}
[ $# = 3 ] || [ -n "$MORE" ] || { echo "audit_commitment.sh: the fourth word is empty (a variable that was not set?): write --rehearsal or --erase-kept-files there. Nothing was looked at or destroyed"; exit 2; }
case $MORE in ''|--rehearsal|--erase-kept-files) ;; *) echo "audit_commitment.sh: the fourth word can only be --rehearsal or --erase-kept-files. Nothing was looked at or destroyed"; exit 2;; esac
# An unquoted variable that is empty vanishes from the call, and the next word takes its place:
# without the work folder's name, the last word would be taken for that name. So a call with one of
# the two words in a folder's place is rejected.
for p in "$@"; do [ "$p" != --rehearsal ] && [ "$p" != --erase-kept-files ] || [ -n "$MORE" ] || { echo "audit_commitment.sh: «$p» stands where a folder must be named (a variable before it that was empty or not set?): name all three folders. Nothing was looked at or destroyed"; exit 2; }; done
[ -n "$MORE" ] || { echo "audit_commitment.sh: the fourth word is missing: say which call this is: --rehearsal (all the checks; nothing is destroyed) or --erase-kept-files (the audit itself: if every check passes, it destroys the kept files in <secret folder>/audit/ and says YES). Nothing was looked at or destroyed"; exit 2; }
# For the same reason, no folder's name may begin with "-".
for p in "${@:1:3}"; do case $p in -*) echo "audit_commitment.sh: «$p» stands where a folder must be named (a variable that was not set?): name all three folders, and write ./ before a name that begins with «-». Nothing was looked at or destroyed"; exit 2;; esac; done
for p in python3 g++ sha256sum find id realpath flock sort xargs head cut; do command -v $p > /dev/null || { echo "$p is not to be found"; exit 3; }; done
[ -n "$3" ] || { echo "the work folder's name is empty (a variable that was not set?): name a work folder of your own that does not exist yet"; exit 2; }
WK=$3   # the work folder: made whole here, and walked below
[[ $WK == /* ]] || WK=$PWD/$WK
while [[ $WK == */ && $WK != / ]]; do WK=${WK%/}; done
case ${WK##*/} in .|..) { echo "name the work folder by a name of its own, not «.» or «..»"; exit 2; } ;; esac
[ -n "$1" ] && [ -n "$2" ] || { echo "the public or the secret folder's name is empty (a variable that was not set?): name the folder in full"; exit 2; }
IVK=$1 ISK=$2   # as typed, made whole without following a link (the current folder changes below): by these names the folders are walked, opened and read, and the kept files destroyed
[[ $IVK == /* ]] || IVK=$PWD/$IVK
[[ $ISK == /* ]] || ISK=$PWD/$ISK
ivk_real=$(realpath -m -- "$IVK") && isk_real=$(realpath -m -- "$ISK") || { echo "realpath could not resolve the public or the secret folder's name. Nothing was looked at or destroyed"; exit 2; }   # every link followed, as the kernel reads them: only for comparisons of names below
[ "$WK" != / ] || { echo "the work folder is «/»: name a work folder of your own that does not exist yet"; exit 2; }   # («/» would also pass the two tests on folders below: their patterns begin with «//»)
[ "$ivk_real" != / ] && [ "$isk_real" != / ] || { echo "the public or the secret folder is «/»: name the folder in full"; exit 2; }
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
# Checked before anything is created: an account that can change a directory above the work folder,
# or a symbolic link on the way to it, could put another directory in the work folder's place.
# open_above() looks only at the directories above; the work folder is checked after its mkdir.
# After this check, WK is the parent's path as walked, without links, and the last part as typed.
if open_above "${WK%/*}/"; then
  case $why in
  'it does not exist') { echo "$place, above the work folder, does not exist: make it first; this script makes only the work folder and f<field> in it"; exit 2; } ;;
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
here=$(cd "$(dirname "$0")" && pwd) || exit 2
T=$(cd "$here/../.." && pwd) || exit 2
# A link or directory of another account's on a folder's path could lead the audit, and the
# destruction of the kept files, to another folder. The secret folder gets open_above()'s full
# check; cp_prove asks this and more of its -isk: a full path and no link (Prover::read_isk()). On
# the public folder's path, a directory that its group or others can write in passes. A part that is
# missing or is no directory is refused, as on the secret folder's path.
# owners_above <absolute path>/: open_above()'s walk, asking only who owns each place. It calls
# open_above(), so that both follow links and ".." alike and trust the same owners. open_above()
# learns a place's owner, mode and kind from one command alone, `stat -c '%u %a %F'`. While
# owners_above() calls it, a shell function named `stat` stands in for the command: it runs the real
# stat, passes on the owner and the kind, and prints 700 for every mode. So open_above() finds no
# directory that its group or others can write in, and walks on to the places below it.
# `unset -f stat` after that call removes the function: nothing else in this script meets it.
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
    { echo "on the way to the $1 $named, $place: $why: another account may be able to lead this audit to another folder. Nothing was read or destroyed. Name the folder by a path on which $3$cure"; exit 2; }
    ;;
  *) { echo "on the way to the $1 $named, $place: $why. Nothing was read or destroyed. Name a folder that is there and that you can read, in full (a mistyped name?)"; exit 2; } ;;
  esac
}
if open_above "$ISK/"; then away "secret folder" "$ISK" "every part is yours or root's and closed to others"; fi
# open_above() asks who can change the folder. cp_prove also asks that nobody else can read it.
case $(command stat -c %a -- "$real") in 0|*00) ;; *) echo "the secret folder $ISK: its group or others have a permission on it (mode $(command stat -c %a -- "$real")): cp_prove refuses such a secret folder too: find out who changed the mode and who may have read it before you close it (chmod go-rwx). Nothing was read or destroyed"; exit 2;; esac
if owners_above "$IVK/"; then away "public folder" "$IVK" "every part is yours or root's"; fi
[ -s "$ISK/config/policy.txt" ] || { echo "no $ISK/config/policy.txt: is this a secret folder of cp_commit's?"; exit 2; }
case $(head -n 1 "$ISK/config/policy.txt") in
  'version H821') F=2;;
  'version H811') echo "$ISK/config/policy.txt: its first line is «version H811», the format of field 1, which this version does not build (blocks/base.h): this tool audits files of version H821 alone"; exit 2;;
  *) echo "$ISK/config/policy.txt: its first line names no field that this tool knows"; exit 2;;
esac
case $WK/ in "$ivk_real"/*|"$isk_real"/*) echo "$WK lies inside the public or the secret folder: name a work folder outside both"; exit 2;; esac
for d in "$ivk_real" "$isk_real"; do   # nor the other way round: the helpers are written into the work folder
  case $d/ in "$WK"/*) echo "$d is inside the work folder: name a work folder that holds neither the public nor the secret folder"; exit 2;; esac
done
tree_real=$(cd -- "$T" && pwd -P) || { echo "audit_commitment.sh: $T cannot be entered. No kept file was looked at or destroyed"; exit 2; }   # nor below the sources from which the helpers are built: what is built there would count as a change of the tree (below)
case $WK/ in "$tree_real"/verifier/src/*|"$tree_real"/blocks/*|"$tree_real"/prover/*) { echo "$WK lies below verifier/src, blocks or prover of this tree, the sources from which the helpers are built: name a work folder outside them"; exit 2; };; esac
# The two levels are made one at a time, each with mode 700, the second from inside the first. One
# that exists already is accepted and checked: the work folder here, f<field> below. From f<field>
# on, the helpers are named relative to the current directory, so that a directory that another
# account later puts at f<field>'s path is not used; and what searches the current directory first
# (`python3 -c` for a module, the linker for a library) searches f<field>.
# `realpath .` asks the system where the process is; bash's `pwd -P` only resolves once more the
# name that bash has stored.
mkdir -m 700 -- "$WK" 2> /dev/null || [ -d "$WK" ] || { echo "$WK cannot be made (is something else there under that name?): name a work folder that does not exist yet"; exit 2; }
cd -P -- "$WK" && [ "$(realpath .)" = "$WK" ] && [ -O . ] || { echo "$WK is a link, or is not your own: name a work folder that does not exist yet"; exit 2; }
mkdir -m 700 -- "f$F" 2> /dev/null || [ -d "f$F" ] || { echo "$WK/f$F cannot be made (is something else there under that name?): name a work folder that does not exist yet"; exit 2; }
cd -P -- "f$F" || { echo "$WK/f$F cannot be entered. No kept file was looked at or destroyed"; exit 2; }
[ "$(realpath .)" = "$WK/f$F" ] && [ -z "$(find . \( ! -uid "$(id -u)" -o -perm /022 -o ! \( -type f -o -type d \) \) -print -quit 2>&1)" ] ||   # «.» is the first that find looks at; a complaint of find's counts as a find
  { echo "$WK/f$F is a link, or it or something in it is not your own, or is anything other than a plain file or a folder, or its group or others can write it: the helpers there would run beside the secrets, and this script's own writing would follow a link. Name a work folder that does not exist yet"; exit 2; }
python3 -B -E -s -c 'import numpy' 2> /dev/null || { echo "$(command -v python3) cannot import numpy: PYTHONPATH and your user site-packages (pip --user) are ignored here, so numpy must come from the system's packages or from a virtual environment. No kept file was looked at or destroyed"; exit 3; }   # -E -s, here and in the last line: no PYTHON* variable and no user site-packages decide where python3 looks for modules
cannot() { echo "$1. No kept file was looked at or destroyed: if there are any, they are still there"; exit "${2:-3}"; }
# The helpers must match the sources: pubtool writes the plans by which audit.py parses the files,
# so helpers built from other sources could end an honest audit without a YES. Every call compares
# tree_sum() with built-from.sum. If they differ or a helper is missing, the helpers are rebuilt
# under temporary names and renamed only if a second tree_sum() gives the same sum; built-from.sum
# is written last. f<field> is the caller's own, not writable by its group or others, and checked
# from inside (above), so no account but the caller's and root can put a matching sum there.
# tree_sum(): one sum over the compiler's version line and the path and bytes of every file below
# the three directories that the compiler searches for included files (-I), the helpers' two sources
# and the two scripts that build them. Left out: paths that begin with prover/build, and __pycache__
# directories. A symbolic link below the three directories stops the script: the repository has
# none, and what a link leads to could change unseen. System headers and libraries are not in the
# sum; of the installed compiler, only its version line is.
# gxx(): the compiler gets PATH and nothing else of the caller's environment: otherwise it would
# search CPATH for headers and write to a file named in DEPENDENCIES_OUTPUT. Its temporary files go
# into f<field>, named as "." (TMPDIR=.) and not by its path. An account that can write in the work
# folder could put another directory at that path, but cannot replace the script's current
# directory; and whoever replaces a temporary file chooses code that runs under the account that
# reads the secrets.
# `pub`: the name of pubtool's output directory ends with the first 16 digits of tree_sum()'s sum,
# so that the output of an older pubtool is never used.
# One call at a time works in f<field>: it holds a lock on the directory itself from before the
# first sum until audit.py has ended, and the kernel drops the lock when the process ends. Nothing
# is removed.
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
  mkdir "$pub.$$" && ./pubtool "$ISK/config" 128 "$pub.$$" > "$pub.log" 2>&1 9<&- && mv -T "$pub.$$" "$pub" || { echo "the public inputs could not be made (see $WK/f$F/$pub.log)"; exit 3; }
fi
AUDIT_LIB=./libfield.so exec python3 -B -E -s "$here/audit.py" "$T" "$pub" - $F --level=1 --commitment --ivk="$IVK" --isk="$ISK" ${MORE:+"$MORE"}
