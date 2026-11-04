#!/bin/bash
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.

# Builds weight_sums/weight_sums.cpp into bin/weight_sums at the top of the checkout or, with
# "- <output directory>", into that directory. weight_sums writes the sums file (wsums.txt) that
# cp_commit needs to commit to a real model's weights.
# The script runs from any directory and takes a few seconds. It needs g++ 13 or later (tested with
# g++ 13.3.0 only), GNU coreutils and GNU find, and no GPU.
# It first checks whether another account could change the output directory; a failed check ends it
# with status 2, and nothing is built (the checks, and one exception, are described below).
# An existing weight_sums in the output directory is overwritten, with a notice on stderr; a build
# that fails or is interrupted leaves it unchanged. The script itself changes or removes no other
# existing file. It writes only in the output directory and points g++ there too (-o, TMPDIR). g++
# gets the caller's environment without DEPENDENCIES_OUTPUT and SUNPRO_DEPENDENCIES. Left after
# a run: the program, if it was built; the output directory, if the script created it, even if
# nothing was built; and the work directory weight_sums.building.XXXXXX inside it, if the script did
# not remove it (for example after SIGKILL, or if g++ left a temporary file in it). No later build
# uses or removes that work directory.
# usage: build.sh [- <output directory>]
set -euo pipefail
unset CDPATH # with CDPATH set, a cd below could end in a directory of the same name elsewhere
here=$(cd "$(dirname "$0")" && pwd -P) || exit 2 # without links, as the places that open_above() names
if [[ $# -ne 0 && ! ($# -eq 2 && $1 == - && -n $2) ]]; then
  echo "usage: build.sh [- <output directory>]" >&2
  exit 2
fi
named=${2:-} # empty: the default output directory, inside the checkout
out=${2:-"${here%/*}/bin"}
# The program built here will be run. So the script builds only in an output directory where, by
# owners and permission bits, no other account can add a file or put another directory in its place.
# open_above(), whose text is the same in every script of this repository that has it, checks every
# place (directory or symbolic link) on the path to the directory; then the directory's mode and
# contents are checked. A check that fails ends the script with status 2.
# One exception, for the default output directory only: if another account may be able to change a
# place at or above the top of the checkout, the script prints a note and builds. Such an account
# can change this script and its sources too, so a rejection (refuse()) would protect nothing.
# After the note the rest of the path and the directory's mode are not checked, and a failed check
# of the contents prints only a note.

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
# Called when open_above() has returned 0: prints why nothing was built and exits. In the excepted
# case (`named` and `itself` empty, and `why` is the place's owner or mode) it prints the note, sets
# `excused` and returns.
refuse() {
  if [[ -z $named && -z $itself && $why != it\ * ]]; then
    excused=1
    echo "build.sh: note: another account may be able to change $place ($why), under which this checkout lies, and so could" \
      "change what is built, and this script itself; for a build that no other account can touch, keep the checkout below" \
      "directories that only you can write in$cure" >&2
    return
  fi
  local how='name a directory of your own below directories that only you or root can change (such as one in your home'
  how+=' directory): end the call with "- DIRECTORY"'
  if [ "$why" = 'it does not exist' ]; then
    echo "build.sh: nothing was built: $place, above the output directory, does not exist: make it first; build.sh makes" \
      "only the last part of the path" >&2
  elif [[ $why == it\ * ]]; then # it is not a directory; it could not be examined
    echo "build.sh: nothing was built: the output directory $(printf '%q' "$out") cannot be used: $place: $why: $how" >&2
  else
    echo "build.sh: nothing was built: the output directory $(printf '%q' "$out") is, or lies below, $place, which another" \
      "account may be able to change ($why): it could put another directory in its place while the build runs and choose" \
      "what you later run as weight_sums: $how$cure" >&2
  fi
  exit 2
}
# `itself`: set if the place that open_above() reports is the output directory itself or, for the
# default, not at or above the top of the checkout. `excused`: set once refuse() has printed a note.
itself='' excused='' made='' # made: set once this call has created the output directory
[[ $out == /* ]] || out=$PWD/$out
while [[ $out == */ && $out != / ]]; do out=${out%/}; done
# A missing output directory is created: only the last part of its path, by a plain mkdir, which
# fails if anything has taken the name, and not writable by its group or others. If the caller named
# the directory, its parent must pass open_above() first.
if [ ! -e "$out" ] && [ ! -L "$out" ]; then
  if [ -n "$named" ]; then
    if open_above "${out%/*}/"; then refuse; fi
    out=${real%/}/${out##*/}
  fi
  if ! (umask go-w && mkdir -- "$out"); then
    echo "build.sh: the output directory $(printf '%q' "$out") cannot be created: see the message above. Name another" \
      "directory." >&2
    exit 2
  fi
  made=1
fi
if open_above "$out/"; then
  # No hint to run chmod if the place is the output directory itself: chmod removes nothing that
  # another account has put there.
  if [ "$place" = "$(printf %q "$(realpath -- "$out" 2>/dev/null)")" ] || [ "$place" = "$(printf %q "$out")" ]; then cure='' itself=1; fi
  # For the default, only a place at or above the top of the checkout is excused, not one that a
  # symbolic link at the default's path leads to.
  [[ -n $named || $(printf %q "${here%/*}")/ == "${place%/}"/* ]] || itself=1
  refuse
  real=$(cd -P -- "$out" && pwd) || exit 2 # the default, behind the note
fi
out=$real # absolute, without links: the path that the walk has looked at (behind a note: as cd -P resolves it)
# The sticky bit excuses a writable directory above the output directory, not the output directory
# itself: there another account could still add files beside the program.
mode=$(LC_ALL=C stat -c %a -- "$out") || mode=7777 # if stat fails, the directory is refused
if [ -z "$excused" ] && ((8#$mode & 8#1022)); then
  echo "build.sh: nothing was built: the output directory $(printf '%q' "$out") can be written in by its group or by" \
    "others, or has the sticky bit, or could not be examined: name a new directory of your own: end the call with \"- DIRECTORY\"" >&2
  exit 2
fi
# Rejected contents: anything that another account owns, since what it added while the directory
# was writable is still there; and any directory that its group or others can write in, where
# another account can still add files.
# Trusted owners, as in open_above(): root, and this account or, for root under sudo, the account
# that ran sudo (`upper`). find selects every other owner by "greater than" and "less than", never
# by a negation: if a file vanishes during the search (a second build in the same directory creates
# and removes files), every test on its owner is false, and a negation would turn that into a match.
upper=$(id -u) || upper=0
if [ "$upper" = 0 ] && [[ ${SUDO_UID:-} =~ ^[1-9][0-9]*$ ]]; then upper=$SUDO_UID; fi
# If another build's work directory vanishes during the search, find fails (-ignore_readdir_race
# covers files only). So the search is run up to three times.
found='' searched=''
for _ in 1 2 3; do
  if [ -n "$searched" ]; then break; fi
  if found=$(find "$out" -ignore_readdir_race -mindepth 1 \( -uid +0 -uid "-$upper" -o -uid "+$upper" -o -type d -perm /022 \) -print -quit 2>/dev/null); then
    searched=yes
  elif [ -n "$found" ]; then
    break
  fi
done
if [[ -z $searched || -n $found ]]; then
  name=${found:+ ($(printf '%q' "$found"))}
  found="the output directory $(printf '%q' "$out") holds something that belongs to another account, or a directory that"
  found+=" its group or others can write in$name, or could not be searched"
  if [ -z "$excused" ]; then
    echo "build.sh: nothing was built: $found: name a new directory: end the call with \"- DIRECTORY\"" >&2
    exit 2
  fi
  echo "build.sh: note: $found" >&2 # the default, below an open place: a note, and the build goes on
fi
umask go-w # nothing that the build makes there can be written by its group or by others, whatever the caller's umask
where=$(printf '%q' "$out") # shell-quoted for the messages, so that any name prints on one line
# roots_bin: where the default output directory belongs to root in a checkout of another account's,
# as after a build under sudo, it prints what that means and the ways on that need no root.
# Otherwise it prints nothing, also where the program's source is not in `here`: that directory
# comes from the name by which the script was called, and through a link elsewhere, or from standard
# input, it is not the script's own.
roots_bin() {
  local top=${here%/*} owner
  [[ -z $named && -n $top && $out == "$top/bin" && -f $here/weight_sums.cpp ]] || return 0
  [[ $(stat -c %u -- "$out" 2>/dev/null) == 0 ]] && owner=$(stat -c %u -- "$top" 2>/dev/null) || return 0
  [[ $owner =~ ^[1-9][0-9]*$ ]] || return 0
  printf '%s belongs to root, in %q, which belongs to the account with uid %s. A build that does not run as root' "$where" "$top" "$owner"
  printf ' cannot write into it. To go on without root, name another output directory (end the call with "- DIRECTORY").'
  printf ' The tests and other scripts of the repository look for weight_sums in %s: for them, start again in a new copy of' "$where"
  printf " the repository. That the directory is root's does not protect it: whoever owns %q can replace it, weight_sums included." "$top"
}
# Root has just made the default directory: one note, if it is in a checkout of another account's.
if [[ -n $made && $(id -u) == 0 ]] && told=$(roots_bin) && [[ -n $told ]]; then
  (printf 'build.sh: note: %s\n' "$told" >&2) || true # in a subshell: if nobody reads standard error any more, the build goes on
fi
# g++ writes into a new directory inside the output directory ($work), and a rename then installs
# the program. So no unfinished file is ever at the program's path, and a symbolic link at that path
# is replaced, not followed.
work=$(mktemp -d -- "$out/weight_sums.building.XXXXXX") || {
  told=''
  if [[ $(id -u) != 0 ]]; then told=$(roots_bin) || told=''; fi
  if [[ -n $told ]]; then
    printf 'build.sh: nothing was built (see the message above): %s\n' "$told" >&2
  else
    echo "build.sh: nothing was built: no work directory can be created in $where: see the message above. If you have" \
      "no permission to write there, name another output directory: build.sh - DIRECTORY" >&2
  fi
  exit 2
}
work_id=$(stat -c %d:%i -- "$work") # device and inode
export TMPDIR=$work; unset DEPENDENCIES_OUTPUT SUNPRO_DEPENDENCIES # g++'s temporary files go there; g++ appends to a file that either variable names
# in_work <command>: runs the command inside the work directory, but only if $work still leads to
# the directory created above. So nothing is moved out of, or removed from, a directory that has
# taken its place.
in_work() (cd -- "$work" && [[ $(stat -c %d:%i ./) == "$work_id" ]] && "$@")
# The work directory must be empty, so that finish() removes only what this build created.
if ! found=$(in_work ls -Aq) || [[ -n $found ]]; then
  echo "build.sh: nothing was built: the new work directory $(printf '%q' "$work") is not empty, or it is no longer" \
    "the directory that this build created. It was left as it is. Build again." >&2
  exit 2
fi
# The EXIT trap. It removes the work directory, unless g++ left a temporary file there or another
# directory has taken its place. After a build that failed or was interrupted it prints what is at
# the program's path, so that no one mistakes an earlier weight_sums for this build's. After SIGKILL
# it does not run, and the work directory stays.
finish() {
  local status=$?
  set +e # go on to the end even if a removal fails
  in_work rm -f -- weight_sums && [[ $(stat -c %d:%i -- "$work") == "$work_id" ]] && rmdir -- "$work" ||
    echo "build.sh: the work directory $(printf '%q' "$work") could not be removed: it is not empty, or it is no" \
      "longer the directory that this build created. The build does not need it any more." >&2
  [[ $status -eq 0 ]] && return
  if [[ -e $out/weight_sums || -L $out/weight_sums ]]; then
    echo "build.sh: the build failed (status $status). The weight_sums in $where is unchanged and is not from this" \
      "build. See the messages above, if any, and build again." >&2
  else
    echo "build.sh: the build failed (status $status). There is no weight_sums in $where. See the messages above, if" \
      "any, and build again." >&2
  fi
  exit "$status"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP
# weight_sums converts stored weights to integers by a rule that rounds to binary32 after every
# float operation. So -ffp-contract=off forbids the compiler to merge a product and a sum into one
# operation, and no fast-math option may be added.
# -fno-ipa-modref disables an optimiser pass that has a bug in g++ 13.3.0: it can mark memory that
# a function reads as not read, and the stores into that memory are then silently deleted.
# verifier/tests/compiler_pass_control.sh shows the bug on a short program.
g++ -O2 -fno-ipa-modref -std=c++17 -ffp-contract=off -pthread -Wall -Wextra -o "$work/weight_sums" \
  "$here/weight_sums.cpp"
# From here on SIGINT, SIGTERM, SIGHUP and SIGPIPE are ignored, and so is a failing echo
# ("|| true"): after the rename the script must not end with a failure status.
trap '' INT TERM HUP PIPE
had=0 # 1 if a weight_sums, or a symbolic link of that name, is there
[[ -e $out/weight_sums || -L $out/weight_sums ]] && had=1
in_work mv -fT -- weight_sums "$out/weight_sums"
if [[ $had == 1 ]]; then
  echo "build.sh: overwrote the weight_sums that was already in $where." >&2 || true
fi
