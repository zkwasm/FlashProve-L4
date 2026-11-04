#!/bin/bash
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.

# Builds the verifier program cp_verify a second time, with clang++ instead of g++
# (verifier/build.sh --clang), and tests that build in three parts, named as in the output:
# `plans`, `made-up` and `world`.
# Why: other tests run only g++'s cp_verify. Reliance on behaviour that C++ leaves open, or a bug in
# one compiler, can show as a difference. Equal answers show nothing beyond these calls. A read of
# an uninitialized local variable cannot show: both builds zero it (-ftrivial-auto-var-init=zero).
# usage: clang_build_test.sh [--field 2] <cp_verify> <new folder> [<fixture> ..]
# --field 2 changes nothing: there is one field, number 2.
# <cp_verify>: verifier/build.sh's g++ build (not checked);
# `gcc_v` in the code, where `v` is the clang++ build.
# <new folder>: must not exist yet; the directory above it must exist, and no other account may be able to change a directory above it (open_above() below). It gets
# the clang++ build, work files and logs, and is left in place.
# <fixture> (`world` in the code and output): config/, ivk/key.bin, ivk/leaf/0.bin, in/example.req,
# out/example.y, out/example.proof, id.txt; verifier/tests/other_parameters_worlds.sh makes them.
# Exit status 0: all as expected; 1: otherwise; 2: bad call, failed build or set-up; 77: no clang++, clang++ --version fails, or what has that name is not clang (it does not define __clang__).
# Started by the runner; by hand: below a directory of your own. (Runner: run_suite.py. The work folder goes there.)
set -uo pipefail
unset CDPATH
made_here() { # <a folder that this script has just made>: it must be this account's, closed to others (mode 700; 2700 below a set-group-ID directory) and empty
  # One cd, then three tests of ".", so that all three see the same directory; if stat or ls fails,
  # made_here() fails too. -P: cd resolves ".." as mkdir does. The slash: a directory named "-" is
  # not taken for the previous directory.
  (cd -P -- "$1/" && [ -O . ] && [[ $(stat -c %a .) == *700 ]] && list=$(ls -A) && [ -z "$list" ]) && return
  echo "clang_build_test.sh: $1 is not the folder that was just made: another was put in its place: name a new folder below folders in which only you can write" >&2
  return 1
}
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
export LC_ALL=C
here=$(cd -- "$(dirname "$0")" && pwd) || exit 2
tree=$(cd -- "$here/../.." && pwd) || exit 2
usage() {
  echo "usage: clang_build_test.sh [--field 2] <cp_verify> <new folder> [<world> ..]" >&2
  exit 2
}
field=2
if [[ ${1:-} == --field ]]; then
  [[ $# -ge 2 && $2 == 2 ]] || usage
  field=$2
  shift 2
fi
[[ $# -ge 2 && -x $1 && -n $2 ]] || usage
if [[ -e $2 ]]; then
  echo "clang_build_test.sh: $2 exists already; name a directory that does not exist yet" >&2
  exit 2
fi
gcc_v=$(cd -- "$(dirname "$1")" && pwd)/${1##*/} || exit 2
new=$2
shift 2
worlds=()
for w in "$@"; do
  [[ -f $w/config/policy.txt && -f $w/ivk/key.bin && -f $w/out/example.proof && -f $w/id.txt ]] || usage
  world=$(cd -- "$w" && pwd) || exit 2
  worlds+=("$world")
done
if ! command -v clang++ >/dev/null || ! clang++ --version >/dev/null 2>&1 || ! printf '#ifndef __clang__\n#error\n#endif\n' | clang++ -x c++ -fsyntax-only - >/dev/null 2>&1; then
  echo "not run: clang_build_test.sh: there is no clang++ on this machine, clang++ --version fails, or the program of that name is not clang (it does not define __clang__)"
  exit 77
fi
case $new in /*) out=$new ;; *) out=$PWD/$new ;; esac; case /$out/ in */-*|*/../*|*[!A-Za-z0-9_./-]*) echo "clang_build_test.sh: the work folder's whole path (the current directory included) may hold letters, digits" \
  "and _ . / - only, and no part of it may be \"..\" or begin with \"-\": name another" >&2; exit 2 ;; esac
# The script exits, with nothing created, if open_above() finds that another account may be able to
# change a directory or a symbolic link above the new folder. Otherwise it uses `real`, the path
# without links.
while [[ $out == */ && $out != / ]]; do out=${out%/}; done
if open_above "${out%/*}/"; then
  case $why in
  'it does not exist') echo "clang_build_test.sh: $place, above the new folder, does not exist: make it first; this script makes only the new folder; nothing was built" >&2 ;;
  'it is not a directory' | 'it could not be examined') echo "clang_build_test.sh: $place, above the new folder: $why; nothing was built" >&2 ;;
  *)
    printf -v named %q "$out"
    echo "clang_build_test.sh: the new folder $named lies below $place, which another account may be able to change ($why): it could put another directory in the new folder's place while" \
      "the script runs and lead what it writes into other files of yours. Name a new folder below directories that only you or root can change (such as one in your home directory," \
      "or in /tmp, whose sticky bit keeps others from replacing it)$cure; nothing was built" >&2
    ;;
  esac
  exit 2
fi
new=${real%/}/${out##*/}
case /$new/ in */-*|*/../*|*[!A-Za-z0-9_./-]*) echo "clang_build_test.sh: the work folder's path without links may hold letters, digits and _ . / - only, and no part of it may begin with \"-\": name another; nothing was built" >&2; exit 2 ;; esac
mkdir -m 700 -- "$new" && made_here "$new" || exit 2 # a plain mkdir fails if anything has appeared at the name since the -e test on it above
out=$(cd -- "$new" && pwd) && [ "$out" -ef "$new" ] || { echo "clang_build_test.sh: cd does not lead to the folder just made: name another" >&2; exit 2; } # -ef: a second look that cd reached the directory just made; $new is absolute and has no links and no "..", so cd cannot be led elsewhere
clang++ --version | head -n 1
if ! "$here/../build.sh" --field "$field" --clang - "$out/bin" >"$out/build.log" 2>&1; then
  tail -n 20 "$out/build.log"
  echo "clang_build_test.sh: the build failed" >&2
  exit 2
fi
echo "built; $(grep -c 'warning:' "$out/build.log") warnings (in $out/build.log)"
v=$out/bin/cp_verify
cd -- "$tree" || exit 2

# `plans` (the loop after both()): planning calls (-plan; given no proof; 0 and 128 are the
# requested bits) on every directory under prover/examples/ with a policy.txt (`kit` in the code),
# and on a synthetic request for each target_* directory that has a layout.txt and a policy.txt
# whose first line is `version H821`.
# Both builds must agree in status and output, except times at line ends. At least two directories
# must be planned at 128 bits (exit status 4, `PLAN ONLY`; `last_status` is the g++ build's).
same=0 differ=0
both() { # <words of a call ..>: the call by both builds; status and output, without the seconds at a line's end, must be the same
  local a=0 b=0
  "$gcc_v" "$@" >"$out/call.gcc.raw" 2>&1 || a=$?
  "$v" "$@" >"$out/call.clang.raw" 2>&1 || b=$?
  sed -E 's/ +[0-9]+\.[0-9]+ s$//; s/(seconds: .*)//; s/, [0-9]+\.[0-9]+ s$//' "$out/call.gcc.raw" >"$out/call.gcc"
  sed -E 's/ +[0-9]+\.[0-9]+ s$//; s/(seconds: .*)//; s/, [0-9]+\.[0-9]+ s$//' "$out/call.clang.raw" >"$out/call.clang"
  last_status=$a
  if [[ $a -eq $b ]] && cmp -s "$out/call.gcc" "$out/call.clang"; then
    same=$((same + 1))
  else
    differ=$((differ + 1))
    cp "$out/call.gcc.raw" "$out/differs.$differ.gcc" && cp "$out/call.clang.raw" "$out/differs.$differ.clang"
    echo "NOT THE SAME: cp_verify $*: status $a by g++, $b by clang++; the two outputs are $out/differs.$differ.gcc and .clang"
  fi
}
planned=0
for policy in prover/examples/*/policy.txt; do
  kit=$(dirname "$policy")
  both -plan accounting "$kit" 0
  both -plan accounting "$kit" 128
  if [[ $last_status -eq 4 ]]; then planned=$((planned + 1)); fi
  both -plan arrays "$kit" 128
  if [[ $(basename "$kit") == target_* && $(head -n 1 "$policy") == "version H8${field}1" && -f $kit/layout.txt ]]; then
    python3 -B prover/tools/make_request.py "$policy" "$out/made-up.$(basename "$kit").req" || exit 2 # a file name for each example configuration: the tool overwrites nothing
    printf '\005\000\000\000' >"$out/made-up.y"
    for more in "-shapes" "-shapes -fields -q" "-arrays"; do
      # shellcheck disable=SC2086 # the options are words
      both -plan request "$policy" "$kit/KL.bin" 128 -layout "$kit/layout.txt" -prompt "$out/made-up.$(basename "$kit").req" -reply "$out/made-up.y" $more
    done
  fi
done
plans=$same
echo "plans: $same planning calls with the same status and output, $differ otherwise; $planned folders are planned at 128 bits"

# `made-up`: verifier/tests/made_up_files_test.py must pass with the clang++ build (4 threads) on
# prover/examples/target_{small,full}_f2. It makes key, leaf and proof files of valid lengths
# that contain no proof, and runs the verification call and the part-calls on them: -key checks only
# the key file, -leaf only a leaf file. The g++ build is not run here; other tests in SUITE.md run
# the same script on it.
made=0
for size in small full; do
  status=0
  python3 -B verifier/tests/made_up_files_test.py "$v" "prover/examples/target_${size}_f2" "$out/made-up.$size" 4 >"$out/made-up.$size.log" 2>&1 || status=$?
  if [[ $status -eq 0 ]]; then made=$((made + 1)); fi
  echo "made-up.$size: status $status | $(tail -n 1 "$out/made-up.$size.log" | cut -c1-200)"
done

# `world`: on each <fixture> (honest files on synthetic weights) both builds must agree likewise on
# the verification call ("whole call" in the output) and on both part-calls, and the g++ build's
# verification call must accept (exit status 0). With no <fixture> nothing is run here, and the
# test can still end with status 0.
good=0
for w in ${worlds[@]+"${worlds[@]}"}; do
  before=$differ id=$(cat "$w/id.txt")
  both "$w/config" "$w/ivk" "$w/in/example.req" "$w/out/example.y" "$w/out/example.proof" -id "$id"
  whole=$last_status
  both -key "$w/config" "$w/ivk/key.bin" -id "$id"
  both -leaf "$w/config" "$w/ivk/key.bin" "$w/ivk/leaf/0.bin" -id "$id"
  if [[ $differ -eq $before && $whole -eq 0 ]]; then good=$((good + 1)); fi
  echo "world $(basename "$w"): the whole call's status $whole by g++; $((differ - before)) of 3 calls not the same"
done
said="no world was named"
if [[ ${#worlds[@]} -gt 0 ]]; then said="${good} of ${#worlds[@]} worlds as expected"; fi
echo "clang_build_test.sh: field $field: $plans planning calls the same, $differ calls not; made-up files: $made of 2 as foretold; $said; the logs are in $out"
[[ $differ -eq 0 && $plans -gt 0 && $planned -ge 2 && $made -eq 2 && $good -eq ${#worlds[@]} ]]
