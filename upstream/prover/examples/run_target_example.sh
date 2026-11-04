#!/bin/bash
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.

# Runs the three programs in turn: cp_commit commits to synthetic (`made-up` in options and output)
# weights and cp_prove answers one request with a proof, both on one GPU; cp_verify checks the
# proof. Each step's command is printed before it runs. No test option is passed: cp_commit draws
# its own seed, so the commitment, its ID and the proof differ from run to run.
# usage: run_target_example.sh <configuration> <new work directory>
# <configuration> (`kit` in this script's names and messages): a directory target_* next to this
# script (prover/examples/TARGET-KITS.md describes each). Its policy.txt must begin with
# `version H821`. target_small_file_f2 has no layout.txt and is rejected. The script has been run
# with target_small_f2_roomy and, not on the released programs, with target_full_f2.
# target_full_f2 is the full-size model (docs/quickstart.md, "The same at full size"). A configuration
# that is larger than the small ones,
# whatever its directory's name, is started only where at least 280 GB of host memory are available:
# else exit status 2, and nothing is created.
# The request is the configuration's own_choices.req, if it has one: a prompt and then the model's
# own first choices. The proof covers each choice and the next token. Otherwise the request
# is a window of synthetic tokens, none of which the proof covers: the proof is about the next token
# only. How own_choices.req was made: prover/examples/TARGET-KITS.md, "The examples' requests".
# <new work directory> (`folder` in some messages): the script creates it, with mode 0700, and exits
# if anything has the name. Its path may contain only letters, digits and / . _ -, no part ".." and
# no part that begins with "-". The directory above it must exist. The work directory is rejected
# if open_above() finds a directory above it that another account may be able to change. The script
# removes nothing: the work directory keeps config/, ivk/ (public), isk/ (the provider's secrets),
# in/, out/, home/, tmp/ and the logs.
# BUILD in the environment: the directory with the three programs. Default: prover/build/hc, where
# prover/build.sh without arguments puts them.
# GPU_RUN in the environment: a command prefix, split at blanks, for cp_commit and cp_prove. It is
# started in the work directory.
# The answer's token and the files' lengths are printed only if cp_verify accepts the proof.
# Exit status: cp_verify's if the run gets that far (0: YES; the rest:
# docs/verifying.md, "Exit codes"); 1 if cp_commit or cp_prove fails, no commitment
# ID is found, cp_prove writes no proof, or the work directory's path no longer leads to it; 2 for a
# wrong call, a missing file or program, too little memory for a large configuration, or a work
# directory that is rejected, cannot be created or was replaced; otherwise the failing command's
# status.
set -euo pipefail
unset CDPATH GREP_OPTIONS # grep before version 3.6 obeys GREP_OPTIONS: --color=always there would spoil what available_memory() reads
here=$(cd -- "$(dirname "$0")" && pwd) || exit 2
tools=$(cd -- "$here/../tools" && pwd) || exit 2
if [ $# -ne 2 ] || [ -z "$2" ] || [[ ! $1 =~ ^target_[A-Za-z0-9_]+$ ]]; then
  echo "usage: run_target_example.sh <a folder target_* of $here> <new work folder>" >&2
  exit 2
fi
kit=$here/$1 work=$2
for file in policy.txt KL.bin layout.txt; do
  if [ ! -r "$kit/$file" ]; then
    echo "run_target_example.sh: $kit/$file is missing: $1 is no kit; see TARGET-KITS.md in $here" >&2
    exit 2
  fi
done
case $(head -n 1 "$kit/policy.txt") in
"version H821") build=${BUILD:-$here/../build/hc} ;;
*)
  echo "run_target_example.sh: the first line of $kit/policy.txt is not 'version H821': these programs take no other kit (TARGET-KITS.md)" >&2
  exit 2
  ;;
esac
for program in cp_commit cp_prove cp_verify; do
  if [ ! -x "$build/$program" ]; then
    echo "run_target_example.sh: $build/$program is missing. Build first: verifier/build.sh; prover/build.sh" >&2
    exit 2
  fi
done
build=$(cd -- "$build" && pwd) || exit 2 # by its full path: the script changes directory below
# The user will later handle the work directory by hand. A quoted "~" or "*" would create a
# directory of that name, which is dangerous to remove; commands read a part that begins with "-" as
# an option; and a path with a part ".." does not lead where it seems to.
if [[ /$work == */-* || /$work/ == */../* || $work == *[!A-Za-z0-9/._-]* ]]; then
  echo "run_target_example.sh: the work directory's path may contain only letters, digits, \"/\", \".\", \"_\" and \"-\", and no part of it may be \"..\" or begin with \"-\";" \
    "nothing was created; name another new directory" >&2
  exit 2
fi
if [ -e "$work" ] || [ -L "$work" ]; then
  echo "run_target_example.sh: $work exists already; name a new folder" >&2
  exit 2
fi
# A full-size run needs more host memory than most machines have, and the programs do not check
# first: a run that exhausts the memory can make the kernel end other processes too.
# So, for a configuration larger than the small ones, the script looks at the available memory before it creates anything, and refuses below the floor.
# Its size is read from policy.txt, not from the directory's name: the line table begins "table b n": the committed table has 2^b rows, each encoded to length 2^n, so it has
# fewer than 2^(b + n) entries. b + n is 31 at the small configurations that this script takes, 34 at target_small_file_f2, 43 at full size. Memory was measured at 43 alone, so
# everything above 34 must meet that one floor, which guards against a run that exhausts the memory.
# What happens below the floor has not been tried.
# MemAvailable counts file cache that can be given up as available.
floor=280 # GB
# available_memory() is a copy of the function in examples/maverick/run.sh: keep the two the same.
# Its loop reads from a pipe, not from a here-string, for the reason given below at GPU_RUN.
# It computes only with whole numbers of at most 15 digits, without a leading zero ($whole): bash
# runs a command written inside a value with which it computes, reads a leading zero as octal, and
# lets a sum overflow without a word. So a limit of 10^15 bytes or more counts as no limit.
# It sets $available, in kB of 1,024 bytes, and $got, which describes it: the smaller of
# MemAvailable and the container's memory limit minus the memory in use. File cache does not count
# as in use. Under cgroup v1, or if the memory in use cannot be read, the limit itself is taken.
# Only a limit in /sys/fs/cgroup is seen: not one on a cgroup above the container's, nor one on a
# job without a cgroup namespace (Slurm). If nothing can be read, $available is 0, which is below
# the floor.
# check() here, unlike the one there, only runs the command line and stores its output in $out.
# `|| true`: under `set -e`, a grep that finds nothing would otherwise end the script.
check() { out=$(eval "$*" 2>&1) || true; }
available_memory() {
  local cgroup=/sys/fs/cgroup line file name value limit='' used='' cache='' free note whole
  line="grep -sE '^(MemAvailable:|inactive_file|active_file) +[0-9]+( kB)?\$' /proc/meminfo"
  line+=" $cgroup/memory.stat; grep -s . $cgroup/memory.max $cgroup/memory.current"
  line+=" $cgroup/memory/memory.limit_in_bytes"
  check "$line"
  available=0 whole='^(0|[123456789][0123456789]{0,14})$'
  while IFS=': ' read -r file name value _; do
    case $file in
    /proc/meminfo) if [[ $value =~ $whole ]]; then available=$value; fi ;;
    */memory.stat) if [[ $value =~ $whole ]]; then cache=$((${cache:-0} + value)); fi ;;
    */memory.max | */memory.limit_in_bytes) limit=$name ;;
    */memory.current) used=$name ;;
    esac
  done < <(printf '%s\n' "$out")
  got="$available kB available"
  if [[ ! $limit =~ $whole ]]; then return; fi # "max" or no such file: no limit is seen
  free=$limit
  note=", the container's memory limit; how much of it is in use was not read"
  if [[ $used =~ $whole && -n $cache ]]; then
    used=$((used < cache ? 0 : used - cache))
    free=$((limit < used ? 0 : limit - used))
    note=" available under the container's memory limit of $((limit / 1024)) kB;"
    note+=' file cache counts as available'
  fi
  if [ $((free / 1024)) -lt "$available" ]; then
    available=$((free / 1024))
    got="$available kB$note"
  fi
}
size=$(awk '$1 == "table" { print $2, $3 }' "$kit/policy.txt")
if [[ ! $size =~ ^[123456789][0123456789]?\ [123456789][0123456789]?$ ]]; then # only two plain numbers may reach the sum below: bash runs a command written inside a value with which it computes, such as x[$(command)]
  echo "run_target_example.sh: the second and third words of the line table in $kit/policy.txt must be whole numbers from 1 to 99: $1 is no kit; nothing was created" >&2
  exit 2
fi
size=$((${size% *} + ${size#* }))
if [ "$size" -gt 34 ]; then
  available_memory
  # $out is what check() stored: the greps' lines, each with its file's name. A figure in GB is printed only where one was read.
  if [[ $out != */proc/meminfo:* ]]; then got="nothing: no line MemAvailable, in kB, could be read in /proc/meminfo, which counts as 0 kB whatever a container's memory limit says"
  else got+=" (about $((available * 1024 / 1000000000)) GB)"; fi
  if [ "$available" -lt $(((floor * 1000000000 + 1023) / 1024)) ]; then
    echo "run_target_example.sh: $1 is a large configuration (the second and third words of its line table add up to $size; at the small ones to at most 34): runs at full size, 43, have used some 270 GB of host memory, and this script starts a large one only where at least $floor GB are" \
      "available. Found: $got. With too little memory the run can exhaust it, and the kernel may then end other processes too. Nothing was" \
      "created. Stop other programs of your own, raise the container's memory limit, or use a larger machine; target_small_f2_roomy shows the same steps at a small size." \
      "The three programs themselves do not look at the memory: whoever knows that this configuration needs less can call them by hand, as the script prints them at a small size" >&2
    exit 2
  fi
fi
# GPU_RUN's words are run as a command, so they come through a pipe, not a here-string: bash before
# version 5.1 writes a here-string to a file in TMPDIR and opens it again by name, and an account that
# can replace files in that directory could put another text in its place.
read -r -a gpu_run < <(printf '%s\n' "${GPU_RUN:-}")
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
# open_above() runs before anything is created. From here on the script uses the path that it has
# checked, which has no links.
[[ $work == /* ]] || work=$PWD/$work
while [[ $work == */ && $work != / ]]; do work=${work%/}; done
if open_above "${work%/*}/"; then
  case $why in
  'it does not exist')
    echo "run_target_example.sh: $place, above the work directory, does not exist: make it first; this script makes only the work directory; nothing was run" >&2
    exit 2
    ;;
  'it is not a directory' | 'it could not be examined')
    echo "run_target_example.sh: $place, above the work directory: $why; nothing was run" >&2
    exit 2
    ;;
  esac
  printf -v named %q "$work"
  echo "run_target_example.sh: the work directory $named lies below $place, which another account may be able to change ($why): it could put another directory in the work" \
    "directory's place while the script runs and lead what it writes into other files of yours. Name a new work directory below directories that only you or root can change (such as one" \
    "in your home directory, or in /tmp, whose sticky bit keeps others from replacing it)$cure; nothing was run" >&2
  exit 2
fi
work=${real%/}/${work##*/}
if [[ $work == *[!A-Za-z0-9/._-]* ]]; then
  echo "run_target_example.sh: the work directory's path without links contains a character other than letters, digits and / . _ -: name another; nothing was run" >&2
  exit 2
fi
# The work directory will hold the provider's secrets, so the script creates it itself, closed to
# other accounts. The earlier test for a free name only gives the clearer message: something could
# take the name after it. mkdir without -p fails if anything has the name, even a symbolic link.
if ! mkdir -m 700 -- "$work"; then
  echo "run_target_example.sh: $work could not be created (see the line above); name another new directory" >&2
  exit 2
fi
# From here on the script works inside the directory and names its files relative to it. A process
# keeps its current directory whatever happens to the names above it, so a change there cannot
# redirect what is written under relative names. First the script checks that it is in a directory
# that looks like the one just created: owned by this account, closed to others, empty. It cannot
# tell that one from another such directory put at the name right after the mkdir
# (docs/limits-all-rows.tsv, rows LIM-51.3 and R2.10).
cd -- "$work" || exit 2
if [ ! -O . ] || [[ $(stat -c %a .) != *700 ]] || ! list=$(ls -A) || [ -n "$list" ]; then # an ls that fails refuses too; *700: below a directory with the set-group-ID bit the mode reads 2700
  echo "run_target_example.sh: $work is not the directory that was just created: another was put in its place; name a new directory below directories in which only you can write" >&2
  exit 2
fi
set -C # a second guard: ">" refuses a name at which a file exists
printf '   $ cd %q\n' "$work"
mkdir config in home tmp
still_here() { # ends the run if the path no longer leads to the directory in which the script works
  [ "$work" -ef . ] && return
  echo "run_target_example.sh: $work NO LONGER LEADS TO THE DIRECTORY IN WHICH THIS RUN WORKS: a directory above it was changed. Do not trust what lies there now" >&2
  exit 1
}
shown() { # prints a command as it could be typed, then runs it
  printf '   $'
  printf ' %q' "$@"
  printf '\n'
  "$@"
}
# Why a response that cp_verify has not accepted goes to nobody.
reason="a response that cp_verify has not accepted can hold unmasked values (docs/limits.md, \"What a provider must do\", duty 6)"
logged() { # <log> <command ..>: the same, with all that the command prints in the log
  local log=$1
  shift
  printf '   $'
  printf ' %q' "$@"
  printf ' > %q 2>&1\n' "$log"
  "$@" >"$log" 2>&1 || {
    echo "run_target_example.sh: this step failed with status $?; the last lines of $work/$log:" >&2
    tail -n 5 "$log" | cut -c1-300 >&2
    if [ "$log" = cp_prove.log ]; then
      echo "run_target_example.sh: cp_prove can fail after it has written a response, and no verifier has seen what lies in $work/out. Give no one anything from it, and keep it as secret as $work/isk: $reason" >&2
    fi
    exit 1
  }
}

echo "1. The public configuration: the parameter file, the scales, the layout of a proof's arrays"
shown cp "$kit/policy.txt" "$kit/KL.bin" "$kit/layout.txt" config/

# With target_small_f2, cp_commit exits with status 7: its worst-case check finds places in the
# forward pass where a prompt could drive a value out of its range. The check passes under the
# quantization scales (KL.bin) of target_small_f2_roomy.
echo "2. cp_commit draws a seed and commits to the weights: ivk/ is for the clients, isk/ stays with the provider"
logged cp_commit.log ${gpu_run[@]+"${gpu_run[@]}"} "$build/cp_commit" -weights made-up-blocks -config config -M 1 -ivk ivk -isk isk
# This script acts as provider and client. A real client must get the commitment ID from a party
# that vouches for the model, before it sends a request, and never from files sent by the provider.
id=$(sed -n "s/^commitment ID: //p" cp_commit.log | tr -d ' ')
if [[ ! $id =~ ^[0-9a-f]{64}$ ]]; then
  echo "run_target_example.sh: $work/cp_commit.log has no line with the commitment's ID (commitment ID: ..)" >&2
  exit 1
fi
echo "   the commitment's ID, to be published: $id"

if [ -r "$kit/own_choices.req" ]; then
  echo "3. The request: a made-up prompt, the model's own first choices after it, and the question whether each of them is that, and which token comes next"
  shown cp "$kit/own_choices.req" in/example.req
else
  echo "3. The request: a window of made-up tokens, and the question which token comes next"
  shown python3 -I -B "$tools/make_request.py" config/policy.txt in/example.req
fi

echo "4. cp_prove runs the model on the request and writes the answer and its proof"
# cp_prove does not check what it writes: step 5 does. Like cp_commit, it rejects a GPU on which
# more than 2 GiB of memory are in use. Of its environment it keeps only HOME, TMPDIR,
# CUDA_VISIBLE_DEVICES and CUDA_DEVICE_ORDER; as root it also reads SUDO_UID once, to check the path
# given with -isk. HOME and TMPDIR must name existing directories by absolute paths of at most 400
# characters (letters, digits, / . _ -); this script checks the characters of the work directory's
# path, not its length. This call names the work directory by its path, so still_here() checks
# that path first.
still_here
logged cp_prove.log ${gpu_run[@]+"${gpu_run[@]}"} env HOME="$work/home" TMPDIR="$work/tmp" "$build/cp_prove" -isk "$work/isk" -in "$work/in" -out "$work/out"
if [ ! -e out/example.proof ]; then
  why="$work/cp_prove.log may say why"
  if [ -e out/example.failed ] && [ -r "$kit/own_choices.req" ]; then
    why="One cause may be this: under the exact-match policy example.failed is its answer where a covered token of the request is not the model's first choice: $1/own_choices.req fits this kit's KL.bin, the model's lines of its policy.txt and made-up weights, and no others (TARGET-KITS.md). $why"
  fi
  holds=$(ls -A out 2>/dev/null | tr '\n' ' ') || true
  echo "run_target_example.sh: cp_prove has made no proof for this request ($work/out ${holds:+holds: }${holds:-is empty or missing}). $why" >&2
  exit 1
fi

echo "5. cp_verify: the client's one call, on public files alone, under the commitment ID; printed below: if cp_verify accepts, the answer and the files' lengths; then its line on what the proof covers, its accounting as one number (where proof of work is counted: with it and, in brackets," \
  "without it) with its caution on such figures, its line on the commitment, its exit status, its line on proof of work, and its last line, the verdict"
printf '   $'
printf ' %q' "$build/cp_verify" config ivk in/example.req out/example.y out/example.proof -id "$id"
printf ' > %q\n' cp_verify.log
status=0
"$build/cp_verify" config ivk in/example.req out/example.y out/example.proof -id "$id" >cp_verify.log 2>&1 || status=$?
# cp_prove does not check what it writes, so nothing from out/ is shown unless cp_verify accepts it
# (docs/limits.md, duty 6). A rejected response stays in the work directory, which only this account
# can open.
if [ "$status" -eq 0 ]; then
  echo "   the answer: token $(od -An -tu4 out/example.y | tr -d ' '); key $(stat -c%s ivk/key.bin), entry $(stat -c%s ivk/leaf/0.bin)," \
    "proof $(stat -c%s out/example.proof) bytes"
fi
# cp_verify's line `covered:`: which positions and tokens of the request the proof covers, and
# whether it covers y, the next token (out/example.y). It too precedes the status line, for the
# reason given in the next comment.
sed -n '/^covered: /p' cp_verify.log | tail -n 1
# Two lines of cp_verify's accounting: the compiled bound as one number, without its explanation,
# and the caution that no such figure is a security level. Where proof of work is counted, the
# number is given with it and, in brackets, without it. The number is never shown without the
# caution. After a NO, cp_verify prints no such number, and neither line is shown. Both precede the
# status line, because programs that call this script read its output's last lines.
number=$(sed -n 's/^\(accounting: as one number, by the compiled bound: .*\): the largest L such that .*/\1/p' cp_verify.log | tail -n 1)
caution=$(sed -n '/no figure here is a security level/p' cp_verify.log | tail -n 1)
if [ -n "$number" ] && [ -n "$caution" ]; then printf '%s\n' "$number" "$caution"; fi
# cp_verify's line `commitment:`, which says that the binding of the weights rests on a theorem
# taken as given. It too precedes the status line.
sed -n '/^commitment: .*; the table has a record: /p' cp_verify.log | tail -n 1
if [ "$status" -ne 0 ]; then
  echo "run_target_example.sh: cp_verify did not accept this response (exit status $status). Give no one anything from $work/out or $work/cp_verify.log, and keep them as secret as $work/isk: $reason" >&2
fi
echo "   status $status"
sed -n '/^proof of work: /p' cp_verify.log | tail -n 1
tail -n 1 cp_verify.log
still_here
exit "$status"
