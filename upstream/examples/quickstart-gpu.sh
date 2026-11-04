#!/bin/bash
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.

# Runs the GPU route of docs/quickstart.md in one call, on one NVIDIA H100. It checks the machine,
# builds the verifier and the GPU programs, and runs the small example on synthetic weights.
# It prints each step's command with the expected and the actual result, and stops at the first
# difference. Exit status: 0 if every step gave the expected result, 1 if one did not, 2 if the
# script could not start.
# On 11 CPU cores a run took 29 to 34 minutes. It needs 10 GB of available memory, 1 GB of free
# disk space in the repository and 2 GB in TMPDIR or /tmp. After step 1 it prints its other needs.
# It writes to bin/ and prover/build/ in the repository, where each build replaces the programs of
# an earlier build (if prover/build.sh fails, prover/build/hc can hold programs of two builds, or
# no cp_verify), and to one new directory, which mktemp creates under TMPDIR or /tmp. That directory
# is HOME and TMPDIR for the later commands and gets a log of all output. The script does not start
# if another account may be able to replace it or a directory above it. No step removes the new
# directory: each run leaves one more, with its logs and what the example wrote.
# The script itself deletes nothing and sends no signal. The builds delete only their own work
# directories and what they replace. The directories that the example passes to cp_commit and
# cp_prove, and the two programs' HOME and TMPDIR, are the new directory or lie below it. cp_commit
# kills only the process groups of the cp_verify runs that it started. Ctrl-C stops
# prover/build.sh where env knows --default-signal (GNU coreutils 8.31 or later). With an older env
# it may not: the script builds cp_prove in a background call of itself, which then ignores SIGINT,
# though the compiler in that call can still die of the signal (prover/docs/build.md, "Build").
# Of the caller's environment only PATH, TMPDIR, CUDA_VISIBLE_DEVICES (which
# selects the GPU) and SUDO_UID reach the commands. For root under sudo, this script, the builds and
# the example let SUDO_UID's account own the directories that they write in, but not the toolkit's.
# As root, under sudo too, all that the run writes belongs to root.
#
# usage: examples/quickstart-gpu.sh
if [ -z "${BASH_VERSION:-}" ]; then
  echo "quickstart-gpu: run this script with bash: bash examples/quickstart-gpu.sh" >&2
  # "return" comes first because "exit" would close a shell that sourced this file.
  # shellcheck disable=SC2317
  return 2 2>/dev/null || exit 2
fi
if [ "${BASH_SOURCE[0]:-}" != "$0" ] || [ $# -ne 0 ]; then
  echo "quickstart-gpu: do not source this script, and give it no arguments" >&2
  # shellcheck disable=SC2317
  return 2 2>/dev/null || exit 2
fi
if [ "${QUICKSTART_GPU_PROCESS:-}" != $$ ]; then # restart once, with a clean environment
  shopt -s execfail
  exec env -i PATH="$PATH" ${TMPDIR+TMPDIR="$TMPDIR"} QUICKSTART_GPU_PROCESS=$$ \
    ${CUDA_VISIBLE_DEVICES+CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"} \
    ${SUDO_UID+SUDO_UID="$SUDO_UID"} /bin/bash -- "${BASH_SOURCE[0]}"
  echo "quickstart-gpu: cannot restart with a clean environment: env was not found" >&2
  exit 2
fi
unset QUICKSTART_GPU_PROCESS
set -uo pipefail
# A caller can close standard output or standard error. The redirections below need both open.
{ true 3>&1; } 2>/dev/null || exec >/dev/null
true 3>&2 || exec 2>/dev/null # no "2>/dev/null" here: it would open the descriptor under test

cannot_start() {
  (echo "quickstart-gpu: $*" >&2)
  exit 2
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
{ cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")/.."; } 2>/dev/null ||
  cannot_start "cannot change to the repository's top directory"
needed=(verifier/build.sh prover/build.sh prover/check_nvcc_version.sh)
needed+=(prover/examples/run_target_example.sh prover/examples/target_small_f2_roomy)
for path in "${needed[@]}"; do
  [ -x "$path" ] || cannot_start "$(pwd)/$path is missing, or may not be run or opened:" \
    "run this script in a complete copy of the repository"
done
# open_above() checks TMPDIR before this script's first here-document: bash before version 5.1
# writes a here-document to a file in TMPDIR (in /tmp, if it cannot write there) and opens it again
# by name. Everything else that the script writes outside the repository goes below one new
# directory in TMPDIR or /tmp. So the script stops if another account may be able to replace what is
# created there or a directory above it; otherwise it uses the path that open_above() examined,
# without links.
bad_tmpdir() {
  cannot_start "TMPDIR must be unset, or the absolute path of an existing directory that you" \
    "can write in, outside the repository"
}
plain() { # <path>: the example takes a work directory only below such a path
  [[ /$1 != */-* && /$1/ != */../* && $1 != *[!A-Za-z0-9/._-]* && ${#1} -le 300 ]] ||
    cannot_start "TMPDIR, as given and with links resolved, may contain only ASCII letters," \
      "digits, \"/\", \".\", \"_\" and \"-\", at most 300 of them, and no part of it may be" \
      "\"..\" or begin with \"-\", because the example's work directory is created under it;" \
      "unset TMPDIR or set it to another directory"
}
tmp=${TMPDIR:-/tmp}
[[ $tmp == /* ]] || bad_tmpdir
plain "$tmp"
if open_above "$tmp/"; then
  if [ "$why" = 'it does not exist' ] || [ "$why" = 'it is not a directory' ]; then bad_tmpdir; fi
  cannot_start "the new directory that this script makes in $tmp would lie below $place, which" \
    "another account may be able to change ($why): it could put another directory in the new" \
    "directory's place while the steps run and choose what they run as you. Set TMPDIR to a" \
    "directory that only you or root can change, below such directories (such as one in your home" \
    "directory, or a /tmp with the sticky bit, which keeps others from replacing what you make in" \
    "it)$cure; nothing was run"
fi
tmp=$real
plain "$tmp"
[[ -w $tmp && $tmp/ != "$PWD"/* ]] || bad_tmpdir
export TMPDIR=$tmp

# The commands of docs/quickstart.md, "3. GPU: a small model", each as printed there, but without
# its line "unset ...": this script already runs with a clean environment.
# examples/quickstart_gpu_test.py compares this list with the page's.
mapfile -t commands <<'END'
W=$(mktemp -d) && WORK="$W/example" && echo "WORK=$WORK"
/sbin/ldconfig -p | grep -E 'libcuda|libcudart|libcublas'
grep -m1 -ow sha_ni /proc/cpuinfo
verifier/build.sh
prover/build.sh
prover/examples/run_target_example.sh target_small_f2_roomy "${WORK:?first run the line with mktemp}"
END
verifier_directory=bin # verifier/build.sh puts cp_verify here
build_directory=prover/build # prover/build.sh writes here
programs=$build_directory/hc # and puts cp_commit, cp_prove and a copy of cp_verify here
good_verdict=YES # the last line that cp_verify prints for a correct proof of the small example
# Used in messages only: prover/check_nvcc_version.sh does the check.
toolkit='CUDA 13.2 Update 2 (nvcc 13.2.86)'
# cp_commit and cp_prove restart themselves without LD_LIBRARY_PATH. So the system's loader must
# find the driver's library through its own configuration. It finds the CUDA toolkit's three in
# the same way, or through a run path (rpath) to the toolkit's directory lib64 or lib, which
# prover/build.sh gives the programs under the conditions in prover/docs/build.md. The names are
# CUDA 13's: a later major version has others.
driver_library=libcuda.so.1
toolkit_libraries=(libcudart.so.13 libcublas.so.13 libcublasLt.so.13)
build_page=prover/docs/build.md
# The line that prover/build.sh prints if it sets no run path begins with these words.
no_run_path='build.sh: warning: the programs get no rpath'
section='docs/quickstart.md, "3. GPU: a small model"'

gpu=${CUDA_VISIBLE_DEVICES-0}
gpu=${gpu%%,*}
if [[ ! $gpu =~ ^[A-Za-z0-9-]+$ ]]; then
  cannot_start "CUDA_VISIBLE_DEVICES is set but does not begin with a GPU's number or UUID" \
    "(under a job scheduler: the job has no GPU)"
fi
export PATH=$PATH:/usr/local/cuda/bin # as in prover/build.sh: used only if PATH has no nvcc
export CUDA_DEVICE_ORDER=PCI_BUS_ID # a GPU's number then means the same to CUDA as to nvidia-smi
again='run this script again as: TMPDIR=DIRECTORY examples/quickstart-gpu.sh (DIRECTORY: the'
again+=' absolute path of an existing directory on a local disk, outside the repository, that only'
again+=' you or root can change)'

step=0 out='' status=0 built='' log='' screen=open
# say <line> ...: appends the lines to the log, once there is one, then prints them. It prints in a
# subshell: if the reader has closed the pipe, as after "| head", SIGPIPE kills only the subshell,
# and the run goes on to its end with a complete log.
say() {
  { printf '%s\n' "$@" >>"${log:-/dev/null}"; } 2>/dev/null
  if [ -n "$screen" ]; then (printf '%s\n' "$@") 2>/dev/null || screen=''; fi
}
announce() {
  step=$((step + 1))
  if [ "$step" -gt 1 ]; then say ''; fi
  say "[step $step] \$ $1"
}
check() { # runs a quick command line: output to `say` and into $out, exit status into $status
  announce "$1"
  out=$(eval "$1" 2>&1)
  status=$?
  if [ -n "$out" ]; then say "$out"; fi
}
run() { # runs a slow command line: each output line to the file $2 and to `say`, $status as above
  local line
  announce "$1"
  eval "$1" 2>&1 | while IFS= read -r line || [ -n "$line" ]; do
    printf '%s\n' "$line" 2>/dev/null
    say "$line" >&3
  done 3>&1 >"$2"
  status=${PIPESTATUS[0]}
}
result() { # result <expected> <got> [<why the script stops here> ...]
  say "expected: $1" "got:      $2"
  if [ $# -gt 2 ]; then
    out="quickstart-gpu: stopped at step $step: ${*:3}${log:+. All output of this run is in $log}"
    screen=''
    say "$out"
    (echo "$out" >&2)
    exit 1
  fi
}
major() { # the number before the first dot of line $1 of $out, or 0
  sed -n "$1s/^\([0-9][0-9]*\)\..*/\1/p" <<<"$out" | grep . || echo 0
}
# cp_prove refuses a GPU on which more than 2 GiB are in use, its own CUDA context included. The
# script asks for less than half of that before cp_prove starts.
check_gpu() {
  local expected='a GPU of compute capability 9.0, as an H100 has, with less than 1024 MiB in use'
  local fields=name,compute_cap,driver_version,memory.used name capability driver used got
  check "nvidia-smi --id=$gpu --query-gpu=$fields --format=csv,noheader,nounits"
  if [ "$status" -ne 0 ]; then
    result "$expected" "exit status $status" "nvidia-smi did not report GPU $gpu" \
      "(a status of 127 means that nvidia-smi was not found); it comes with the NVIDIA driver," \
      "and a container has it only if it was started with the GPU and the driver's utilities$built"
  fi
  IFS=, read -r name capability driver used <<<"$out"
  capability=${capability//[[:space:]]/} driver=${driver//[[:space:]]/} used=${used//[[:space:]]/}
  got="$name, compute capability $capability, driver $driver, $used MiB in use"
  if [ "$capability" != 9.0 ]; then
    result "$expected" "$got" "the GPU programs are built for compute capability 9.0" \
      "and run only on such a GPU"
  elif [[ ! $used =~ ^[0-9]+$ ]]; then
    result "$expected" "$got" "nvidia-smi gives no number for the memory in use on GPU $gpu," \
      "so the script cannot tell whether it is idle$built"
  elif [ "$used" -ge 1024 ]; then
    result "$expected" "$got" "GPU $gpu has 1024 MiB or more in use, and cp_prove refuses a GPU" \
      "that something else uses; CUDA_VISIBLE_DEVICES selects another GPU$built"
  fi
  result "$expected" "$got"
}
# Sets $available, in kB, and $got, a text that describes it. $available is the smaller of
# MemAvailable and what is left under the container's memory limit: the limit minus the memory in
# use, not counting file cache. Under cgroup v1, or if the use cannot be read, the whole limit
# counts as left. Only a limit in /sys/fs/cgroup is seen: not one on a parent of the container's
# cgroup, nor one on a job without a cgroup namespace (Slurm). $whole: see examples/maverick/run.sh.
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

W='' WORK=''
announce "${commands[0]}"
eval "${commands[0]}" >/dev/null # in this shell: it sets W and WORK; `say` prints its one line
status=$?
if [ "$status" -eq 0 ]; then say "WORK=$WORK"; fi
top=$W
if [ "$status" -ne 0 ] || [ ! -d "$top" ]; then
  result 'a new directory' "exit status $status" "mktemp could not create a directory under $tmp"
fi
log=$top/quickstart-gpu.log
{ printf '[step 1] $ %s\nWORK=%s\n' "${commands[0]}" "$WORK" >"$log"; } 2>/dev/null
result 'a new directory' "$top"
export HOME=$top TMPDIR=$top
say "All output of this run also goes to $log."
say "From here on HOME and TMPDIR are $top."

say ''
say "quickstart-gpu needs: one NVIDIA H100 that nothing else uses, and its driver;"
say "$toolkit; the driver's library"
say "registered with the system's loader, and the toolkit's libraries registered too or in its"
say "directory lib64 or lib, if that meets the conditions in $build_page; gcc and g++ 13;"
say "python3, binutils and flock; an x86-64 CPU with the SHA extensions; 10 GB of available"
say "memory; 1 GB of free disk space in the repository and 2 GB in $tmp, both on a local disk."
say "It checks for each of these first, except for those conditions, which the second build checks."
say "It writes to $verifier_directory/ and $build_directory/ in the repository and to $top."
if [ "$EUID" -eq 0 ]; then say "It runs as root, so what it writes will belong to root."; fi
say "The whole run takes about half an hour, nearly all of it the second build."
# Only a note: the run goes on. open_above() checks the directories above the repository. For the
# repository's own directory only a write bit for others counts: group write is common (umask 002).
if ! open_above "${PWD%/*}/" && (((8#$(LC_ALL=C stat -L -c %a -- "$PWD") & 8#1002) == 2)); then
  printf -v place %q "$PWD"
  why='others can write in it, and it has no sticky bit'
  [ ! -O "$PWD" ] || [ "$(id -u)" = 0 ] || cure="; if $place is yours, close it with chmod go-w"
fi
if [ -n "$place" ]; then
  say '' "quickstart-gpu: note: another account may be able to change $place ($why), under which" \
    'this run works, and so could change what the builds run or read; for a run that no other' \
    "account can touch, keep the repository below directories that only you can write in$cure"
fi

check_gpu

expected='a path, a version, and exit status 0'
check 'type -P nvcc && nvcc --version | tail -n 2; prover/check_nvcc_version.sh'
if [ "$status" -ne 0 ]; then
  result "$expected" "exit status $status" "$toolkit is needed;" \
    "prover/build.sh takes the first nvcc on PATH, and /usr/local/cuda/bin/nvcc if PATH has none"
fi
result "$expected" "exit status $status"
IFS= read -r compiler <<<"$out" # the path in the first line of the output

expected='gcc and g++, each 13 (tested: 13.3) or later (untested)'
check 'gcc -dumpfullversion -dumpversion; g++ -dumpfullversion -dumpversion'
got="gcc $(major 1), g++ $(major 2)"
if [ "$(major 1)" -lt 13 ] || [ "$(major 2)" -lt 13 ]; then
  result "$expected" "$got (0 means: not found)" \
    "the first gcc and the first g++ on PATH must both be version 13 (later versions are untested)"
fi
result "$expected" "$got"

expected='eight paths and a version'
check 'type -P python3 ld nm objcopy strings ldd sha256sum flock && python3 --version'
if [ "$status" -ne 0 ]; then
  result "$expected" "exit status $status" \
    "one of these is not on PATH (its path is missing above): python3; ld, nm, objcopy, strings" \
    "(binutils); ldd; sha256sum (coreutils); flock (util-linux); or python3 does not start"
fi
result "$expected" "$(tail -n 1 <<<"$out")"

# As in prover/build.sh: the first of lib64 and lib, next to the compiler's own directory, that
# holds libcudart.so, with links resolved. If neither holds it, the build sets no run path.
beside=$(readlink -m -- "$(dirname -- "$(readlink -e -- "$compiler")")/..")
libraries=''
for name in lib64 lib; do
  if [ -e "$beside/$name/libcudart.so" ]; then
    libraries=$(readlink -m -- "$beside/$name")
    break
  fi
done
expected="a line for $driver_library; for each of ${toolkit_libraries[*]},"
expected+=" a line${libraries:+ or a file in $libraries}"
check "${commands[1]}"
if ! grep -qF "$driver_library " <<<"$out"; then
  result "$expected" "no line for $driver_library" \
    "the system's loader does not list $driver_library (ldconfig -p) and must find it without" \
    "LD_LIBRARY_PATH; the library comes with the driver (in a container: with the GPU);" \
    "see $section"
fi
unlisted='' missing=''
for library in "${toolkit_libraries[@]}"; do
  if grep -qF "$library " <<<"$out"; then continue; fi
  unlisted+=" $library"
  { [ -n "$libraries" ] && [ -e "$libraries/$library" ]; } || missing+=" $library"
done
if [ -n "$missing" ] && [ -z "$libraries" ]; then
  result "$expected" "no line for:$missing" \
    "cp_commit and cp_prove would not start: they need$missing from the CUDA toolkit; the" \
    "loader's list (ldconfig -p) does not have these, and prover/build.sh would set no run path," \
    "because libcudart.so is neither in $beside/lib64 nor in $beside/lib; use a toolkit that has" \
    "libcudart.so and these libraries in its directory lib64 or lib (the build uses the toolkit" \
    "whose compiler comes first on PATH); see $build_page"
elif [ -n "$missing" ]; then
  result "$expected" "in neither:$missing" \
    "cp_commit and cp_prove would not start: they need$missing from the CUDA toolkit, and these" \
    "are neither in the loader's list (ldconfig -p) nor in $libraries, where the run path that" \
    "prover/build.sh sets would lead; use a toolkit that has them beside libcudart.so, in its" \
    "directory lib64 or lib (the build uses the toolkit whose compiler comes first on PATH); see" \
    "$build_page"
fi
if [ -z "$unlisted" ]; then
  result "$expected" 'all are listed'
else
  result "$expected" "a line for $driver_library; only in $libraries:$unlisted"
  say "cp_commit and cp_prove will find these only through a run path (rpath) to that directory,"
  say "which prover/build.sh sets under the conditions in $build_page. If it sets none, it says"
  say "so only when it ends, in about half an hour, and this script then stops."
fi

check "${commands[2]}"
if [ "$out" != sha_ni ]; then
  result sha_ni "${out:-nothing}" "this CPU, or this virtual machine, does not offer" \
    "the SHA extensions, which the GPU programs need: use another machine"
fi
result sha_ni "$out"

expected="at least 10000000 kB available, on the machine and under the container's memory limit,"
expected+=' if there is one'
available_memory
if [ "$available" -lt 10000000 ]; then
  result "$expected" "$got" "10 GB of memory must be available" \
    "(for the second build), and this machine or container has less: stop other programs of" \
    "your own, raise the container's memory limit, or use a larger machine"
fi
result "$expected" "$got"

expected="at least 2000000 kB available in $tmp and 1000000 kB in the repository"
check "df -k --output=avail $tmp ."
read -r in_tmp in_repository < <(tail -n +2 <<<"$out" | tr '\n' ' ')
got="${in_tmp:-no number} kB and ${in_repository:-no number} kB"
if [ "$status" -ne 0 ] || [[ ! $in_tmp =~ ^[0-9]+$ || ! $in_repository =~ ^[0-9]+$ ]] ||
  [ "$in_tmp" -lt 2000000 ] || [ "$in_repository" -lt 1000000 ]; then
  result "$expected" "$got" "$tmp needs 2 GB of free disk space and the repository's file" \
    "system 1 GB; with too little in $tmp, $again"
fi
result "$expected" "$got"

# cp_commit, cp_prove and prover/build.sh lock directories with flock(), which some network file
# systems do not support for a directory.
expected='exit status 0: both directories can be locked'
# shellcheck disable=SC2016
check 'flock -n "$W" true && flock -n . true'
if [ "$status" -ne 0 ]; then
  result "$expected" "exit status $status" "$top or the repository cannot be locked with flock," \
    "as on some network file systems: both must be on a local disk; for the first, $again"
fi
result "$expected" "exit status $status"

# Tries the compiler, loader, driver and GPU on a tiny program before the long build. It is linked
# to the static CUDA runtime, so the driver's library is the only CUDA library that it loads.
say ''
say "A short program that calls the CUDA runtime goes to $top/cuda_check.cu."
cat >"$top/cuda_check.cu" <<'END'
#include <cstdio>
#include <cuda_runtime.h>
int main() {
  int count = 0;
  void *p = nullptr;
  cudaError_t e = cudaGetDeviceCount(&count);
  if (e == cudaSuccess) e = cudaSetDevice(0);
  if (e == cudaSuccess) e = cudaMalloc(&p, 1 << 20);
  if (e == cudaSuccess) e = cudaFree(p);
  puts(e == cudaSuccess ? "the CUDA runtime reached the GPU" : cudaGetErrorString(e));
  return e != cudaSuccess;
}
END
expected='the CUDA runtime reached the GPU'
# This script runs this check before it compiles with nvcc.
# A step above has passed the same check. Its note on a later nvcc is printed again, not logged.
prover/check_nvcc_version.sh || exit 1
# shellcheck disable=SC2016
compile='nvcc -std=c++17 --gpu-architecture=sm_90a --cudart=static -o "$W/cuda_check"'
compile+=' "$W/cuda_check.cu"'
check "$compile && \"\$W/cuda_check\""
if [ "$status" -eq 126 ]; then
  result "$expected" "exit status $status" "$tmp does not allow programs to run: $again"
elif grep -q 'driver version is insufficient' <<<"$out"; then
  result "$expected" "exit status $status" "the $driver_library that the system's loader finds is" \
    "older than CUDA 13 needs, or cannot be loaded; one that is found only through" \
    "LD_LIBRARY_PATH, as with NVIDIA's compatibility package, does not reach the programs;" \
    "see $build_page"
elif [ "$status" -ne 0 ] || [ "$(tail -n 1 <<<"$out")" != "$expected" ]; then
  result "$expected" "exit status $status" "$top/cuda_check.cu could not be compiled or run here;" \
    "the lines above give the compiler's, the loader's or the driver's reason"
fi
result "$expected" "$(tail -n 1 <<<"$out")"

say ''
say "Next: two builds. The first takes a minute or two. The second takes about half an hour and"
say "prints little but compiler warnings, which are not failures: this script checks each build's"
say "exit status."
expected="exit status 0, and the program $verifier_directory/cp_verify"
run "${commands[3]}" "$top/verifier-build.log"
got="exit status $status"
[ -x "$verifier_directory/cp_verify" ] || got+=', no cp_verify'
if [ "$got" != 'exit status 0' ]; then
  result "$expected" "$got" "the verifier's build failed; its output is in $top/verifier-build.log"
fi
result "$expected" "$got, and the program"

# cp_commit starts cp_verify with an empty environment. A g++ outside the system's
# directories can build a program that needs a newer libstdc++ than the system's loader finds.
expected='a path for libstdc++, and no "not found"'
check "ldd $verifier_directory/cp_verify 2>&1 | grep -E 'libstdc|not found'"
if [ -z "$out" ] || grep -q 'not found' <<<"$out"; then
  result "$expected" 'not found, or no line' "without LD_LIBRARY_PATH the system's loader finds" \
    "no libstdc++, or too old a one, for a program that this g++ builds: use the system's g++, or" \
    "one whose libstdc++ the loader finds"
fi
result "$expected" 'a path'

expected="exit status 0, and the programs cp_commit, cp_prove and cp_verify in $programs"
run "${commands[4]}" "$top/prover-build.log"
got="exit status $status"
for program in cp_commit cp_prove cp_verify; do
  [ -x "$programs/$program" ] || got+=", no $program"
done
if [ "$got" != 'exit status 0' ]; then
  result "$expected" "$got" "the GPU programs' build failed; its output is in $top/prover-build.log"
fi
result "$expected" "$got, and the three programs"

expected='a path for each CUDA library that cp_commit and cp_prove need, and no "not found"'
check "ldd $programs/cp_commit $programs/cp_prove 2>&1 | grep -E 'libcuda|libcublas|not found'"
if [ -z "$out" ] || grep -q 'not found' <<<"$out"; then
  warning=$(grep -m 1 "^$no_run_path" "$top/prover-build.log" 2>/dev/null)
  warning=${warning%.} # `result` ends the sentence
  result "$expected" 'not found, or no line' "without LD_LIBRARY_PATH the loader does not find a" \
    "library that the programs need, so they cannot start;" \
    "${warning:+the second build printed: }${warning:-see $build_page}"
fi
result "$expected" 'a path for each'

built="; the builds are complete, so the line with mktemp and the last command of $section,"
built+=' can be run by themselves'
check_gpu # again: the builds took a while

expected="exit status 0, and from cp_verify: $good_verdict"
run "${commands[5]}" "$top/example.log"
verdict=$(tail -n 1 "$WORK/cp_verify.log" 2>/dev/null)
if [ "$status" -ne 0 ] || [ "$verdict" != "$good_verdict" ]; then
  result "$expected" "exit status $status, and from cp_verify: ${verdict:-nothing}" \
    "the example failed: give no one anything from $WORK/out, nor the logs named here." \
    "All that the three programs printed is in $WORK:" \
    "cp_commit.log, cp_prove.log, cp_verify.log$built"
fi
result "$expected" "exit status $status, and from cp_verify: $verdict"

# The commitment ID is the only string of 64 hexadecimal digits that the example prints.
id=$(grep -oE '\<[0-9a-f]{64}\>' "$top/example.log" | sort -u)
if [[ ! $id =~ ^[0-9a-f]{64}$ ]]; then id="in the example's output above"; fi
say ''
say "The proof is $WORK/out/example.proof."
say "The commitment ID is $id."
say "quickstart-gpu: all steps ended as expected; your files are in $top"
