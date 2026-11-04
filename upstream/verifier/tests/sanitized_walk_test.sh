#!/bin/bash
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.

# Builds the verifier program cp_verify with gcc's address and undefined-behaviour sanitizers, from
# unchanged sources, and runs it on hostile files: ill-formed, cut, lengthened, or with extreme
# words and numbers.
#
# Without --world it makes only planning calls (-plan), which check no proof and read no file made
# by a prover. parameter_walk.py writes parameter files (policy.txt) as a malicious provider might.
# Then, for the first config folder only, extreme_files_test.py alters the four other files that a
# planning call reads: KL.bin (quantization scales), layout.txt (where a proof places each array),
# a request and a token file (`y`).
# With --world it runs neither script. extreme_proof_files_test.py makes checking calls on honest
# files (a commitment's key file and one leaf file, a proof, its request and token file), which must
# pass, and then on altered copies, which must not.
#
# A read or write past the end of an array, which can go unnoticed while the bytes after it do no
# harm, ends a call here with a report; so does a shift or an overflow of a signed number that C++
# leaves undefined. Four negative controls, each with a bug, must be reported.
# The test cannot see an overflow inside one object, a data race, a read of memory that was never
# written, a floating-point number converted to an integer type that cannot hold it (undefined in
# C++; g++ 13's -fsanitize=undefined leaves that check out), or a code path that no file here takes.
# Tried on weakened copies of the sources once, by hand, in a build for another field, never for the
# field that this script builds. Without the check of KL.bin's length in cp_verify.cpp,
# extreme_files_test.py counted faults. Without the check in Reader::take() (blocks/iop/message.h)
# that n bytes are left, extreme_proof_files_test.py counted cut key files as not-checked, and the
# address sanitizer reported reads past the end of a heap buffer.
# Started by the runner; by hand: below a directory of your own. (Runner: run_suite.py.)
# usage: sanitized_walk_test.sh [--field 2] [--jobs <n>] [--steps <n>] <new folder> <config folder> ..
#    or: sanitized_walk_test.sh [--field 2] [--jobs <n>] --world <fixture> <new folder>
# <new folder> must not exist; the script creates it, after any missing parent folders (status 2 if
# it cannot), and does not remove it. It makes those parents under umask 077, to close them to
# other accounts.
# --field 2 changes nothing: there is one field, number 2 (p = 2^64 - 2^32 + 1).
# --jobs: calls at a time (default 8; a call can take up to 12 GB before a memory bound ends it).
# --steps: steps of parameter_walk.py's walk, a random search (default 60).
# <fixture> (`world` in the code and the printed usage): honest files made from synthetic weights;
# the docstring of extreme_proof_files_test.py lists them.
# Needs g++ 13 or later with its libasan and libubsan, and python3.
# Exit status 0: no fault found, all four controls reported; 1: otherwise; 2: a wrong call or a
# failed build; 3: an altered file passed (<new folder>/files.txt says WAS NOT REFUSED); 77: this
# machine's g++ cannot build with the sanitizers.
set -uo pipefail
unset CDPATH
export LC_ALL=C
here=$(cd -- "$(dirname "$0")" && pwd) || exit 2
usage() {
  echo "usage: sanitized_walk_test.sh [--field 2] [--jobs <n>] [--steps <n>] <new folder> <config folder> .." >&2
  echo "   or: sanitized_walk_test.sh [--field 2] [--jobs <n>] --world <world> <new folder>" >&2
  exit 2
}
field=2 jobs=8 steps=60 world=
while [[ ${1:-} == --* ]]; do
  [[ $# -ge 2 ]] || usage
  case $1 in
  --field) [[ $2 == 2 ]] || usage; field=$2 ;;
  --jobs) [[ $2 =~ ^[1-9][0-9]?$ ]] || usage; jobs=$2 ;;
  --steps) [[ $2 =~ ^[1-9][0-9]{0,3}$ ]] || usage; steps=$2 ;;
  --world) [[ -f $2/config/policy.txt && -f $2/out/example.proof ]] || usage; world=$(cd -- "$2" && pwd) || usage ;;
  *) usage ;;
  esac
  shift 2
done
if [[ -n $world ]]; then
  [[ $# -eq 1 && -n $1 && $1 != - ]] || usage
else
  [[ $# -ge 2 && -n $1 && $1 != - ]] || usage
fi
if [[ -e $1 ]]; then
  echo "sanitized_walk_test.sh: $1 exists already; name a directory that does not exist yet" >&2
  exit 2
fi
new=$1
shift
if [[ $here$PWD$new == *[!A-Za-z0-9_./-]* ]]; then
  echo "sanitized_walk_test.sh: a folder name contains a character other than letters, digits and _ . / -" >&2
  exit 2
fi
configs=() # by their whole paths: the script changes directory below
for c in "$@"; do
  c=$(cd -- "$c" && pwd) || usage
  configs+=("$c")
done
(umask 077 && mkdir -p -- "$(dirname -- "$new")") && mkdir -m 700 -- "$new" || exit 2 # the mkdir without -p fails if anything has appeared at the name since the -e test on it above
# From here on the script works inside <new folder>, under relative names. A process keeps its
# current directory even if a directory above it is renamed or replaced, so the files that the
# script itself writes, and the programs built here that it itself starts, are those in this folder.
# cd would take a lone "-", rejected above, for the previous directory; with -P it resolves ".."
# after a symbolic link as mkdir did. Before anything is written, the folder must belong to this
# account, have mode 700 (2700 below a set-group-ID directory) and be empty.
# The sanitizers' reports (log_path below), temporary files (TMPDIR below) and the three Python
# tests, which pass ./cp_verify and their work folder through os.path.abspath(), still go by the
# whole path: a directory renamed or replaced above this folder can lead them elsewhere.
cd -P -- "$new" && out=$(realpath .) && [ -O . ] && [[ $(stat -c %a .) == *700 ]] && list=$(ls -A) && [ -z "$list" ] ||
  { echo "sanitized_walk_test.sh: cd does not lead to the folder just made: name another" >&2; exit 2; }
if [[ $out == *[!A-Za-z0-9_./-]* ]]; then # this path, with links resolved, goes into the C text of settings.c below
  echo "sanitized_walk_test.sh: a folder name contains a character other than letters, digits and _ . / -" >&2
  exit 2
fi
mkdir reports tmp || exit 2
# TMPDIR becomes a directory inside the one just made, which is closed to other accounts: bash may
# write a here-document to a file in TMPDIR and open it again by name (before version 5.1 it always
# does), and an account that can replace files there could change the five sources below, which are
# compiled and run.
export TMPDIR=$out/tmp
flags=(-O1 -g -fno-ipa-modref -D_GLIBCXX_ASSERTIONS -fno-omit-frame-pointer -fsanitize=address,undefined -fno-sanitize-recover=all
  -std=c++17 -pthread)
# Of ../build.sh's flags, -O2 becomes -O1 -g, and the warning flags and every hardening flag but
# -D_GLIBCXX_ASSERTIONS are left out; -fno-ipa-modref stays. -D_GLIBCXX_ASSERTIONS ends the program
# at an index past a vector's size, also where the memory at that index is the vector's own and the
# address sanitizer sees nothing.
# The sanitizer settings are linked in, so that they also apply where a script starts cp_verify with
# its own environment. A report ends the process with status 97, which cp_verify does not use.
# The address sanitizer writes its report to a separate file under reports/, also from the child
# processes in which cp_verify runs its stages. It also reports an abort, which is how an assertion
# of the C++ library ends a process (handle_abort). With both sanitizers in one program, g++ 13's
# undefined-behaviour sanitizer writes its report, a line with "runtime error:", to the standard
# error stream and ignores log_path.
# Leak detection is off: memory that cp_verify still holds when it ends is not treated as a bug.
# A parameter file with huge values can ask for huge arrays. The two bounds on memory are meant to
# keep such a call from harming the machine; a call that either ends leaves a report like any other.
# The two log paths are absolute, so that a report reaches reports/ whatever the current directory
# of the process that writes it.
cat >settings.c <<END
const char *__asan_default_options(void) {
  return "detect_leaks=0:exitcode=97:log_path=$out/reports/address:handle_abort=1:max_allocation_size_mb=8192:hard_rss_limit_mb=12288";
}
const char *__ubsan_default_options(void) { return "print_stacktrace=1:exitcode=97:log_path=$out/reports/undefined"; }
END
gcc -fno-ipa-modref -c -o settings.o settings.c || exit 2
if ! echo 'int main() { return 0; }' | g++ "${flags[@]}" -x c++ -o nothing - 2>/dev/null; then
  echo "not run: this machine's g++ cannot build a program with the two sanitizers (libasan and libubsan are needed)"
  exit 77
fi


# The negative controls, built with the same flags: a read one word past the end of a vector; a copy
# of ten words into an array of eight; a shift of a 32-bit number by 32 places; an index past a
# vector's size but inside its own memory. `volatile` hides each bug from the compiler.
cat >control_read.cpp <<'END'
#include <vector>
int main() {
  std::vector<unsigned> v(8, 1);
  volatile unsigned long i = 8;
  return (int)v.data()[i];
}
END
cat >control_copy.cpp <<'END'
#include <cstring>
static void copy(unsigned *to, const unsigned *from, volatile unsigned long words) {
  memcpy(to, from, 4 * words);
}
int main() {
  unsigned from[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, to[8];
  copy(to, from, 10);
  return (int)to[0] - 1;
}
END
cat >control_shift.cpp <<'END'
int main() {
  volatile unsigned places = 32;
  unsigned one = 1;
  return (int)(one << places);
}
END
cat >control_index.cpp <<'END'
#include <vector>
int main() {
  std::vector<unsigned> v;
  v.reserve(16);
  v.resize(8, 1);
  volatile unsigned long i = 8;
  return (int)v[i];
}
END
seen=0
for control in control_read control_copy control_shift control_index; do
  g++ "${flags[@]}" -o "$control" "$control.cpp" settings.o || exit 2
  "./$control" >"$control.txt" 2>&1
  status=$?
  sign='ERROR: AddressSanitizer'
  if [[ $control == control_shift ]]; then sign='runtime error: shift exponent'; fi
  if [[ $status -eq 97 ]] && grep -a -q -s "$sign" "$control.txt" reports/*; then
    seen=$((seen + 1))
    echo "$control: reported, as expected"
  else
    echo "$control: not reported (status $status)"
  fi
  mkdir -p "controls/$control" && mv reports/* "controls/$control/" 2>/dev/null
done


# The -D settings below are as in ../build.sh, except that 64 zeros replace the source digest.
word=$(printf '0%.0s' {1..64})
if ! g++ "${flags[@]}" -I"$here/../src" -I"$here/../../blocks" -I"$here/../../prover" -DBLK_FIELD=$field -DBLK_COUNT_HASH_CALLS=1 -DSERVING_BUILD=1 -DDIGEST_WORDS=8 \
  -DVF_SOURCE_DIGEST="\"$word\"" -o cp_verify "$here/../src/cp_verify.cpp" settings.o; then
  echo "sanitized_walk_test.sh: the build failed" >&2
  exit 2
fi
if [[ -n $world ]]; then
  status=0
  python3 -B "$here/extreme_proof_files_test.py" ./cp_verify files "$world" --jobs "$jobs" >files.txt 2>&1 || status=$?
  tail -n 20 files.txt | cut -c1-300
  left=$(find reports -type f | wc -l)
  if [[ $status -eq 0 && $left -eq 0 && $seen -eq 4 ]]; then
    echo "sanitized_walk_test.sh: field $field: checking calls: no report, 4 of 4 controls reported: all as expected"
    exit 0
  fi
  echo "sanitized_walk_test.sh: field $field: checking calls: status $status, $left report files, $seen of 4 controls reported: NOT AS EXPECTED; reports in $out/reports"
  cat reports/* 2>/dev/null | grep -a -m 3 -E 'ERROR: AddressSanitizer|runtime error:' | cut -c1-200
  if [[ $status -eq 3 ]]; then exit 3; fi
  exit 1
fi
# parameter_walk.py ends with status 1 if a call was ended by a signal or with a status that it does
# not expect of cp_verify, such as a sanitizer's 97, or if a report file is left. Neither a status
# 97 nor a report file counts if the report is on a linked-in memory bound or says "out-of-memory".
# Calls that it lists as `unchecked`, `slow` or `large` do not change its status.
status=0
python3 -B "$here/parameter_walk.py" ./cp_verify walk "${configs[@]}" --sanitized --steps "$steps" --wide 16 --jobs "$jobs" --seconds 300 --mb 8192 >walk.txt 2>&1 || status=$?
grep -v '^form changed' walk.txt | tail -n 40
# extreme_files_test.py makes no exception for a memory bound. It also ends with status 1 on a call
# with status 5 (`NOT CHECKED`) or an "internal-" rejection, a call of over 300 s, or, on the
# unaltered files, a status other than 4 (a plan). Its calls write their reports where the walk's
# are, so only new report files are counted.
before=$(find reports -type f | wc -l)
extremes=0
python3 -B "$here/extreme_files_test.py" ./cp_verify extremes "${configs[0]}" --jobs "$jobs" >extremes.txt 2>&1 || extremes=$?
tail -n 20 extremes.txt | cut -c1-300
left=$(($(find reports -type f | wc -l) - before))
if [[ $extremes -ne 0 || $left -ne 0 ]]; then status=1; fi
if [[ $status -eq 0 && $seen -eq 4 ]]; then
  echo "sanitized_walk_test.sh: field $field: no report, 4 of 4 controls reported: all as expected"
  exit 0
fi
echo "sanitized_walk_test.sh: field $field: the walk's and the other files' status $status (the other files' own: $extremes; $left report files of theirs), $seen of 4 controls reported: NOT AS EXPECTED; reports in $out/reports"
cat reports/* 2>/dev/null | grep -a -m 3 -E 'ERROR: AddressSanitizer|runtime error:' | cut -c1-200
exit 1
