#!/bin/bash
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.

# Takes the public Llama 4 Maverick weights from download to a checked proof, in stages: usage()
# lists them and the options, and stage_<name>() runs one.
# Described in README.md next to it. docs/quickstart.md, "4. Llama 4 Maverick: the full-size model",
# has the lines that run it.
# With --width 8 or 16 the stages use the same publisher's 8-bit (Q8_0) or 16-bit (BF16) files: see
# "case $width" below, the directories 8-bit/ and 16-bit/ next to this script, and README.md, "The
# 8-bit and the 16-bit files".
# The seed file, public-seed.bin, is published, so no commitment that this script generates hides
# anything of the weights. The script prints each step's command, the expected and the actual
# result, and stops at the first step where they differ. Exit status: 0 as expected; 1 stopped;
# 2 wrong command line, directory or limits file.
# It needs one NVIDIA H100 with 80 GB on which nothing else is running, hundreds of GB each of host
# memory and disk space, and hours. figures.txt has the amounts; "needs" prints them with each
# stage's tools, and checks the machine for all of these but build's tools and the hard links that
# fit needs in the weights directory.
# fit reads the calibration set, which is not in the repository. The user makes it, as
# calibration_set.npy in the weights directory: need_set().
# It writes only to the work directory (it makes that directory and none above it), in "download"
# and "fit" to the weights directory, and through the build scripts to their output directories.
# Except for the builds' output and the programs' caches, no file that exists before a call is
# removed, replaced or renamed. Such a file is appended to only by download (a weight file X next to
# an X.download-may-append file) and by cp_prove (a count.bin file of the commitment). fit creates
# wsums.txt and fold.txt in the weights directory only if nothing has these names.
# These stay until you remove them: the work directory, with logs/, home/ and tmp/ (the programs'
# HOME and TMPDIR) and every fit.N, commitment.N and proof.N, also those of calls that failed or
# were interrupted; in the weights directory the weight files (232 GB; with --width 8 or 16, 426 or
# 801 GB), their .download-may-append files, wsums.txt, fold.txt and, after a fit that was killed,
# possibly wsums.txt.part-NUMBER or fold.txt.part-NUMBER; and in the repository the builds' output.
# The script itself sends no signal. If it alone is killed, what it started runs on: a build, a
# program or, in download, up to five background loops that call curl. Ctrl-C at a terminal goes to
# all of these. It stops prover/build.sh where env knows --default-signal (GNU coreutils 8.31 or
# later); with an older env it may not (prover/docs/build.md, "Build"). cp_commit kills only the
# process groups of its own cp_verify runs.
# The script rejects a work directory that another account may be able to change or replace and,
# in "needs" and "download" if bytes are missing, such a weights directory. Otherwise it only prints
# a note on that directory and the repository: there it has no defence against another account that
# can write as it runs. In every stage it rejects, in one look at its start, a path to the weights
# directory in which, as typed, a symbolic link belongs to another account. It does not look at
# further links that a link in that path leads through, or at who owns the directories on the way.
# shellcheck disable=SC2154,SC2317
if [ -z "${BASH_VERSION:-}" ]; then
  echo "run.sh: run this script with bash: bash examples/maverick/run.sh" >&2
  # "return" keeps a shell that sourced this file open.
  return 2 2>/dev/null || exit 2
fi
if [ "${BASH_SOURCE[0]:-}" != "$0" ]; then
  echo "run.sh: run this script as a file: bash examples/maverick/run.sh" >&2
  return 2 2>/dev/null || exit 2
fi
if [ "${MAVERICK_RUN_PROCESS:-}" != $$ ]; then # start again, once, with only these variables
  keep=(PATH="$PATH" MAVERICK_RUN_PROCESS=$$)
  # SUDO_UID: for root under sudo, open_above(), foreign_in() and the build scripts read it.
  for variable in CUDA_VISIBLE_DEVICES {https,all,no}_proxy {HTTPS,ALL,NO}_PROXY \
    CURL_CA_BUNDLE SSL_CERT_FILE SUDO_UID; do
    if [ -n "${!variable+set}" ]; then keep+=("$variable=${!variable}"); fi
  done
  exec env -i "${keep[@]}" /bin/bash -- "${BASH_SOURCE[0]}" "$@"
fi
unset MAVERICK_RUN_PROCESS
set -uo pipefail
# A caller can close standard output or standard error. The redirections below need both open.
{ true 3>&1; } 2>/dev/null || exec >/dev/null
true 3>&2 || exec 2>/dev/null # no "2>/dev/null" here: it would open the descriptor under test
umask 077

here=examples/maverick me=examples/maverick/run.sh
# The weight files: a third party's conversion of the model to the GGUF format, not Meta's files.
repository=https://huggingface.co/unsloth/Llama-4-Maverick-17B-128E-Instruct-GGUF
revision=41032e5471dd6ea5b349062d978626215d6c5bba
stem=Llama-4-Maverick-17B-128E-Instruct-UD-Q4_K_XL
lengths=(49807659968 49830263808 49832905856 49508645760 33155199296) # bytes of files 1 to 5
total=232134674688
folder=UD-Q4_K_XL # their directory in that repository
# A file's name is $stem-$pad<one entry of numbers>-of-<how many files, in five digits>.gguf.
pad=0000 numbers=(1 2 3 4 5)
many=five size='232 GB' exact='232.1 GB' # for messages; exact is $total to the nearest 0.1 GB
# The directory of weights.sha256, expected.txt and figures.txt.
data=$here
# The fit starts from this configuration's parameter file, policy.txt, and the commitment uses its
# layout.txt.
base=prover/examples/target_full_f2
# The build scripts, in the order in which they must run (the prover's build copies
# bin/cp_verify), and where they put the programs.
builds=(verifier/build.sh prover/build.sh weight_sums/build.sh)
programs=prover/build/hc verifier=bin/cp_verify
# The SHA-256 that the calibration set must have, SET_SHA256 in scales/make_calibration_set.py.
calibration_digest=a2ce04bd55b1b6fa2b9148744e087f5a2a484c1b3d525ea9fb2fc1ab8c6c2c02
# The versions under which the fit gave the digests in expected.txt, as check_python() prints them.
versions='Python 3.13, torch 2.9.1+cu128, numpy 2.3.5, gguf 0.19.0, GPU: True'
install='python3.13 -m venv VENV && VENV/bin/pip install --require-hashes -r requirements.txt'
install+=" (VENV: a name for a new directory here, not the work directory); then type"
install+=" PATH=\$PWD/VENV/bin:\$PATH before $me. If python3.13 is missing:"
install+=" $here/README.md, \"You need\""
# The figures that the script prints and checks for (times, host memory, disk space) are data, in
# figures.txt. They are read below, once the script is at the repository's top, into variables of
# these names. (SC2154 at the file's head: shellcheck does not see that printf -v assigns them.)
figures=(taken_with machine machine_memory sums_time calibrate_time commit_peak commit_memory
  commit_time worst_case_time prove_peak prove_memory prove_time verify_time build_time commit_disk
  prove_disk all_disk)
# -public-weights gives the seed file, from which the seed is derived.
commit_mode="-public-weights $here/public-seed.bin"
prove_mode=$commit_mode
prove_layout=lean # cp_prove's -memory: how it lays out GPU memory
# The line with which cp_commit prints the commitment ID, up to the ID, and the line that it prints
# when its worst-case check has passed.
id_line='commitment ID: '
fits_line='cp_commit: the worst-case check: every bounded site fits: at none of the sites that the'
fits_line+=' check bounds can a prompt drive the value out of its range'
# cp_verify's planning call: its exit status and last line on success, and the beginning of the
# line with which it reports parameters outside the standard form.
plan_status=4 plan_line='PLAN ONLY: no proof was checked'
plan_warning='plan: not the standard parameter form'
# cp_verify's exit status and last line for a correct proof that covers only the next token.
good_status=0 good_verdict=YES

run_log='' screen=open
# say <line> ...: writes the lines to this call's log, once there is one, then to standard output.
# The second write runs in a subshell: if the reader has closed the pipe, as after "| head", SIGPIPE
# kills only the subshell. So a long stage is not cut short, and the log has every line.
say() {
  { printf '%s\n' "$@" >>"${run_log:-/dev/null}"; } 2>/dev/null
  if [ -n "$screen" ]; then (printf '%s\n' "$@") 2>/dev/null || screen=''; fi
}
stop() { # <exit status> <message, in one or more arguments>: to the log and to standard error
  screen=''
  say "$me: ${*:2}"
  (echo "$me: ${*:2}" >&2)
  exit "$1"
}
usage() {
  cat >&2 <<'END'
usage: examples/maverick/run.sh <stage> <work directory> --weights <directory> [--width 8|16]
         [--limits <file>]
  needs      print what the stages need; check the machine for all but build's needs; write nothing
  build      build the verifier, the GPU programs and bin/weight_sums
  download   fetch the weight files at a pinned revision; a second call continues the files
  checksums  compare their SHA-256 checksums with weights.sha256
  fit        compute the quantization scales from the files and the calibration set, on the GPU,
             compare the digests of wsums.txt and KL.bin with expected.txt, and ask cp_verify, in a
             planning call, whether it accepts the fitted files
  commit     generate the commitment from the files, the fit's results and the published seed
  compare    compare the commitment ID and the digests of four files with expected.txt; after
             commit --limits, only the digests of wsums.txt and KL.bin
  prove      prove one request and check the proof against the commitment ID in expected.txt; after
             commit --limits, against your commitment ID
  all        all of these, in this order; it skips built programs, complete files, checksums and
             fit after a finished fit, and commit after a complete commitment. If expected.txt
             lacks the digest of wsums.txt or KL.bin, it stops after needs. If expected.txt
             lacks the commitment ID or the digest of key.bin or leaf/0.bin, it stops after needs,
             unless the work directory's commitment was made with --limits
  --weights  the directory of the weight files. Neither it nor the work directory may lie inside the
             other. cp_commit and cp_prove read every file *.gguf in it, follow no symbolic link
             there and take no file with a second hard link, so it must hold the weight files
             themselves and no other *.gguf file.
             No symbolic link on the way to it, as you type it, may belong to an account other
             than yours or root.
             Only download and fit write there. While bytes are missing, download refuses a
             directory that another account may be able to change or owns something in.
             For each file that it starts, download also creates
             an empty file with that file's name and .download-may-append appended, and a later call
             continues only files that have one. fit creates wsums.txt and fold.txt, two small text
             files that the programs read next to the weights, if nothing has these names (the
             file system must offer hard links; a call that is killed there can leave a file of
             such a name with .part-NUMBER appended). Next to
             each weight file the directory must contain a file with that file's name and
             .weights-are-public appended. You create it (the stage needs prints a command): it
             contains the weight file's line of weights.sha256 and declares the file with that
             checksum public. cp_commit counts it only if it belongs to the account that runs the
             script, to the weight file's owner or to root.
             fit reads the calibration set there, as calibration_set.npy. The repository does not
             have it, and the script does not create it: you make it, as the account that runs
             the script, with scales/make_calibration_set.py. The stage needs prints the command,
             unless you cannot write in the directory or another account may be able to change it.
             The commitment records where the files are, so both directories must
             stay in place after fit.
  --width    8 or 16: use the same publisher's 8-bit (Q8_0) or 16-bit (BF16) weight files. Without
             the option the script uses the 4-bit files. There are five 4-bit files (232 GB), nine
             8-bit files (426 GB) and eighteen 16-bit files (801 GB). weights.sha256 and
             expected.txt for the 4-bit files are in examples/maverick/, those for the others in
             examples/maverick/8-bit/ and examples/maverick/16-bit/. Give the option at every
             stage, before --limits. Use a separate weights directory and a separate work directory
             for each width: a work directory keeps the width of the call that created it, and
             the script stops if a later call has another.
             The wider files need more host memory and disk space: the stage needs prints how much.
  --limits   for commit only: generate the commitment under a clipped-gap acceptance policy instead
             of exact match, which accepts only the model's top choice for each token of a reply.
             A token's cost is its logit gap to the top choice plus 256. The costs, each clipped
             to C, must average at most theta, and at most cap tokens may cost more than C. The file
             contains the three lines "C N", "theta N" and "cap N" (N: a whole number), as
             integrations/vllm/integer_model/examples/fit_limits.py prints them: see
             integrations/vllm/integer_model/OPERATING.md, "Fit the limits of a clipped-gap policy".
             The commitment ID covers the limits, so this ID is your own, and expected.txt has no
             value for it. Call the stages one by one, from build to prove, with --limits only at
             commit, in a work directory without a commitment: "all" would generate the published
             commitment there.
A work directory holds one complete commitment. For another one, choose a new work directory.
Except for needs, all output also goes to logs/run.N.log (N = 1, 2, ...) in the work directory.
END
  if [ $# -gt 0 ]; then echo "$me: $*" >&2; fi # what was wrong with --width, last on the screen
  exit 2
}
limits='' width=4 wide='' unplayed=''
if [ $# -ge 6 ] && [ "$5" = --width ]; then
  width=$6 wide=" --width $6"
  set -- "${@:1:4}" "${@:7}"
  case $width in
  8 | 16) ;;
  *) usage "--width takes 8 or 16. For the 4-bit files, leave the option out." ;;
  esac
fi
case " $* " in
*' --width '*) usage "--width 8 or --width 16 comes once, right after the weights directory." ;;
esac
if [ $# -eq 6 ] && [ "$5" = --limits ]; then limits=$6; elif [ $# -ne 4 ]; then usage; fi
if [ "$3" != --weights ]; then usage; fi
stage=$1 work=$2 weights=$4
case $stage in
needs | build | download | checksums | fit | commit | compare | prove | all) ;;
*) usage ;;
esac
# The files of 8 and of 16 bits. The repository and the revision are those above.
case $width in
8)
  stem=Llama-4-Maverick-17B-128E-Instruct-Q8_0 folder=Q8_0 data=$here/8-bit
  lengths=(47350553088 45634028224 48293566688 45634028224 48282425536 48276814016 45634028224
    48276855104 48434794944)
  total=425817094048 many=nine size='426 GB' exact='425.8 GB'
  numbers=(1 2 3 4 5 6 7 8 9)
  prove_layout=lean,tensor-rooms=6
  ;;
16)
  stem=Llama-4-Maverick-17B-128E-Instruct-BF16 folder=BF16 data=$here/16-bit
  lengths=(46166870240 42949673376 42949673376 42949673376 47943931840 42949673376 42949673376
    42949673376 47922960288 42949673376 42949673376 47912433568 42949673376 42949673376
    42949673376 47912474624 42949673376 48214491296)
  total=801469242368 many=eighteen size='801 GB' exact='801.5 GB'
  pad=000 numbers=(01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18)
  prove_layout=lean,tensor-rooms=0,experts-by-layer,experts-rooms=1
  ;;
esac
# With --width, what the message on a value that differs from expected.txt says first, when that
# file has no commitment ID (compare_values()): an ID there comes from one run (README.md, "How far this has been run").
if [ -n "$wide" ]; then
  unplayed=' This script as it is here has not been run on these files: the expected values come from another'
  unplayed+=" script of this project's. So the fault may lie in this script rather than in your"
  unplayed+=" files or machine ($here/README.md, \"The 8-bit and the 16-bit files\")."
  unplayed+=' The usual message follows.'
fi
printf -v last %05d "${#numbers[@]}"
# The numbers as a brace list, for the command lines that name all files: {1,2,3,4,5}.
braces=$(IFS=,; printf '{%s}' "${numbers[*]}")
# A quoted "~", "*" or "$HOME" would create a directory of that name, and an unquoted "rm -r" of
# that name would remove something else. So the script creates a directory only if plain() accepts
# its path as typed.
plain() { [[ /$1 != */-* && $1 != *[!A-Za-z0-9/._-]* ]]; }
for path in "$work" "$weights"; do
  if [[ -z $path || $path == -* || /$path/ == */../* || $path == *[[:cntrl:]]* ]]; then
    stop 2 "a directory's path may not be empty, begin with \"-\", or contain \"..\" as a part" \
      "or a control character: $(printf %q "$path"). Nothing was created."
  fi
  if [ ! -e "$path" ] && [ ! -L "$path" ] && ! plain "$path"; then
    tilde=''
    if [[ $path == *'~'* ]]; then tilde=' A shell does not expand "~" inside quotes.'; fi
    stop 2 "the directory $(printf %q "$path") does not exist, and the script creates a directory" \
      "only if its path as typed has only the letters A-Z and a-z, digits and the characters" \
      "/ . _ and -, and no part of it begins with \"-\". Nothing was created.$tilde To create a" \
      "weights directory below one whose name has other characters, change to that directory and" \
      "give a relative path; a weights directory that exists already can have other characters." \
      "The work directory's full path must have this form."
  fi
done
if [ $# -eq 6 ]; then
  if [ "$stage" != commit ]; then
    stop 2 "only the stage commit takes --limits. Call the stages one by one (build, download," \
      "checksums, fit, commit, compare, prove), with --limits only at commit. compare and prove" \
      "read the limits from the commitment's directory. \"all\" takes no limits: it generates the" \
      "published commitment. Nothing was created."
  fi
  if [[ -z $limits || $limits == -* || /$limits/ == */../* || $limits == *[[:cntrl:]]* ]]; then
    stop 2 "the path given with --limits may not be empty, begin with \"-\", or contain \"..\" as" \
      "a part or a control character: $(printf %q "$limits"). Nothing was created."
  fi
  if [[ $limits != /* ]]; then limits=$PWD/$limits; fi
fi
if [[ $work != /* ]]; then work=$PWD/$work; fi
if [[ $weights != /* ]]; then weights=$PWD/$weights; fi
while [[ $work == ?*/ || $work == ?*/. ]]; do work=${work%.} work=${work%/}; done
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
# The stages write their logs and results to paths below the work directory. So the script rejects
# it if another account may be able to change a directory above it, creates no directory above it,
# and from here on reaches it through $real, its parent's path without links, set by open_above().
if open_above "${work%/*}/"; then
  if [ "$why" = 'it does not exist' ]; then
    stop 2 "$place, above the work directory, does not exist: make it first; this script makes" \
      "only the work directory. Nothing was created."
  elif [ "$why" = 'it is not a directory' ]; then
    stop 2 "$place, above the work directory, is not a directory. Nothing was created."
  elif [ "$why" = 'it could not be examined' ]; then
    stop 2 "$place, at or above the work directory, could not be examined: this script needs id," \
      "GNU stat and readlink, and permission to enter each directory above the work directory." \
      "Nothing was created."
  fi
  stop 2 "the work directory $(printf %q "$work") lies below $place, which another account may" \
    "be able to change ($why): it could put another directory in the work directory's place while" \
    "the stages run and choose what they run as you. Name a new work directory below directories" \
    "that only you or root can change (such as one in your home directory, or in /var/tmp, whose" \
    "sticky bit keeps others from replacing it)$cure; nothing was run"
fi
work=${real%/}/${work##*/}
# cp_prove accepts its HOME and TMPDIR, which this script puts below the work directory, only in
# this form. Without this test the fault would show only after cp_commit's long run.
if ! plain "$work" || [ "${#work}" -gt 390 ]; then
  stop 2 "the work directory's full path, $(printf %q "$work"), must have only the letters A-Z" \
    "and a-z, digits and the characters / . _ and -, no part that begins with \"-\", and at most" \
    "390 characters: cp_prove accepts directories below it only in this form. Nothing was created."
fi
while [[ $weights == ?*/ || $weights == ?*/. ]]; do weights=${weights%.} weights=${weights%/}; done
if [ ! -d "$weights" ] && { [ -e "$weights" ] || [ -L "$weights" ]; }; then
  stop 2 "$weights exists and is not a directory: choose another weights directory"
fi
resolved=$(readlink -m -- "$weights")
# Whoever owns a symbolic link on the way to the weights directory chooses which files the stages
# read and where fit creates wsums.txt and fold.txt. So each link in the path as typed, from its
# last part up, must belong to an account that open_above() trusts: this one, root or, for root
# under sudo, the account that ran sudo. A part that does not exist yet is no link. Of "a//b" the
# loop first sees "a/", which is no link, then "a". Not examined: see this file's head.
at=$weights caller=$(id -u)
while [[ $at == /?* ]]; do
  if [ -L "$at" ]; then
    owner=$(LC_ALL=C stat -c %u -- "$at" 2>/dev/null) || owner=unknown
    if [ "$owner" != 0 ] && [ "$owner" != "$caller" ] &&
      { [ "$caller" != 0 ] || [ "$owner" != "${SUDO_UID:-}" ]; }; then
      stop 2 "$(printf %q "$at"), on the way to the weights directory, is a symbolic link that" \
        "belongs to another account (user ID $owner). That account can point it elsewhere at any" \
        "time, and so choose which files the stages read and where fit creates two files. Nothing" \
        "was created. If the weights directory that this path reaches now is the one that you" \
        "mean, give its path without links, with a new work directory if this one has a fit:" \
        "--weights $(printf %q "$resolved")"
    fi
  fi
  at=${at%/*}
done
if [[ $weights/ == "$work"/* || $resolved/ == "$work"/* ]]; then
  stop 2 "choose a weights directory outside $work, which must be new at the first call"
fi
# Only download and fit write below the weights directory, but every stage except "needs" writes to
# the work directory. So the work directory must lie outside the weights directory. "%/" is for the
# weights directory "/": its pattern is then "/*", which every work directory matches, not "//*".
if [[ $work/ == "${weights%/}"/* || $work/ == "${resolved%/}"/* ]]; then
  stop 2 "the work directory $work lies inside the weights directory $weights, once symbolic" \
    "links are followed: choose a work directory outside it, or another weights directory." \
    "Nothing was created."
fi
{ cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."; } 2>/dev/null ||
  stop 2 "cannot change to the repository"
absent=''
for file in "$data"/{weights.sha256,expected.txt,figures.txt} \
  "$here/public-seed.bin" \
  "$base"/{policy.txt,layout.txt} prover/tools/make_request.py; do
  if [ ! -f "$file" ] || [ ! -r "$file" ]; then absent+=" $file"; fi
done
if [ -n "$absent" ]; then
  stop 1 "stopped: $PWD does not contain, as regular files that you can read:$absent. No stage" \
    "can run without these files."
fi
# figures.txt: "name value" on each line; each name in $figures once, and no other; empty lines and
# lines that begin with "#" are skipped. The values go into messages and arithmetic, never to eval,
# and the tests below limit their characters.
form='^[0-9A-Za-z]([0-9A-Za-z,. -]{0,78}[0-9A-Za-z])?$'
as_published="the file's first lines give the form, and a new copy of the repository has the file"
as_published+=' as published'
for name in "${figures[@]}"; do unset -v "$name"; done # so that "set" below means: by this file
while IFS=' ' read -r name value || [ -n "$name" ]; do
  case $name in '' | '#'*) continue ;; esac
  if [[ " ${figures[*]} " != *" $name "* || -n ${!name+set} || ! $value =~ $form ]]; then
    stop 1 "stopped: $data/figures.txt has a line that this script cannot use, which begins with" \
      "$(printf %q "$name"): $as_published"
  fi
  printf -v "$name" %s "$value"
done <"$data/figures.txt"
for name in "${figures[@]}"; do
  [ -n "${!name+set}" ] ||
    stop 1 "stopped: $data/figures.txt has no line that begins with $name: $as_published"
done
for name in commit_memory prove_memory commit_disk prove_disk all_disk; do
  [[ ${!name} =~ ^[1-9][0-9]{0,4}$ ]] ||
    stop 1 "stopped: in $data/figures.txt the value of $name must be digits alone, without a" \
      "unit: a whole number of GB from 1 to 99999, without a leading zero"
done
# The script must ask for more host memory than was in use in the measured run.
for name in commit prove; do
  peak=${name}_peak floor=${name}_memory
  [[ ${!peak} =~ ^([1-9][0-9]{0,4})\ GB$ ]] ||
    stop 1 "stopped: in $data/figures.txt the value of $peak must be a whole number from 1 to" \
      "99999, without a leading zero, then one blank and GB. A new copy of the repository has" \
      "the file as published"
  [ "${!floor}" -gt "${BASH_REMATCH[1]}" ] ||
    stop 1 "stopped: in $data/figures.txt the value of $floor, ${!floor}, must be greater than" \
      "the number in $peak, ${!peak}: the script must ask for more host memory than was in use" \
      "in the measured run of cp_$name. A new copy of the repository has the file as published"
done
# The caller's PATH comes first, so /usr/local/cuda/bin supplies nvcc only if PATH has none.
# ldconfig is often not on PATH.
export PATH=$PATH:/usr/local/cuda/bin:/usr/sbin:/sbin
export CUDA_DEVICE_ORDER=PCI_BUS_ID # a GPU's number then means the same to CUDA as to nvidia-smi
# foreign_in() takes "find printed nothing" as "nothing found". So find must start and must know the
# tests and actions that foreign_in() uses.
find / -maxdepth 0 -mindepth 0 -uid +0 -uid -1 -print -quit >/dev/null 2>&1 ||
  stop 1 "stopped: this script needs GNU find: no find is on PATH, or the first one there does" \
    "not start or does not know -mindepth, -uid or -quit. GNU find comes with the package" \
    "findutils. Nothing was created."
# W, G (below also L, F, C, P): paths as printed in commands, which run, and can be pasted, at the
# repository's top.
printf -v W %q "$work"
printf -v G %q "$weights"
arguments="$W --weights $G$wide" # for messages that give the next call
# The calibration set's path, and S, that path as printed: need_set().
calibration_set=$weights/calibration_set.npy S=$G/calibration_set.npy
# The file given with --limits is read once, here. Commands contain only its three numbers, and
# only after this test of their form, because check() and run() pass their arguments to eval.
limit_C='' limit_theta='' limit_cap='' L=''
if [ -n "$limits" ]; then
  printf -v L %q "$limits"
  if [ ! -f "$limits" ] || [ ! -r "$limits" ]; then
    stop 2 "$L, given with --limits, does not exist or is not a regular file that you can read." \
      "Nothing was created."
  fi
  digits='(0|[1-9][0-9]{0,9})' text=''
  form="^C $digits"$'\n'"theta $digits"$'\n'"cap $digits"$'\n$'
  # "read" succeeds only at a NUL byte or after 64 characters, and a valid file has at most 45.
  # Otherwise $text is the whole file, with its last newline.
  if { IFS= read -r -d '' -n 64 text <"$limits"; } 2>/dev/null || [[ ! $text =~ $form ]]; then
    stop 2 "$L, given with --limits, must contain exactly the three lines \"C N\", \"theta N\"" \
      "and \"cap N\", in this order. N is a whole number of at most ten digits without a leading" \
      "zero, after one space. Each line ends with a newline, without a carriage return." \
      "integrations/vllm/integer_model/examples/fit_limits.py prints this form." \
      "Nothing was created."
  fi
  limit_C=${BASH_REMATCH[1]} limit_theta=${BASH_REMATCH[2]} limit_cap=${BASH_REMATCH[3]}
  if [ "$limit_C $limit_theta $limit_cap" = '256 256 0' ]; then
    stop 2 "$L contains the limits of exact match (C 256, theta 256, cap 0). They give the" \
      "published commitment, which commit generates without --limits: $me commit $arguments"
  elif [ "$limit_theta" -lt 256 ] || [ "$limit_theta" -gt "$limit_C" ]; then
    stop 2 "$L has a theta below 256 or above C. A token costs at least 256, so no reply meets a" \
      "theta below 256, and the programs reject a theta above C. Nothing was created."
  elif [ "$limit_C" -gt 4294967295 ] || [ "$limit_cap" -gt 1024 ]; then
    stop 2 "$L has a C above 4294967295, the largest that a parameter file can hold, or a cap" \
      "above 1024, the number of tokens in a request. Nothing was created."
  elif [ "$limit_theta" = "$limit_C" ] && [ "$limit_cap" != 0 ]; then
    stop 2 "$L has theta equal to C, and cap above 0. Under such limits no token is ruled out in" \
      "a request that checks at most \"cap\" tokens. Use a theta below C, or cap 0." \
      "Nothing was created."
  fi
fi
files=()
for number in "${numbers[@]}"; do files+=("$stem-$pad$number-of-$last.gguf"); done

step=0 out='' status=0 libraries_checked='' fit='' commitment=''
announce() {
  step=$((step + 1))
  say '' "[step $step] \$ $*"
}
check() { # <a short command line, in one or more arguments>: output to say and into $out
  announce "$@"
  out=$(eval "$*" 2>&1)
  status=$?
  if [ -n "$out" ]; then say "$out"; fi
}
run() { # <new log file> <a long command line, in one or more arguments>: output to it and to say
  local line
  announce "${@:2}"
  eval "${*:2}" 2>&1 | while IFS= read -r line || [ -n "$line" ]; do
    printf '%s\n' "$line" 2>/dev/null
    say "$line" >&3
  done 3>&1 >"$1"
  status=${PIPESTATUS[0]}
}
result() { # <expected> <got> [<why the script stops here, in one or more arguments>]
  say "expected: $1" "got:      $2"
  if [ $# -gt 2 ]; then stop 1 "stopped in stage $stage at step $step: ${*:3}"; fi
}
status_0() { # <why the script stops here if the last command's exit status is not 0>
  if [ "$status" -ne 0 ]; then result 'exit status 0' "exit status $status" "$@"; fi
  result 'exit status 0' 'exit status 0'
}
new_log() { # a free name in the work directory's logs/: <name>.1.log, <name>.2.log, ..
  local log=$work/logs/$1 number=1
  while [ -e "$log.$number.log" ] || [ -L "$log.$number.log" ]; do number=$((number + 1)); done
  printf '%s' "$log.$number.log"
}
value_of() { awk -v name="$2" '$2 == name { print $1 }' "$1"; } # <file of "value  name"> <name>
first_of() { # a list's first word
  local words
  read -ra words < <(printf '%s\n' "$1")
  printf '%s' "${words[0]}"
}
expected_of() { value_of "$data/expected.txt" "$1"; }
digest_of() { # a regular file's SHA-256, or "absent"
  local line=absent
  if [ -f "$1" ]; then { line=$(sha256sum <"$1"); } 2>/dev/null || line=absent; fi
  printf '%s' "${line%% *}"
}
same() { # <expected> <got> <why the script stops here if they differ, in one or more arguments>
  if [ "$1" != "$2" ]; then result "$@"; fi
  result "$1" "$2"
}
# <entries of expected.txt, in one argument> <what a difference means, in one or more arguments>:
# compares the lines "value  name" in $out with those in expected.txt
compare_values() {
  local names name expected got different=''
  read -ra names < <(printf '%s\n' "$1")
  for name in "${names[@]}"; do
    expected=$(expected_of "$name") got=$(value_of /dev/stdin "$name" < <(printf '%s\n' "$out"))
    say '' "$name"
    if [[ ! $expected =~ ^[0-9a-f]{64}$ ]]; then
      result '64 hex digits' "${expected:-nothing}" "$data/expected.txt has no value for $name;" \
        "your run's value is $got"
    fi
    result "$expected" "${got:-nothing}"
    if [ "$expected" != "$got" ]; then different+="${different:+, }$name"; fi
  done
  if [ -n "$different" ]; then
    if [[ $(expected_of commitment-id) =~ ^[0-9a-f]{64}$ ]]; then unplayed=''; fi
    stop 1 "stopped in stage $stage: not equal: $different.$unplayed ${*:2}"
  fi
}
existing() { # the nearest existing directory at or above a path
  local path=$1
  while [ ! -d "$path" ]; do path=$(dirname -- "$path"); done
  printf '%s' "$path"
}
length_of() { stat -L -c %s -- "$weights/$1" 2>/dev/null || echo 0; } # a weight file's bytes
missing_bytes() { # how many of the download's bytes are not in the weights directory
  local missing=$total name
  for name in "${files[@]}"; do missing=$((missing - $(length_of "$name"))); done
  printf '%s' "$missing"
}
cpus() { # the CPUs that this process may use: nproc, or fewer under a container's limit (cgroup v2)
  local count quota='' period=''
  count=$(nproc)
  { read -r quota period </sys/fs/cgroup/cpu.max; } 2>/dev/null
  if [[ $quota =~ ^[1-9][0-9]*$ && $period =~ ^[1-9][0-9]*$ ]] &&
    [ $((quota / period)) -lt "$count" ]; then
    count=$((quota < period ? 1 : quota / period))
  fi
  printf '%s' "$count"
}
gigabytes() { printf '%d.%d GB' $(($1 / 1000000000)) $(($1 / 100000000 % 10)); } # from bytes
# <GB, of 10^9 bytes>, rounded up to kB of 1,024 bytes, the unit of "df -k" and /proc/meminfo
kb_of() { printf '%d' $((($1 * 1000000000 + 1023) / 1024)); }

check_gpu() {
  local expected gpu=${CUDA_VISIBLE_DEVICES-0} name capability memory used got
  expected='a GPU of compute capability 9.0, as an H100 has, with at least 80000 MiB,'
  expected+=' under 2048 MiB in use'
  gpu=${gpu%%,*}
  if [[ ! $gpu =~ ^[A-Za-z0-9-]+$ ]]; then
    stop 1 "stopped: CUDA_VISIBLE_DEVICES is set but does not begin with a GPU's number or UUID" \
      "(under a job scheduler: the job has no GPU)"
  fi
  check "nvidia-smi --id=$gpu --query-gpu=name,compute_cap,memory.total,memory.used" \
    "--format=csv,noheader,nounits"
  if [ "$status" -ne 0 ]; then
    result "$expected" "exit status $status" "nvidia-smi did not report GPU $gpu (a status of 127" \
      "means that nvidia-smi was not found); it comes with the NVIDIA driver, and a container has" \
      "it only if it was started with the GPU"
  fi
  IFS=, read -r name capability memory used < <(printf '%s\n' "$out")
  capability=${capability//[[:space:]]/} memory=${memory//[[:space:]]/} used=${used//[[:space:]]/}
  got="$name, compute capability $capability, $memory MiB, $used MiB in use"
  if [ "$capability" != 9.0 ]; then
    result "$expected" "$got" "the GPU programs are built for compute capability 9.0 only"
  elif [[ ! $memory =~ ^[0-9]+$ || ! $used =~ ^[0-9]+$ ]]; then
    result "$expected" "$got" "nvidia-smi gives no numbers for the memory of GPU $gpu"
  elif [ "$memory" -lt 80000 ]; then
    result "$expected" "$got" "this model fills nearly all of 80 GB: use a GPU with 80 GB"
  elif [ "$used" -ge 2048 ]; then
    result "$expected" "$got" "GPU $gpu is in use; to use another GPU, type" \
      "CUDA_VISIBLE_DEVICES=NUMBER (NUMBER: its number in nvidia-smi's list) before $me"
  fi
  result "$expected" "$got"
  if [ -n "$libraries_checked" ]; then return; fi # only the GPU's load changes between two stages
  # cp_commit and cp_prove restart themselves without LD_LIBRARY_PATH. So the loader must find the
  # driver's library and the toolkit's libraries without it.
  expected="a line for libcuda.so.1"
  check "ldconfig -p | grep -E 'libcuda|libcudart|libcublas'"
  if ! grep -qF 'libcuda.so.1 ' < <(printf '%s\n' "$out"); then
    result "$expected" 'no line for libcuda.so.1' "the system's loader must find libcuda.so.1" \
      "without LD_LIBRARY_PATH; it comes with the driver (in a container: with the GPU)"
  fi
  result "$expected" 'libcuda.so.1 is listed'
  check 'grep -m1 -ow sha_ni /proc/cpuinfo'
  same sha_ni "${out:-nothing}" "this CPU, or this virtual machine, does not offer the SHA" \
    "extensions, which the programs need: use another machine"
  if ! built; then
    say 'Whether the programs find the libraries of CUDA 13 is checked once they are built.'
    return
  fi
  libraries_checked=yes
  expected='a path for each CUDA library that cp_prove needs'
  check "ldd $programs/cp_prove | grep -E 'libcuda|libcudart|libcublas'"
  if [ -z "$out" ] || grep -q 'not found' < <(printf '%s\n' "$out"); then
    result "$expected" 'not found, or no line' "without LD_LIBRARY_PATH the loader does not find" \
      "a library that the programs need, so they cannot start. prover/build.sh gives the programs" \
      "a run path to the toolkit's libraries, or prints a warning (\"no rpath\") that says why:" \
      "see prover/docs/build.md, \"What you need\", which also says what to do then."
  fi
  result "$expected" 'a path for each'
}
check_python() {
  check "python3 -Bs -c 'import sys, importlib.metadata as m, numpy, torch;" \
    'print("Python %d.%d, torch %s, numpy %s, gguf %s, GPU: %s" % (*sys.version_info[:2],' \
    "torch.__version__, numpy.__version__, m.version(\"gguf\"), torch.cuda.is_available()))'"
  same "$versions" "$(tail -n 1 < <(printf '%s\n' "$out"))" "the digests in $data/expected.txt" \
    "were computed under these versions, and torch must see the GPU. To install them: $install"
}
# Sets $available, in kB, and $got, which describes it: the smaller of MemAvailable and the
# container's memory limit minus the memory in use. File cache does not count as in use. Under
# cgroup v1, or if the memory in use cannot be read, the limit itself is taken. Only a limit in
# /sys/fs/cgroup is seen: not one on a cgroup above the container's, nor one on a job without a
# cgroup namespace (Slurm). Only values that match $whole are used: at most 15 digits, which cannot
# overflow the arithmetic; no leading zero, which would mean octal; no range, so no locale matters.
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
check_memory() { # <GB> <program> <the most host memory that was in use while it ran>
  local expected available floor got
  floor=$(kb_of "$1")
  expected="at least $floor kB ($1 GB) available, on the machine and under the container's memory"
  expected+=' limit, if there is one'
  available_memory
  if [ "$available" -lt "$floor" ]; then
    result "$expected" "$got" "in a measured run up to $3 of host memory was in use while $2" \
      "ran, not counting file cache (with $taken_with, on a machine with $machine_memory)." \
      "Here at most" \
      "$(gigabytes $((available * 1024))) is available: stop other programs of your own, raise" \
      "the container's memory limit, or use a larger machine"
  fi
  result "$expected" "$got"
}
check_disk() { # <directory, need not exist> <kB> [local]: "local" rejects network file systems
  local path free expected
  expected="at least $2 kB ($(gigabytes $(($2 * 1024)))) available"
  printf -v path %q "$(existing "$1")"
  check "df -Pk $path"
  free=$(awk 'NR == 2 { print $4 }' < <(printf '%s\n' "$out"))
  if [[ ! $free =~ ^[0-9]+$ ]] || [ "$free" -lt "$2" ]; then
    result "$expected" "${free:-no number} kB" "too little free disk space. The" \
      "second argument sets the work directory, and --weights sets the directory for the $size" \
      "of weight files."
  fi
  result "$expected" "$free kB"
  [ $# -eq 3 ] || return 0
  check "stat -f -c %T $path"
  case $out in
  nfs* | cifs | smb* | 9p | v9fs | lustre | ceph | gpfs)
    result 'a local file system' "$out" "cp_commit needs a rename that network file systems do" \
      "not offer: choose a work directory on a local disk. The directory given with --weights" \
      "can be on a network file system."
    ;;
  esac
  result 'a local file system' "$out"
}
check_disks() { # the rest of the download, and the room for commit and prove; one sum on one disk
  local fetch=$((($(missing_bytes) + 1023) / 1024)) disks
  if [ "$fetch" -lt 0 ]; then fetch=0; fi # a file is longer than published: download reports it
  disks=$(stat -L -c %d -- "$(existing "$weights")" "$(existing "$work")" | sort -u | wc -l)
  if [ "$disks" -eq 1 ]; then
    check_disk "$work" $((fetch + $(kb_of $((commit_disk + prove_disk))))) local
  else
    if [ "$fetch" -gt 0 ]; then check_disk "$weights" "$fetch"; fi
    check_disk "$work" "$(kb_of $((commit_disk + prove_disk)))" local
  fi
}
# <directory>: prints the first thing below it whose owner open_above() would not trust: not this
# account, root or, for root under sudo, the account that ran sudo. It uses no "! -uid": that is
# also true of a file that vanishes during find's walk.
foreign_in() {
  local upper
  upper=$(id -u) || upper=0
  if [ "$upper" = 0 ] && [[ ${SUDO_UID:-} =~ ^[1-9][0-9]*$ ]]; then upper=$SUDO_UID; fi
  find "$1/" -mindepth 1 \( -uid +0 -uid "-$upper" -o -uid "+$upper" \) -print -quit 2>/dev/null
}
# Accepts an existing work directory only if this user's earlier call created it, with the same
# --width as this call or both without one, and it is still closed to others. Returns 1 if nothing
# has the name.
check_work() {
  local recorded
  if [ ! -e "$work" ] && [ ! -L "$work" ]; then return 1; fi
  # If the name is a link, $real differs from $work. The sticky bit excuses only directories above.
  # What another account put into the directory while it was open would still be there.
  if open_above "$work/" || [ "$real" != "$work" ] || [ ! -O "$work" ] ||
    ((8#$(LC_ALL=C stat -c %a -- "$work") & 8#22)) ||
    [ ! -f "$work/created-by-maverick-run.txt" ] ||
    [ -n "$(foreign_in "$work")" ]; then
    stop 2 "$work exists and was not created by your earlier call of this script, or is a" \
      "symbolic link, or its group or others can write in it, or it holds something that belongs" \
      "to another account: choose a new directory, or in place of a link the directory that it" \
      "leads to"
  fi
  if fitted; then
    recorded=$(cat -- "$work/$fit/weights-directory.txt" 2>/dev/null)
    if [ "$recorded" != "$weights" ]; then
      stop 2 "$work/$fit was computed from the files in $recorded: use --weights" \
        "$(printf %q "$recorded"), or choose a new work directory for other files"
    fi
  fi
  # The call that created the directory wrote its --width into the first line of this file.
  recorded=''
  { IFS= read -r recorded <"$work/created-by-maverick-run.txt"; } 2>/dev/null
  case $recorded in
  *' --width 8.') recorded='with --width 8' ;;
  *' --width 16.') recorded='with --width 16' ;;
  'The work directory of '*.) recorded='without --width' ;;
  *)
    stop 2 "$work/created-by-maverick-run.txt cannot be read, or its first line is not this" \
      "script's, so the script cannot tell which weight files $work is for: choose a new work" \
      "directory"
    ;;
  esac
  if [ "$recorded" != "${wide:+with}${wide:-without --width}" ]; then
    stop 2 "$work was created by a call $recorded. A work directory serves one width of weight" \
      "files: call every stage there $recorded, or choose a new work directory"
  fi
}
built() {
  local program
  for program in "$programs"/{cp_commit,cp_prove,cp_verify} "$verifier" bin/weight_sums; do
    [ -x "$program" ] || return 1
  done
}
# <name>: one more than the highest N, from 1 to 9999, of a <work directory>/<name>.N; 1 if there is
# none. A <name>.10000 is made and then not counted: later stages go by the newest <name>.N up to
# 9999, and the next call that would make a <name>.N stops at mkdir.
next_number() {
  # fit, commit and prove each write to a new fit.N, commitment.N or proof.N in the work directory.
  # fitted() and committed() look downward from this number. So a directory that its owner has
  # removed must not hide those with higher numbers, and a new directory must get the highest.
  local number=0 path end
  for path in "$work/$1".*; do # a pattern that matches nothing stays as typed, and ends in "*"
    end=${path#"$work/$1."}
    if [[ $end =~ ^[1-9][0-9]{0,3}$ ]] && [ "$end" -gt "$number" ]; then number=$end; fi
  done
  printf '%s' $((number + 1))
}
# Sets $fit to the newest finished fit.N: its KL.bin has the expected digest, and its files passed
# plan().
fitted() {
  local number
  for ((number = $(next_number fit) - 1; number > 0; number--)); do
    fit=fit.$number
    [ -f "$work/$fit/plan-passed.txt" ] || continue
    [ "$(digest_of "$work/$fit/config/KL.bin")" = "$(expected_of config/KL.bin)" ] || continue
    return 0
  done
  fit=''
  return 1
}
# Sets $commitment to the newest complete commitment.N. cp_commit works in ivk.making and gives the
# directory the name ivk by its last rename.
committed() {
  local number
  for ((number = $(next_number commitment) - 1; number > 0; number--)); do
    commitment=commitment.$number
    if [ -f "$work/$commitment/ivk/leaf/0.bin" ]; then return 0; fi
  done
  commitment=''
  return 1
}
own_limits() { [ -d "$work/$commitment/config" ]; } # was $commitment generated with --limits?
printed_id() { # the commitment ID that cp_commit printed when it generated $commitment
  sed -n "s/^$id_line//p" "$work/$commitment/cp_commit.log" 2>/dev/null
}
need_built() { built || stop 1 "the programs are not built; run first: $me build $arguments"; }
need_fit() {
  fitted || stop 1 "$work contains no finished fit; run first: $me fit $arguments"
}
# No other account can write in the weights directory or swap one above it, or owns anything in it?
# If it exists, need_marker() prints its command only then: the command goes by a path, which
# another account could lead elsewhere.
weights_closed() {
  local mode
  [ -d "$weights" ] && ! open_above "$weights/" &&
    mode=$(LC_ALL=C stat -L -c %a -- "$weights") && ! ((8#$mode & 8#22)) &&
    [ -z "$(foreign_in "$weights")" ]
}
# Creates a declaration as the first command does in docs/operators-duties.md, "Keys, seeds and
# machines", "Declaring a model file's weights public": in one step, only if nothing has the name,
# through no link, never writable by its group or others.
declare_public='import os, sys; sys.tracebacklimit = 0; os.write(os.open(sys.argv[1], os.O_WRONLY |'
declare_public+=' os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644), (sys.argv[2] + "\n").encode())'
# The same step for an empty file, in the command that stage_download() prints.
create_empty='import os, sys; sys.tracebacklimit = 0; os.close(os.open(sys.argv[1], os.O_WRONLY |'
create_empty+=' os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644))'
need_marker() { # the user's declaration: next to each weight file a file with its published SHA-256
  local path number name listed declared list expected elsewhere owner of_file ways
  local missing='' empty='' other='' loose='' foreign='' whose='' beside='' make_it='' shared=''
  local advice page as_root='' loose_paths='' empty_paths=''
  expected="$many regular files that start with the checksums in weights.sha256"
  elsewhere='Choose another weights directory, one that only you can write to: copy the'
  elsewhere+=" $many weight"
  elsewhere+=' files there, or let the stage download fetch them. If the work directory has a fit,'
  elsewhere+=' choose a new work directory too.'
  printf -v list %q "$PWD/$data/weights.sha256"
  # No -p for the last component (see stage_download()): if mkdir fails, the printed command stops.
  if [ ! -d "$weights" ]; then
    make_it="mkdir -p $(printf %q "$(dirname -- "$weights")") && mkdir $G && "
  fi
  # -d: if one of these names is a directory, ls prints the name, not what another user put in it
  check "ls -d $G/$stem-$pad$braces-of-$last.gguf.weights-are-public"
  for number in "${numbers[@]}"; do # 10#: read 08 and 09 as decimal, not octal
    name=${files[10#$number - 1]} path=$weights/${files[10#$number - 1]}.weights-are-public
    listed=$(value_of "$data/weights.sha256" "$name")
    if [[ ! $listed =~ ^[0-9a-f]{64}$ ]]; then
      result "$expected" "no checksum for $name" \
        "$data/weights.sha256 has no checksum for $name: take a new copy of the repository"
    fi
    if [ -f "$path" ] && [ ! -L "$path" ]; then # cp_commit follows no link here
      # The file must start with the checksum, followed by a space, a newline or the file's end,
      # and have one name: cp_commit rejects a file with a second hard link.
      declared=''
      { IFS= read -r -n 65 declared <"$path"; } 2>/dev/null
      if [ "${declared% }" = "$listed" ] && [ "$(stat -c %h -- "$path")" = 1 ]; then
        if ((8#$(stat -c %a -- "$path") & 8#22)); then
          loose+=" $number" loose_paths+=$'\n'$(printf %q "$path")
        fi
        # cp_commit's rule, in prover/hash_commitment/public_weights_markers.h: a declaration counts
        # only if it belongs to the account that runs cp_commit, to the weight file's owner or to
        # root; if the weight file cannot be examined, as before download, to that account or root.
        owner=$(stat -c %u -- "$path" 2>/dev/null) || owner=unknown
        of_file=$(stat -L -c %u -- "$weights/$name" 2>/dev/null) || of_file=''
        if [ "$owner" != "$caller" ] && [ "$owner" != "$of_file" ] && [ "$owner" != 0 ]; then
          if [ -z "$foreign" ]; then whose=$owner beside=${of_file:+user ID $of_file}; fi
          foreign+=" $number"
        fi
        continue
      fi
      # Listed as empty only if this user owns it; else with the files of other content.
      if [[ ! -s $path && -O $path && -r $path && -w $path && $(stat -c %h -- "$path") == 1 ]]; then
        empty+=" $number" empty_paths+=$'\n'$(printf %q "$path")
      else
        other+=" $number"
      fi
      continue
    fi
    if [ -e "$path" ] || [ -L "$path" ]; then
      result "$expected" "a symbolic link, or not a regular file: ${path##*/}" \
        "$path is a symbolic link, or is not a regular file; cp_commit follows no link here." \
        "$elsewhere"
    fi
    missing+=" $number"
  done
  if [ -n "$other" ]; then
    result "$expected" "unreadable, other content, or a second hard link: files$other" \
      "in $weights, the .weights-are-public files of weight files$other cannot be read, do not" \
      "start with the SHA-256 that $data/weights.sha256 lists for those weight files, or have a" \
      "second hard link. $elsewhere"
  fi
  if [ -n "$foreign" ]; then
    advice="another account (user ID $whose, for the first of them)."
    ways="Either call every stage as the account that owns these files, or keep this account and"
    ways+=" create them anew elsewhere: $elsewhere There this stage prints the command that creates"
    ways+=" the files."
    if [ "$caller" = 0 ] && [ "$whose" = "${SUDO_UID:-}" ]; then
      advice="the account that ran sudo (user ID $whose), but under sudo cp_commit runs as root."
      ways="Two ways work. With sudo: $elsewhere There this stage prints the command that creates"
      ways+=" the files; run that command as root too, for example in a shell opened with"
      ways+=" \"sudo -s\". Without sudo: call every stage as the account that owns these files,"
      ways+=" with a work directory of its own. It must be able to read the weight files: those"
      ways+=" that download fetched under sudo, only root can read."
    fi
    result "$expected, and that belong to the script's account, the weight file's owner or root" \
      "of another owner: files$foreign" \
      "in $weights, the .weights-are-public files of weight files$foreign belong to $advice" \
      "cp_commit counts such a file only if it belongs to the account that runs it (user ID" \
      "$caller), to the owner of its weight file (for the first of them:" \
      "${beside:-not there yet}) or to root, so the stage commit would stop at its start. $ways"
  fi
  # The command below is a group in parentheses, which cannot follow the word sudo.
  if [ "$caller" = 0 ] && [ -n "${SUDO_UID:-}" ]; then
    as_root=' Under sudo, run it as root too, for example in a shell opened with "sudo -s".'
  fi
  if [ -n "$missing" ] && [ -d "$weights" ] && ! weights_closed; then
    shared="Another account may be able to change $weights or a directory above it, or owns"
    shared+=" something in it. A link that it puts there could lead a command to another place,"
    shared+=" so this script gives no command for this directory. If the directory and all in it"
    shared+=" are yours, take write permission from its group and others, and call this stage"
    shared+=" again. Otherwise: $elsewhere There this stage prints the command."
  fi
  # For a declaration that exists and does not count, the script prints no command: it lists the
  # files, one path per line, and points to the page.
  page='docs/operators-duties.md, "Keys, seeds and machines", "Declaring a model file'"'"'s weights'
  page+=' public"'
  if [ -n "$loose" ]; then # whoever can write to a declaration can change what it declares
    result "$expected, and that only their owner can write to" \
      "their group or other users can write to: files$loose" \
      "in $weights, their group or other users can write to the .weights-are-public files of" \
      "weight files$loose, and so could change your declaration, and cp_commit rejects" \
      "them:$loose_paths"$'\n'"Narrowing their permissions is enough only if no other account" \
      "opened a file while it could still write to it: such an account can write to it afterwards" \
      "too. What to do: $page; if a file is not yours, ask its owner to do it"
  fi
  if [ -n "$empty" ]; then
    result "$expected" "empty: files$empty" \
      "in $weights, the .weights-are-public files of weight files$empty are empty, as a command" \
      "that was cut short can leave them. Each declares its weight file public, and must start" \
      "with the SHA-256 that $data/weights.sha256 lists for that file. The command that creates" \
      "a declaration fails while its name is taken:$empty_paths"$'\n'"One way, at the price of a" \
      "second copy of the weight files ($size): $elsewhere There this stage prints the command" \
      "that creates the files. The page on these files: $page"
  fi
  if [ -n "$missing" ]; then
    # umask: for mkdir, and so that other users of the directory can read the files.
    advice="If the weights in these $many files are public, create the missing files from"
    advice+=" weights.sha256: (umask 022; ${make_it}for n in$missing; do"
    advice+=" H=\$(grep -e -$pad\$n-of- $list) && python3 -I -c '$declare_public'"
    advice+=" $G/$stem-$pad\$n-of-$last.gguf.weights-are-public \"\$H\" || exit 1; done)$as_root"
    number=$(first_of "$missing")
    result "$expected" "missing: files$missing" \
      "there is no regular file named ${files[10#$number - 1]}.weights-are-public in" \
      "$weights. Under a published seed nothing of a file's weights is hidden, so each weight" \
      "file must be declared public. ${shared:-$advice}"
  fi
  result "$expected" "$expected"
}
need_commitment() {
  local kept
  if committed; then return; fi
  kept=commitment.$(($(next_number commitment) - 1))/config/limits.txt
  if [ -f "$work/$kept" ]; then # the last call of commit had --limits and did not finish
    stop 1 "$work contains no complete commitment; run first: $me commit $arguments --limits" \
      "$W/$kept (the limits of the last call of commit, which did not finish)"
  elif [ -d "$work/${kept%/*}" ]; then
    stop 1 "$work contains no complete commitment: the last call of commit had --limits and" \
      "stopped before cp_commit; run first: $me commit $arguments --limits FILE"
  fi
  stop 1 "$work contains no complete commitment; run first: $me commit $arguments"
}
# <a file's name>: if it is the name of a weight file of another width than this call's, prints a
# sentence, with a blank before it, that says how to call the script for that width.
other_width() {
  local bits option
  case $1 in
  "$stem"-*) return ;;
  Llama-4-Maverick-17B-128E-Instruct-UD-Q4_K_XL-*.gguf) bits=4 option='without --width' ;;
  Llama-4-Maverick-17B-128E-Instruct-Q8_0-*.gguf) bits=8 option='with --width 8' ;;
  Llama-4-Maverick-17B-128E-Instruct-BF16-*.gguf) bits=16 option='with --width 16' ;;
  *) return ;;
  esac
  printf ' %s' "By its name, $1 is one of the $bits-bit files. If you meant to use those, call" \
    "every stage $option, with a new work directory."
}
own_files() { # every *.gguf in the weights directory is one of $files, not a link, with one name
  local path name known
  for path in "$weights"/*.gguf "$weights"/.*.gguf; do
    if [ ! -e "$path" ] && [ ! -L "$path" ]; then continue; fi # no file matches the pattern
    known=''
    for name in "${files[@]}"; do if [ "${path##*/}" = "$name" ]; then known=yes; fi; done
    if [ -z "$known" ]; then
      stop 1 "$path is not one of the $many weight files. cp_commit and cp_prove read every file" \
        "*.gguf in the weights directory as a part of the model. Copy the $many files into a new" \
        "directory, or let the stage download fetch them there: --weights" \
        "NEW_DIRECTORY.$(other_width "${path##*/}")"
    elif [ -L "$path" ] || { [ -f "$path" ] && [ "$(stat -c %h -- "$path")" != 1 ]; }; then
      stop 1 "$path is a symbolic link, or has a second hard link. cp_commit and cp_prove read a" \
        "weight file only if it is in the directory itself, under one name. Copy the $many files" \
        "into a new directory, or let the stage download fetch them there: --weights NEW_DIRECTORY."
    fi
  done
}
need_room_beside() { # can fit create wsums.txt and fold.txt, unless they are there?
  if [ -d "$weights" ] && [ ! -w "$weights" ] &&
    { [ ! -f "$weights/wsums.txt" ] || [ ! -f "$weights/fold.txt" ]; }; then
    stop 1 "you cannot write to $weights. The stage fit creates two small files there, wsums.txt" \
      "and fold.txt, next to the weight files."
  fi
}
# The calibration set is not in the repository. The user makes it with
# scales/make_calibration_set.py, as calibration_set.npy in the weights directory, and this script
# only reads it there. It must be a regular file, no link, like the other files that the stages read
# in that directory, and have the SHA-256 $calibration_digest: the values in expected.txt hold for
# that set alone. The path goes to no eval here. As in need_marker(), a command that writes to the
# weights directory is printed only if weights_closed() finds no fault.
need_set() {
  local fault='' how got created='Nothing was created.'
  if [ "$stage" != needs ]; then created='No directory fit.N was created.'; fi
  if [ -L "$calibration_set" ]; then
    fault='is a symbolic link'
  elif [ ! -e "$calibration_set" ]; then
    fault='does not exist'
  elif [ ! -f "$calibration_set" ]; then
    fault='is not a regular file'
  elif [ ! -r "$calibration_set" ]; then
    fault='is a file that you cannot read'
  fi
  if [ -n "$fault" ]; then
    how="This script gives no command for this directory: you cannot write in it, or another"
    how+=" account may be able to change it or a directory above it, or owns something in it, and a"
    how+=" link that it puts there could lead a command to another place. Either the directory's"
    how+=" owner makes the set there, or you choose another weights directory, one that only you"
    how+=" can write to: copy the $many weight files there, or let the stage download fetch them."
    how+=" If the work directory has a fit, choose a new work directory too."
    if [ "$fault" != 'does not exist' ]; then
      how="The stage takes only a regular file that you can read, and no link."
      how+=" make_calibration_set.py writes only where nothing has the name, and this script"
      how+=" replaces and removes no file. To look at what has the name: ls -ld $S"
    elif [ -w "$weights" ] && weights_closed; then
      how="To make it, as the account that runs this script, with the python3 that has the"
      how+=" packages of requirements.txt: python3"
      how+=" scales/make_calibration_set.py TRAIN_PARQUET LLAMA2_TOKENIZER_JSON"
      how+=" LLAMA4_TOKENIZER_JSON $S (the three words in capitals: the paths of your copies of the"
      how+=" three files). Then call again: $me $asked $arguments"
    fi
    stop 1 "stopped in stage $stage: the calibration set, $S, $fault. The stage fit reads it" \
      "there. It is not in this repository: scales/make_calibration_set.py makes it from three" \
      "published files, which you fetch yourself. scales/README.md, \"The calibration set\"," \
      "lists them. Two of them, the tokenizers of Llama 2 and Llama 4, are open only to users" \
      "whose request for access the publisher has granted; the request includes accepting the" \
      "model's licence. $created $how"
  fi
  announce "sha256sum <$S"
  got=$(digest_of "$calibration_set")
  same "$calibration_digest" "$got" "$S has the SHA-256 $got, and this script expects" \
    "$calibration_digest. The values in $data/expected.txt hold for the set with that sum alone." \
    "If scales/make_calibration_set.py made your file: it says at its end whether its two sums" \
    "are as expected. If the drawn rows' sum is as expected and the set's is not, the difference" \
    "arose in its steps 4 to 7, else in its steps 1 to 3. Another version of Python or of the" \
    "package tokenizers may give another set (scales/README.md, \"The calibration set\"); the" \
    "line CHECKED_WITH in that program has the versions with which the sums were checked. It" \
    "writes only where nothing has the name, and this script replaces and removes no file. So," \
    "before you make the set again, you must yourself take this one file, $S, out of the way." \
    "With another set this script does not go on: it could give other scales and so another" \
    "commitment ID than the one in expected.txt. scales/README.md, \"A real model\", has the" \
    "steps for such a commitment of your own. $created"
}
need_files() {
  local index
  own_files
  for index in "${!files[@]}"; do
    if [ "$(length_of "${files[index]}")" -ne "${lengths[index]}" ]; then
      stop 1 "$weights/${files[index]} has $(length_of "${files[index]}") of ${lengths[index]}" \
        "bytes. To fetch it: $me download $arguments"
    fi
  done
}
# <file> <directory> <name> <ending>: writes a copy of the file to <name><ending> in the directory,
# gives it the name <name> too, then removes the first name, also after a failure. Fails if either
# name is taken. The directory is opened once, so these steps act in the same one even if another
# account puts something else at its path; the link goes by the open file: its name can be swapped.
lay_beside='import os, sys; source, directory, name, ending = sys.argv[1:]; os.umask(0o022)
data = open(source, "rb").read(); flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
there = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
part = os.open(name + ending, flags, 0o644, dir_fd=there)
try:
  with os.fdopen(part, "wb") as file:
    file.write(data); file.flush(); os.fsync(part)
    os.link("/proc/self/fd/%d" % file.fileno(), name, dst_dir_fd=there)
finally: os.unlink(name + ending, dir_fd=there)'
# cp_commit and cp_prove read wsums.txt and fold.txt next to the weight files. With "create", in
# fit, each is copied there from $fit/model if nothing has its name: first to NAME.part-NUMBER, so
# that a failed write leaves no short file called NAME. Fit, commit and prove compare the copies.
beside_weights() { # [create]
  local name
  for name in wsums.txt fold.txt; do
    if [ $# -eq 1 ] && [ ! -e "$weights/$name" ] && [ ! -L "$weights/$name" ]; then
      # The weights directory may be shared, and root may be the caller. O_EXCL creates the file
      # only if nothing has its name, and follows no link there ("set -C" would open a link to a
      # device); link() gives it its final name only if nothing has that one. Mode 644: the weights
      # are declared public, and another user of the directory needs the file.
      # readlink: the caller's own link may lead there; the program refuses a link as the last part.
      check "python3 -I -S -B -c '$lay_beside' $W/$fit/model/$name" \
        "$(printf %q "$(readlink -e -- "$weights")") $name .part-$$"
      status_0 "$weights/$name could not be created, or its temporary name $name.part-$$ could" \
        "not be removed: python3's last line above gives the reason. This stage writes wsums.txt" \
        "and fold.txt to the weights directory, so you must be able to write there, its disk" \
        "needs room for them, and its file system must offer hard links."
    fi
    check "test -f $G/$name -a ! -L $G/$name -a \"\$(stat -c %h $G/$name)\" = 1 &&" \
      "cmp $W/$fit/model/$name $G/$name"
    status_0 "$weights/$name is missing, cannot be read, differs from $work/$fit/model/$name," \
      "which the stage fit computed, is a symbolic link, or has a second hard link (a" \
      "$name.part-NUMBER next to it, left by a call that was killed, can be that second name)." \
      "cp_commit and cp_prove read it next to the weight files, and this script replaces no file." \
      "Where nothing has this name, the stage fit creates the file: $me fit $arguments"
  done
}

# <directory, as printed>: cp_verify's planning call, which reads no proof: does the verifier accept
# this policy.txt, KL.bin and layout.txt? Some of its lines give digests that cover policy.txt (see
# stage_fit()), so sed passes on only the last line, and the line that reports parameters outside
# the standard form.
plan() {
  local expected="exit status $plan_status and only the line $plan_line"
  check "$verifier -plan accounting $1 128 2>&1 |" \
    "sed -n '/^h_par=/d; /^configuration=/d; /^$plan_warning/p; \$p'"
  if [ "$status" -ne "$plan_status" ] || [ "$out" != "$plan_line" ]; then
    result "$expected" "exit status $status, and the lines above" \
      "cp_verify does not take the files in $1, so no stage uses them; a last line \"NO:\" names" \
      "the test that failed. Is $verifier built from this version? Is $base unchanged?"
  fi
  result "$expected" "exit status $status and only that line"
}

declare -A needs=( # what each stage needs
  [build]="build: CUDA 13.2 Update 2 (nvcc 13.2.86), gcc and g++ 13, binutils.
  examples/quickstart-gpu.sh checks for these.
  About $build_time."
  [download]="download: curl; $total bytes ($exact) of disk space for $many files.
  The files are a third party's conversion of Meta's model. Its model card on Hugging Face has the
  fields \"license: other\" and \"license_name: llama4\": see THIRD_PARTY.md."
  [checksums]='checksums: sha256sum. It reads all files once.'
  [fit]="fit: the GPU; on PATH a python3 with the packages of requirements.txt. The check prints:
  $versions.
  bin/weight_sums reads all files twice, in $sums_time. scales/calibrate.py takes $calibrate_time.
  scales/calibrate.py reads the calibration set, calibration_set.npy in the weights directory. The
  set has 131,200 bytes and is not in this repository: scales/make_calibration_set.py makes it from
  three published files. Two of them, the tokenizers of Llama 2 and Llama 4, are open only to users
  whose request for access the publisher has granted; the request includes accepting the model's
  licence.
  The weights directory's file system must offer hard links, unless it has wsums.txt and fold.txt."
  [commit]="commit: the GPU and its driver; $commit_memory GB of available host
  memory; libcuda.so.1, which comes with the driver, registered with the loader; a CPU with the SHA
  extensions; $commit_disk GB on a local disk. cp_commit takes $commit_time."
  [compare]='compare: sha256sum. A few seconds.'
  [prove]="prove: the GPU; $prove_memory GB of available host memory; python3; $prove_disk GB of
  disk space. cp_prove takes $prove_time, cp_verify $verify_time."
)
stage_needs() {
  local name
  say 'In all: one NVIDIA H100 with 80 GB, on which nothing else is running,' \
    "$commit_memory GB of available host memory, $all_disk GB of disk space."
  say "The programs' times below were measured with $taken_with, on a machine with" \
    "$machine." \
    "There, not counting file cache, the most host memory in use was $commit_peak while cp_commit" \
    "ran and $prove_peak while cp_prove ran."
  for name in build download checksums fit commit compare prove; do say "${needs[$name]}"; done
  check_work
  need_closed_weights
  check 'type -P curl flock sha256sum ldd python3'
  if [ "$status" -ne 0 ]; then
    result 'five paths' "exit status $status" "one of curl, flock (in util-linux), sha256sum (in" \
      "coreutils), ldd and python3 is not on PATH (its path is missing above)"
  fi
  result 'five paths' 'five paths'
  own_files
  need_room_beside
  check_gpu
  fitted || check_python
  if committed; then # only compare and prove are left
    check_memory "$prove_memory" cp_prove "$prove_peak"
    check_disk "$work" "$(kb_of "$prove_disk")"
  else
    check_memory "$commit_memory" cp_commit "$commit_peak"
    check_disks
  fi
  need_marker
  fitted || need_set # after need_marker(): the weights directory is there now
}
stage_build() {
  local script log
  umask 022 # so that another user can run what this user has built
  for script in "${builds[@]}"; do
    log=$(new_log "build-${script%%/*}")
    run "$log" "$script"
    status_0 "$script failed; its output is in $log;" \
      'docs/quickstart.md, "3. GPU: a small model", lists what the builds need'
  done
  umask 077
  if [ "$asked" = all ]; then check_gpu; fi # before the download: can the programs start?
}
curl_line() { # the command line that fetches one file, or the rest of it
  printf '%s' "curl --disable --fail --location --silent --show-error --speed-limit 100000" \
    " --speed-time 60" \
    " --continue-at - --output $G/$1 $repository/resolve/$revision/$folder/$1"
}
create() { # <path>: creates an empty file, or stops if anything has that name, even a broken link
  local path=$1
  # The test comes first because "set -C" protects only regular files: ">" would open a FIFO.
  { [ ! -e "$path" ] && (set -C; true >"$path"); } 2>/dev/null ||
    stop 1 "stopped in stage download: $path could not be created: the name is taken, or the" \
      "directory cannot be written to. Look at it with ls -ld $(printf %q "$path"), or use" \
      "--weights NEW_DIRECTORY, where the script fetches all $many files."
}
fetch() { # <index>: calls curl until the file is complete or five successive calls added nothing
  local code=0 idle=0 before
  # Ctrl-C must stop this loop and curl along with the script. In a script, bash starts background
  # commands with SIGINT ignored (5.3 does so for a function too), and curl would inherit that.
  # With this trap curl starts with the default action, and fetch's subshell ("&") exits after curl.
  trap 'exit 1' INT
  while [ "$idle" -lt 5 ]; do
    before=$(length_of "${files[$1]}")
    eval "$(curl_line "${files[$1]}")"
    code=$?
    if [ "$(length_of "${files[$1]}")" -eq "${lengths[$1]}" ]; then return 0; fi
    if [ "$code" -eq 22 ]; then return 22; fi # an HTTP error; curl's message has its number
    if [ "$(length_of "${files[$1]}")" -gt "$before" ]; then idle=0; else idle=$((idle + 1)); fi
    sleep 30
  done
  return $((code ? code : 1))
}
# download appends to the weight files by their paths for hours. So, while bytes are missing, the
# weights directory must be closed: open_above() finds no fault on its path, its group and others
# cannot write in it even if it is sticky, and foreign_in() finds nothing in it.
weights_seen='' # what need_closed_weights() found: complete (so not judged), absent or closed
need_closed_weights() {
  local path=$weights/ mode
  weights_seen=complete
  if [ "$(missing_bytes)" -eq 0 ]; then return; fi
  weights_seen=closed
  if [ ! -e "$weights" ] && [ ! -L "$weights" ]; then path=${weights%/*}/ weights_seen=absent; fi
  if open_above "$path"; then
    if [ "$why" = 'it does not exist' ]; then
      stop 2 "$place, above the weights directory, does not exist: make it first; download makes" \
        "only the weights directory. Nothing was fetched."
    elif [ "$why" = 'it is not a directory' ]; then
      stop 2 "$place, at or above the weights directory, is not a directory. Nothing was fetched."
    elif [ "$why" = 'it could not be examined' ]; then
      stop 2 "$place, at or above the weights directory, could not be examined: this script needs" \
        "id, GNU stat and readlink, and permission to enter each directory above the weights" \
        "directory. Nothing was fetched."
    fi
    # No $cure (chmod) here: what was put below the place while it was open would still be there.
    stop 2 "the weights directory $G is, or lies below, $place, which another account may be able" \
      "to change ($why): it could put a link in a weight file's place while download appends to" \
      "it and choose what is written to as you. Name a new weights directory below directories" \
      "that only you or root can change (such as one in your home directory, or in /var/tmp," \
      "whose sticky bit keeps others from replacing it), or fetch the missing files yourself;" \
      "nothing was fetched"
  fi
  if [ "$weights_seen" = absent ]; then return; fi
  mode=$(LC_ALL=C stat -L -c %a -- "$weights") ||
    stop 2 "the weights directory $G could not be examined. Nothing was fetched."
  if ((8#$mode & 8#22)) || [ -n "$(foreign_in "$weights")" ]; then
    stop 2 "its group or others can write in the weights directory $G, or it holds something that" \
      "belongs to another account: download appends to files there, so name a new weights" \
      "directory, or fetch the missing files yourself. Nothing was fetched."
  fi
}
may_write() { # the directory was judged only if bytes were missing at the stage's start
  [ "$weights_seen" != complete ] ||
    stop 1 "stopped in stage download: a weight file in $weights was complete when this call" \
      "began and is not now: something else is changing the directory. Nothing was fetched."
}
# For stage_download(), whose $count it advances: waits ten seconds, and prints one line a minute.
wait_and_tell() {
  sleep 10
  count=$((count + 1))
  if [ $((count % 6)) -eq 0 ]; then
    say "$(date +%H:%M:%S) still to fetch: $(gigabytes "$(missing_bytes)") of $exact"
  fi
}
stage_download() {
  local index name have unmarked='' count=0 started=() logs=() failed=''
  need_closed_weights
  own_files
  for index in "${!files[@]}"; do
    name=${files[index]} have=$(length_of "${files[index]}")
    if [ "$have" -gt "${lengths[index]}" ]; then
      stop 1 "stopped in stage download: $weights/$name has $have bytes, more than the published" \
        "${lengths[index]}, and this script does not replace files. With --weights" \
        "NEW_DIRECTORY it fetches all $many files there."
    elif [ -e "$weights/$name" ] && [ ! -f "$weights/$name" ]; then
      stop 1 "stopped in stage download: $weights/$name is not a regular file. With --weights" \
        "NEW_DIRECTORY the script fetches all $many files there."
    elif [ "$have" -lt "${lengths[index]}" ] && [ -e "$weights/$name.download-may-append" ] &&
      [ ! -f "$weights/$name.download-may-append" ]; then
      stop 1 "stopped in stage download: $weights/$name.download-may-append is not a regular" \
        "file. With --weights NEW_DIRECTORY the script fetches all $many files there."
    elif [ "$have" -lt "${lengths[index]}" ] && [ -e "$weights/$name" ] &&
      [ ! -f "$weights/$name.download-may-append" ]; then
      unmarked+=" ${numbers[index]}"
    fi
  done
  if [ -n "$unmarked" ]; then
    index=$((10#$(first_of "$unmarked") - 1)) name=${files[10#$(first_of "$unmarked") - 1]}
    stop 1 "stopped in stage download: $weights/$name has $(length_of "$name") of the published" \
      "${lengths[index]} bytes and no file $name.download-may-append next to it. The script" \
      "appends only to a weight file that has this empty file next to it, and creates one for" \
      "each weight file that it starts. Nothing in $weights was changed. Either fetch all $many" \
      "files elsewhere, with --weights NEW_DIRECTORY. Or, if each file named in this command is" \
      "the beginning of the published one, type the command and call download again, which" \
      "appends the rest (the stage checksums checks the result): (for n in$unmarked; do python3" \
      "-I -c '$create_empty' $G/$stem-$pad\$n-of-$last.gguf.download-may-append || exit 1; done)"
  fi
  # Nothing had the name when need_closed_weights() looked, so there is no test here: mkdir fails if
  # anything has appeared since. -m also limits a default access control list.
  if [ "$weights_seen" = absent ] && ! mkdir -m 700 -- "$weights"; then
    stop 1 "stopped in stage download: $weights could not be created"
  fi
  for name in model/wsums.txt config/KL.bin; do
    if [[ ! $(expected_of "$name") =~ ^[0-9a-f]{64}$ ]]; then
      say "$me: note: $data/expected.txt has no value for $name, so no stage after checksums" \
        "can finish on these $size of files."
      break
    fi
  done
  if [ "$(missing_bytes)" -gt 0 ]; then check_disks; fi
  # Each missing file is created here, not by curl, which would write through a link that has
  # appeared at the file's name. The .download-may-append file comes first, so a call that is cut
  # short between the two leaves no weight file that the next call would reject.
  for name in "${files[@]}"; do
    if [ -e "$weights/$name" ]; then continue; fi
    may_write
    if [ ! -f "$weights/$name.download-may-append" ]; then
      create "$weights/$name.download-may-append"
    fi
    create "$weights/$name"
    say "Created empty: $name, with $name.download-may-append next to it"
  done
  for index in "${!files[@]}"; do
    if [ "$(length_of "${files[index]}")" -eq "${lengths[index]}" ]; then
      say "${files[index]} is complete and is not fetched again."
    else
      # At most five at a time, as many as there are 4-bit files. If this script alone is killed,
      # the calls of fetch() that run go on until their files are complete; no further one starts.
      while [ -n "$(jobs -pr | sed -n 5p)" ]; do wait_and_tell; done
      may_write
      announce "$(curl_line "${files[index]}") &"
      logs+=("$(new_log "curl-$((index + 1))")")
      fetch "$index" >/dev/null 2>"${logs[-1]}" &
      started+=($!)
    fi
  done
  while [ -n "$(jobs -pr)" ]; do wait_and_tell; done
  for index in "${!started[@]}"; do
    wait "${started[index]}" ||
      failed+=" status $?, $(tail -n 1 "${logs[index]}") (${logs[index]##*/});"
  done
  if [ -n "$failed" ] || [ "$(missing_bytes)" -ne 0 ]; then
    result "$many files, $total bytes" "$(missing_bytes) bytes are missing; curl:$failed" \
      "the download is incomplete; curl's messages are in $work/logs. After error 401, 403 or" \
      "404, look at $repository/tree/$revision/$folder in a browser. Otherwise continue later" \
      "where it stopped: $me download $arguments"
  fi
  result "$many files, $total bytes" "$many files, $total bytes"
}
stage_checksums() {
  need_files
  run "$(new_log checksums)" "(cd $G && sha256sum -c) <$data/weights.sha256"
  if [ "$status" -ne 0 ]; then
    result "$many lines that end in OK" "exit status $status" "a file marked FAILED is not the" \
      "published one. Do not run the other stages on it. download fetches only files that are" \
      "missing or incomplete and does not replace this one; with --weights NEW_DIRECTORY it" \
      "fetches all $many files there."
  fi
  result "$many lines that end in OK" "$many lines that end in OK"
}
stage_fit() {
  local number log threads F
  need_built
  need_files
  need_room_beside
  need_marker
  check_gpu
  check_python
  need_set
  number=$(next_number fit)
  fit=fit.$number F=$W/fit.$number
  # cp_commit and cp_prove read the model from one directory, the weights directory: the weight
  # files, each tensor's exponent and checksum (wsums.txt), and two settings (fold.txt; eres: the
  # residual stream's exponent). cp_commit, not cp_prove, also reads the .weights-are-public files.
  check "mkdir $F $F/model $F/config && (set -C; printf '%s\n' $G >$F/weights-directory.txt)"
  status_0 "the directory could not be created"
  if [ "$(digest_of "$weights/wsums.txt")" = "$(expected_of model/wsums.txt)" ]; then
    check "cp $G/wsums.txt $F/model/" # an earlier fit's file
    status_0 "wsums.txt could not be copied"
  else
    threads=$(cpus) log=$(new_log weight_sums)
    run "$log" "bin/weight_sums --eres 10 --threads $((threads < 64 ? threads : 64))" \
      "--out $F/model/wsums.txt $G/$stem-$pad$braces-of-$last.gguf"
    status_0 "bin/weight_sums failed; its output is in $log." \
      "Another call of this stage writes to a new directory."
  fi
  check "(cd $F && sha256sum model/wsums.txt)"
  compare_values model/wsums.txt "bin/weight_sums is written to give the same file on every" \
    "machine, so the weight files or the program are not the published ones. Has the stage" \
    "checksums passed? Is bin/weight_sums built from this version?"
  check "(set -C; printf 'eres 10\nqkrows asis\n' >$F/model/fold.txt) &&" \
    "cp $base/layout.txt $F/config/"
  status_0 "fold.txt or layout.txt could not be written"
  beside_weights create
  log=$(new_log calibrate)
  run "$log" "python3 -Bs scales/calibrate.py --gguf $G --wsums $F/model/wsums.txt" \
    "--tokens $S --out $F/calibration --device cuda --eres 10"
  status_0 "scales/calibrate.py failed; its output is in $log." \
    "Another call of this stage writes to a new directory and reuses wsums.txt."
  # The fitted values are derived from the weights, and of them this repository has those that
  # THIRD_PARTY.md, "What depends on Meta's models", lists; that list may be incomplete. The few
  # fitted lines of policy.txt could be found by trial from a digest that covers that file, so sed
  # drops "w2p policy" here, h_par= in stage_prove(), and h_par= and configuration= in plan().
  # cp_commit keeps both. cp_verify's "result:", "covered:" and "result-v1" give one fitted line, s.
  check "python3 -Bs scales/words_to_policy.py $base/policy.txt $F/calibration/words.json" \
    "$F/model/wsums.txt $F/config/policy.txt $F/config/KL.bin | sed '/^w2p policy /d'"
  status_0 "scales/words_to_policy.py failed, or rejected the fitted scales"
  check "(cd $F && sha256sum config/KL.bin)"
  compare_values config/KL.bin "This file would give another commitment than the published one," \
    "so no stage uses it. The scales come from one float32 pass on the GPU, and its rounding can" \
    "differ between GPU models, drivers and torch's CUDA libraries. The expected digest was" \
    "computed on an H100 under the versions checked above. Install all packages from" \
    "requirements.txt again, or use another H100 machine, and call this stage again: it writes to" \
    "a new directory and reuses wsums.txt."
  plan "$F/config"
  check "echo 'cp_verify -plan accounting passed on config/' >$F/plan-passed.txt"
  status_0 "plan-passed.txt could not be written"
  say '' 'KL.bin, your file of quantization scales, has the expected digest.' \
    'expected.txt has no digest of policy.txt, because the few fitted lines of policy.txt could' \
    'be found from its digest by trial. The commitment ID covers that file: if yours differed' \
    'from the one in the published commitment, that would not show before the stage compare,' \
    "after cp_commit ($commit_time)."
}
stage_commit() {
  local log fits line C config how='without --limits' next="Next: $me compare $arguments"
  local go="To go on with it: $me compare $arguments (without --limits)."
  if committed; then
    if [ -n "$wide" ] && ! own_limits &&
      [[ ! $(expected_of commitment-id) =~ ^[0-9a-f]{64}$ ]]; then
      next="compare and prove need a commitment ID in $data/expected.txt." go=$next
    fi
    if [ -n "$limits" ]; then
      if own_limits; then how="with the limits in $W/$commitment/config/limits.txt"; fi
      stop 1 "$work/$commitment is a complete commitment, generated $how, and a work directory" \
        "holds only one. $go For a" \
        "commitment under the limits in $L: $me fit NEW_WORK_DIRECTORY --weights $G$wide and then" \
        "$me commit NEW_WORK_DIRECTORY --weights $G$wide --limits $L" \
        "(NEW_WORK_DIRECTORY: a name for a new directory)"
    fi
    stop 1 "$work/$commitment is a complete commitment. For another commitment: $me fit" \
      "NEW_WORK_DIRECTORY --weights $G$wide and then $me commit NEW_WORK_DIRECTORY" \
      "--weights $G$wide (NEW_WORK_DIRECTORY: a name for a new directory)." \
      "$next"
  fi
  need_built
  need_files
  need_marker
  need_fit
  beside_weights
  check_gpu
  check_memory "$commit_memory" cp_commit "$commit_peak"
  check_disk "$work" "$(kb_of "$commit_disk")" local
  commitment=commitment.$(next_number commitment)
  C=$W/$commitment
  check "mkdir $C"
  status_0 "the directory could not be created"
  config=$fit/config
  if [ -n "$limits" ]; then
    # The fit's files stay unchanged, so the commitment gets a copy of the configuration. sed
    # replaces three values in policy.txt and appends none: cp_verify rejects a repeated name.
    config=$commitment/config
    check "mkdir $C/config && cp $W/$fit/config/{layout.txt,KL.bin} $C/config/ && (set -C;" \
      "sed 's/^C .*/C $limit_C/; s/^theta .*/theta $limit_theta/; s/^cap .*/cap $limit_cap/'" \
      "$W/$fit/config/policy.txt >$C/config/policy.txt)"
    status_0 "the commitment's configuration could not be written"
    check "grep -xE '(C|theta|cap) [0-9]+' $C/config/policy.txt"
    same "C $limit_C, theta $limit_theta, cap $limit_cap" "${out//$'\n'/, }" \
      "$work/$fit/config/policy.txt does not have each of the lines C, theta and cap once, in" \
      "this order. Run $me fit $arguments (it writes to a new directory), then this call again."
    plan "$C/config"
    check "(set -C; printf 'C %s\ntheta %s\ncap %s\n' $limit_C $limit_theta $limit_cap" \
      ">$C/config/limits.txt)"
    status_0 "limits.txt could not be written"
  fi
  say '' "Next: cp_commit, $commit_time. After $worst_case_time it prints the result of its" \
    'worst-case check: whether its upper bounds on the values of the model, over all prompts, are' \
    'in range.'
  log=$work/$commitment/cp_commit.log
  run "$log" "$programs/cp_commit -weights $G -config $W/$config -M 1" \
    "-ivk $C/ivk -isk $C/isk $commit_mode"
  line=$fits_line
  fits=$(grep -cxF "$line" "$log")
  if [ "$status" -ne 0 ] || [ "$fits" -ne 1 ]; then
    result "exit status 0 and the line \"$line\"" "exit status $status, and that line $fits times" \
      "cp_commit failed; its output is in $log. Another call of this stage writes to a new" \
      "directory." ${limits:+"Unless cp_commit rejected the limits, give it --limits $L again."}
  fi
  result "exit status 0 and the line \"$line\"" 'exit status 0 and that line'
  if [ -n "$limits" ]; then
    say '' "Next: $me compare $arguments (it prints your commitment ID), then $me prove $arguments"
  fi
}
stage_compare() {
  need_fit
  need_commitment
  local made id of_fit='model/wsums.txt config/KL.bin' of_commitment='ivk/key.bin ivk/leaf/0.bin'
  if own_limits; then # cp_commit read the copy of KL.bin in the commitment's directory
    of_fit=model/wsums.txt of_commitment="config/KL.bin $of_commitment"
  fi
  made=$(new_log made)
  run "$made" "(cd $W/$fit && sha256sum $of_fit) &&" \
    "(cd $W/$commitment && sha256sum $of_commitment &&" \
    "sed -n \"s/^$id_line\\(.*\\)/\\1  commitment-id/p\" cp_commit.log)"
  status_0 "the digests could not be computed: see the messages above"
  out=$(cat -- "$made" 2>&1)
  say "These five lines, in the form of expected.txt, are in $made."
  if own_limits; then
    compare_values 'model/wsums.txt config/KL.bin' "So this commitment does not come from the" \
      "published weight files and the expected scales. Has a file in $work changed since fit and" \
      "commit? Has the stage checksums passed? Generate the commitment again in a new work" \
      "directory: $me fit NEW_WORK_DIRECTORY --weights $G$wide and then $me commit" \
      "NEW_WORK_DIRECTORY --weights $G$wide --limits $W/$commitment/config/limits.txt"
    id=$(printed_id)
    [[ $id =~ ^[0-9a-f]{64}$ ]] || stop 1 "$work/$commitment/cp_commit.log has no commitment ID"
    say '' 'wsums.txt and KL.bin have the digests in expected.txt.' \
      'This commitment was generated with --limits, and the commitment ID covers the limits. So' \
      'the ID, key.bin and leaf/0.bin are your own: nothing is published to compare them with,' \
      'and nothing here checks the fitted lines of policy.txt, which the ID also covers.' \
      "Your commitment ID: $id"
    return
  fi
  compare_values 'model/wsums.txt config/KL.bin commitment-id ivk/key.bin ivk/leaf/0.bin' \
    "So this is not the published commitment. Is $here unchanged? Are the programs built from" \
    "this version? If wsums.txt and KL.bin are equal, the fit's policy.txt may differ; the" \
    "commitment ID covers it. To test the C library and the base file, from which it is" \
    "computed: python3 -Bs scales/policy_tables.py --check $base/policy.txt (if it reports a" \
    "difference, the base file is changed or this machine computes the public tables otherwise:" \
    "take a new copy of the repository, or use another machine; if it finds nothing, install" \
    "all packages from requirements.txt, or use another H100 machine). Then generate the" \
    "commitment again in a new work directory: $me all NEW_WORK_DIRECTORY --weights $G$wide"
  say '' 'Your commitment is the published one.'
}
stage_prove() {
  need_built
  need_fit
  need_commitment
  local id id_file=$data/expected.txt verdict expected proof P name value
  local config=$fit/config again="$me all NEW_WORK_DIRECTORY --weights DIRECTORY_OF_THE_FILES$wide"
  local options='--policy clipped-gap'
  id=$(expected_of commitment-id)
  if own_limits; then
    id=$(printed_id) id_file=$work/$commitment/cp_commit.log config=$commitment/config
    again="$me fit NEW_WORK_DIRECTORY --weights DIRECTORY_OF_THE_FILES$wide and then $me commit"
    again+=" NEW_WORK_DIRECTORY --weights DIRECTORY_OF_THE_FILES$wide"
    again+=" --limits $W/$config/limits.txt"
    for name in C s theta cap; do
      value=$(awk -v name="$name" '$1 == name { print $2 }' "$work/$config/policy.txt" 2>/dev/null)
      [[ $value =~ ^[0-9]{1,10}$ ]] ||
        stop 1 "$work/$config/policy.txt does not have the line \"$name\" once, with a number"
      options+=" --$name $value"
    done
  fi
  [[ $id =~ ^[0-9a-f]{64}$ ]] || stop 1 "$id_file has no commitment ID"
  need_files
  beside_weights
  check_gpu
  check_memory "$prove_memory" cp_prove "$prove_peak"
  check_disk "$work" "$(kb_of "$prove_disk")"
  proof=$work/proof.$(next_number proof) P=$W/proof.$(next_number proof)
  check "mkdir $P $P/in"
  status_0 "the directory could not be created"
  say '' 'The example request: 1,024 token ids from a formula, so no tokenizer is needed.' \
    'The proof covers the token that comes next.'
  check "python3 -Bs prover/tools/make_request.py $W/$config/policy.txt $P/in/example.req"
  status_0 "the request file was not written"
  run "$proof/cp_prove.log" "$programs/cp_prove -isk $W/$commitment/isk" \
    "$prove_mode -in $P/in -out $P/out -in-step 1 -memory $prove_layout"
  expected='exit status 0 and the files example.y and example.proof'
  if [ "$status" -ne 0 ] || [ ! -f "$proof/out/example.proof" ] ||
    [ ! -f "$proof/out/example.y" ]; then
    result "$expected" "exit status $status" "cp_prove failed; its output is in" \
      "$proof/cp_prove.log. It can fail after it has written a response, and no verifier has" \
      "seen what lies in $proof/out: give no one anything from it. Another call of this stage" \
      "writes to a new directory. If the weight" \
      "files or the work directory have moved since fit: $again"
  fi
  result "$expected" 'exit status 0 and both files'
  run "$proof/cp_verify.log" "$verifier $W/$config $W/$commitment/ivk" \
    "$P/in/example.req $P/out/example.y $P/out/example.proof -id $id | sed '/^h_par=/d'"
  verdict=$(tail -n 1 "$proof/cp_verify.log")
  expected="exit status $good_status and the last line $good_verdict"
  if [ "$status" -ne "$good_status" ] || [ "$verdict" != "$good_verdict" ]; then
    result "$expected" "exit status $status and the last line ${verdict:-(none)}" \
      "cp_verify did not accept the proof: give no one anything from $proof/out, nor" \
      "$proof/cp_verify.log, nor this run's log"
  fi
  result "$expected" "exit status $status and the last line $verdict"
  say '' "The proof is $proof/out/example.proof." \
    "The next token's id is the 4-byte number in example.y, next to it."
  if own_limits; then
    say '' "This $good_verdict is about the next token only. The example request has no reply, so" \
      'no token was compared with your limits, and the result shows nothing about them.' \
      'To prove replies that were served earlier, and to check them under your limits:' \
      'integrations/vllm/REFERENCE.md, "Prove replies that were served earlier". Its commands use' \
      'shell variables. Five of them come from this work directory and are the same for every' \
      'reply. ID: your commitment ID. K: the directory of the configuration (policy.txt, KL.bin,' \
      'layout.txt). IVK, ISK: the directories of the public key and the secret key. LIMITS: your' \
      'limits, with the line s of policy.txt, as options of integrations/check_proof.py.' \
      "ID=$id" "K=$W/$config" "IVK=$W/$commitment/ivk" "ISK=$W/$commitment/isk" "LIMITS='$options'"
  fi
}

# Prints one note if open_above() finds a fault above the weights directory, as typed or resolved,
# or above the repository. Of these directories themselves it names only one that all others can
# write in: a directory of the user's own that a umask of 002 left open to its group is common.
open='' cures=''
for path in "$weights" "$(readlink -m -- "$weights")" "$PWD"; do
  if ! open_above "${path%/*}/"; then
    [ -d "$path" ] && (((8#$(LC_ALL=C stat -L -c %a -- "$path") & 8#1002) == 2)) || continue
    printf -v place %q "$path"
    why='others can write in it, and it has no sticky bit'
    [ ! -O "$path" ] || [ "$(id -u)" = 0 ] || cure="; if $place is yours, close it with chmod go-w"
  elif [ "$why" = 'it does not exist' ] || [ "$why" = 'it is not a directory' ]; then
    continue # the stage that needs it stops there
  fi
  if [[ " and $open and " != *" and $place ($why) and "* ]]; then
    open+=${open:+ and }"$place ($why)" cures+=$cure
  fi
done
if [ -n "$open" ]; then
  say "$me: note: another account may be able to change $open, under which this run works, and so" \
    'could change what the stages run or read; for a run that no other account can touch, keep' \
    "the weights directory and the repository below directories that only you can write in$cures" ''
fi
asked=$stage
if [ "$asked" = needs ] || [ "$asked" = all ]; then
  if [ "$asked" = all ]; then
    say "After the stage needs, all output also goes to the newest $work/logs/run.N.log."
  fi
  stage=needs
  stage_needs
  say '' "$me: this machine has what the stages after build need; nothing was written"
  # If expected.txt lacks one of the commitment's three values, compare stops, and without the ID
  # prove too, unless the work directory's commitment was made with --limits. So "all" stops before
  # the long stages; "needs" adds a note.
  # If it lacks one of the fit's two values, fit stops, and every later stage needs a finished fit.
  unrecorded=''
  for name in model/wsums.txt config/KL.bin; do
    [[ $(expected_of "$name") =~ ^[0-9a-f]{64}$ ]] || unrecorded+=" $name"
  done
  if [ -n "$unrecorded" ]; then
    unrecorded="$data/expected.txt has no value for:$unrecorded. So there is nothing to"
    unrecorded+=" compare your run with. The stage fit prints your run's own value and stops, and"
    unrecorded+=' commit, compare and prove need a finished fit. Without these values,'
    unrecorded+=' you can run build, download and checksums: call the script once for each stage'
    if [ "$asked" = all ]; then
      stop 1 "stopped after the stage needs: $unrecorded. Nothing was created; this call has no" \
        "log."
    fi
    say "$me: note: \"all\" would stop here: $unrecorded."
    exit 0
  fi
  for name in commitment-id ivk/key.bin ivk/leaf/0.bin; do
    [[ $(expected_of "$name") =~ ^[0-9a-f]{64}$ ]] || unrecorded+=" $name"
  done
  if [ -n "$unrecorded" ] && ! { committed && own_limits; }; then
    unrecorded="$data/expected.txt has no value for:$unrecorded. So compare cannot finish (it"
    unrecorded+=" prints your run's values and stops), and without the commitment ID neither can"
    unrecorded+=' prove. What works, in a work directory without a commitment: build, download,'
    unrecorded+=' checksums, fit and commit, one call each; or a commitment under your own limits,'
    unrecorded+=" which needs none of the missing values: $here/README.md,"
    unrecorded+=' "Commit under your own limits"'
    if [ "$asked" = all ]; then
      stop 1 "stopped after the stage needs: $unrecorded. Nothing was created; this call has no" \
        "log."
    fi
    say "$me: note: \"all\" would stop here: $unrecorded."
  fi
  [ "$asked" = all ] || exit 0
fi
if ! check_work; then
  # The name first and alone: mkdir fails if anything has taken it since, and then nothing is made
  # below it. -m also limits a default access control list.
  if ! { mkdir -m 700 -- "$work" && mkdir -m 700 -- "$work/logs" "$work/home" "$work/tmp"; }; then
    stop 2 "the work directory $work could not be created"
  fi
  # The line must be there: check_work() reads it.
  { echo "The work directory of $me$wide." >"$work/created-by-maverick-run.txt"; } 2>/dev/null ||
    stop 2 "$work/created-by-maverick-run.txt could not be written (is the disk full?). $work" \
      "stays, and is of no use: choose a new work directory"
fi
# Locked while this script or a program that it started runs.
{ exec 9>>"$work/created-by-maverick-run.txt"; } 2>/dev/null ||
  stop 2 "$work/created-by-maverick-run.txt could not be opened"
flock -n 9 2>/dev/null
status=$?
if [ "$status" -eq 1 ]; then
  stop 1 "another call of this script is still running in $work: wait for it"
elif [ "$status" -ne 0 ]; then
  stop 1 "$work could not be locked (exit status $status of flock). flock comes with util-linux," \
    "and the work directory must be on a local disk."
fi
run_log=$(new_log run)
say "From here on all output of this call also goes to $run_log."
export HOME=$work/home TMPDIR=$work/tmp
stages=("$asked")
if [ "$asked" = all ]; then stages=(build download checksums fit commit compare prove); fi
for stage in "${stages[@]}"; do
  say '' "==== stage $stage"
  if [ "$asked" = all ] && [ "$stage" = build ] && built; then
    say 'The programs are built, so this stage is skipped.'
  elif [ "$asked" = all ] && [[ $stage == checksums || $stage == fit ]] && fitted; then
    say "$work/$fit is a finished fit, so this stage is skipped."
  elif [ "$asked" = all ] && [ "$stage" = commit ] && committed; then
    say "$work/$commitment is a complete commitment, so this stage is skipped."
  else
    say "${needs[$stage]}"
    SECONDS=0
    "stage_$stage"
    took=$SECONDS
    if [ "$took" -eq 0 ]; then took='under a second'; else took+=' s'; fi
    say '' "$me: stage $stage finished as expected, in $took"
  fi
done
say "$me: your files are in $work"
