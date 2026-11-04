#!/bin/bash
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.

# Checks that the production build of the programs cp_commit and cp_prove (prover/build.sh without a
# target; "a serving build" in messages) rejects model files that are links. An honest run can meet
# this rejection: a directory of links, such as a download cache's snapshot directory, is not
# accepted (prover/weights_commitment/gguf_file.h: O_NOFOLLOW, and st_nlink must be 1).
# Runs on synthetic weights open no model files,
# and the test build (prover/build.sh tests) accepts
# none, so neither reaches these checks.
# Messages say `folder` for directory, `kit` for example configuration, `plays` for cases.
# usage: serving_links.sh <directory with cp_commit, cp_prove, cp_verify from prover/build.sh>
#   <work directory: must not exist> <example configuration for model files: policy.txt, KL.bin>
#   <directory, only read, that make_model (prover/build.sh make_model) wrote (no real model)
#   from the arguments in the configuration's model.txt>
# Name the work directory without symbolic links: cp_prove rejects an -isk path that has one, and
# the test then ends FAILED.
# GPU_RUN (environment): words put before each call of cp_commit and cp_prove.
# Needs one idle H100, disk space for six copies of the model files, and about two minutes.
# Last line: "serving_links, version <word>: <n> .., all as expected", status 0. With status 1:
# "serving_links, version <word>: FAILED ..", or "NO END: .." if SIGHUP, SIGINT, SIGKILL or SIGTERM
# ended a GPU program (later cases are not run). 2: the set-up failed. 130: Ctrl-C. No time limit.
# Started by the runner; by hand: below a directory of your own. (Runner: run_suite.py. The work folder goes there.)
set -u
unset CDPATH; trap 'echo "serving_links: stopped by Ctrl-C" >&2; exit 130' INT # cp_prove catches INT and ends by itself with status 1: without this the script would go on to the next case
made_here() { # <a folder that this script has just made>: it must be this account's, closed to others (mode 700; 2700 below a set-group-ID directory) and empty
  # One cd, then three tests of ".", so that all three see the same directory; a failure of stat
  # or ls also rejects it. -P: cd resolves ".." as mkdir does. The trailing slash: cd does not take
  # a directory named "-" for the previous directory.
  # The directories above $1 are not examined. Those that this script makes itself (mkdir -p, below)
  # are made under umask 077, to close them to other accounts as -m 700 closes the work directory.
  (cd -P -- "$1/" && [ -O . ] && [[ $(stat -c %a .) == *700 ]] && list=$(ls -A) && [ -z "$list" ]) && return
  echo "serving_links.sh: $1 is not the folder that was just made: another was put in its place: name a new folder below folders in which only you can write" >&2
  return 1
}
S=${1:?the programs\' folder} W=${2:?a work folder} K=${3:?a kit for the files} M=${4:?a model\'s folder}
here=$(cd -- "$(dirname "$0")" && pwd) || exit 2
TOOLS=${TOOLS:-$here/../tools}
case $W in /*) full=$W ;; *) full=$PWD/$W ;; esac; case /$full/ in */-*|*/../*|*[!A-Za-z0-9_./-]*) echo "serving_links: the work folder's path may hold letters, digits and _ . / - only, and no part of it may be \"..\" or begin with \"-\": name another: $full"; exit 2 ;; esac # the path goes into sed commands below
if [ -e "$W" ]; then echo "serving_links: the work folder $W exists: name a new one"; exit 2; fi
for f in "$S/cp_commit" "$S/cp_prove" "$S/cp_verify"; do [ -x "$f" ] || { echo "serving_links: no program $f"; exit 2; }; done
[ -f "$M/wsums.txt" ] && [ -f "$M/fold.txt" ] && ls "$M"/*.gguf >/dev/null 2>&1 || { echo "serving_links: $M is no model's folder (wsums.txt, fold.txt, *.gguf)"; exit 2; }
(umask 077 && mkdir -p -- "$(dirname -- "$W")") && mkdir -m 700 -- "$W" && made_here "$W" && mkdir -p -- "$W/config" "$W/in" "$W/home" "$W/tmp" "$W/aside" || exit 2 # the mkdir without -p fails if anything has appeared at the name since the -e test on it above
full=$(cd -- "$W" && pwd) && [ "$full" -ef "$W" ] && W=$full || exit 2; M=$(cd -- "$M" && pwd) || exit 2 # -ef: stop unless cd reached the directory just made (a lone "-", or ".." after a symbolic link, can lead cd elsewhere)
export HOME=$W/home TMPDIR=$W/tmp
bad=0 played=0
wrong() { echo "   NOT AS EXPECTED: $*"; bad=1; }
cp "$K/policy.txt" "$K/KL.bin" "$W/config/" || exit 2
V=$(awk '$1 == "version" { print $2; exit }' "$K/policy.txt")
"$S/cp_verify" -plan arrays "$W/config" 128 >"$W/arrays.txt" 2>&1 # a planning call: its status is 4
python3 -B "$TOOLS/pack_layout.py" --arrays "$W/arrays.txt" 17 4 >"$W/config/layout.txt" 2>"$W/pack.err" || { echo "the layout file cannot be made: $(head -c 300 "$W/pack.err")"; echo "serving_links, version $V: FAILED"; exit 1; }
python3 -B "$TOOLS/make_request.py" "$W/config/policy.txt" "$W/in/example.req" || exit 2
shard=$(cd -- "$M" && ls -- *.gguf | head -n 1)
# cp_commit's cases. Case <case> runs on m.<case>: copies of the model files, changed as listed.
#   plain                  copies only, for comparison: status 0
#   shard-link             one *.gguf file ("a shard" in messages) is a symbolic link: rejected
#   shard-second           one *.gguf has a second name (a hard link): rejected
#   wsums-link, fold-link  wsums.txt (checksum file) or fold.txt (conversion settings) is a symbolic
#                          link: rejected, with a message that does not mention the link
#   shard-fifo             one *.gguf is a FIFO: rejected, without waiting for a writer
#   folder-link            m.folder-link is itself a symbolic link, to m.plain: status 0;
#                          take_weights(), prover/weights_commitment/arguments.h, resolves it first
folder() { mkdir "$W/m.$1" && cp "$M"/*.gguf "$M/wsums.txt" "$M/fold.txt" "$W/m.$1/"; }
aside() { mv "$W/m.$1/$2" "$W/aside/$1.$2"; } # the true file goes out of the folder
folder plain || exit 2
folder shard-link && aside shard-link "$shard" && ln -s "$W/aside/shard-link.$shard" "$W/m.shard-link/$shard"
folder shard-second && ln "$W/m.shard-second/$shard" "$W/aside/shard-second.$shard"
folder wsums-link && aside wsums-link wsums.txt && ln -s "$W/aside/wsums-link.wsums.txt" "$W/m.wsums-link/wsums.txt"
folder fold-link && aside fold-link fold.txt && ln -s "$W/aside/fold-link.fold.txt" "$W/m.fold-link/fold.txt"
folder shard-fifo && aside shard-fifo "$shard" && mkfifo "$W/m.shard-fifo/$shard"
ln -s "$W/m.plain" "$W/m.folder-link"
# commit <case> <taken | refused> <text that the log must contain, or ->
# `taken`: status 0 and at least one file under ivk.<case>, the public key directory. The message
# says `key` (key.bin) and `entry` (a leaf file, leaf/<k>.bin), but no file's name is checked.
# `refused`: rejected, with no file left under ivk.<case>, isk.<case> (the secret key directory) or
# the .making directories in which cp_commit works. Only regular files count: empty directories
# that remain pass.
commit() {
  local n=$1 want=$2 words=$3 got f left s0; played=$((played + 1)); s0=$(date +%s)
  # No timeout on purpose: ending cp_commit while its GPU kernel runs can take the whole machine out
  # of use. run_suite.py reports a GPU test still alive at its time limit and sends it no signal.
  # shellcheck disable=SC2086 # GPU_RUN is a list of words
  ${GPU_RUN:-} "$S/cp_commit" -weights "$W/m.$n" -config "$W/config" -M 1 -ivk "$W/ivk.$n" -isk "$W/isk.$n" >"$W/$n.log" 2>&1; got=$?
  case $got in 129 | 130 | 137 | 143) # 128 + the signal: ended by SIGHUP, SIGINT, SIGKILL or SIGTERM. What such an end leaves on the GPU is not known, so no further program is started there
    echo "NO END: [$n] a signal ended cp_commit (status $got): nothing is shown on this play or on those behind it; what it printed: $W/$n.log"
    exit 1 ;;
  esac # These four come from outside, mostly to a program that hangs. The program's own crash (134: abort(), 139: SIGSEGV) does not stop the script: the lines below judge it like any other status but 0
  f=$(find "$W/ivk.$n" -type f 2>/dev/null | wc -l)
  left=$(find "$W/ivk.$n"* "$W/isk.$n"* -type f 2>/dev/null | wc -l) # the * also matches <folder>.making, where cp_commit works until it succeeds
  echo "[$n] cp_commit status $got in $(( $(date +%s) - s0 )) s; files under ivk: $f$([ "$got" = 0 ] || echo "; its last line: $(tail -n 1 "$W/$n.log" | cut -c1-200)")"
  if [ "$want" = taken ]; then [ "$got" = 0 ] && [ "$f" != 0 ] || wrong "$n: wanted status 0, a key and an entry"
  else [ "$got" != 0 ] && [ "$left" = 0 ] || wrong "$n: expected a refusal (a nonzero status) and no file under ivk, isk or their .making folders; files found: $left"; fi
  [ "$words" = - ] || grep -a -q -F -- "$words" "$W/$n.log" || wrong "$n: the log lacks «$words»"
}
commit plain taken -
commit shard-link refused "a shard is a symbolic link, which a serving build does not follow"
commit shard-second refused "a shard has a second name (a hard link)"
commit wsums-link refused "wsums.txt cannot be opened"
commit fold-link refused "fold.txt cannot be read"
commit shard-fifo refused "a shard is no regular file"
commit folder-link taken -
# cp_prove's cases. The commitment is the one to m.plain, which is changed only afterwards.
#   prove-plain  status 0, proof file and token file (example.y; "reply" in messages), which
#                cp_verify accepts; count.bin,
#                whose length counts the columns of the openings begun on leaf table 0 (the weight
#                table that requests are proved against), has grown
#   prove-link   one *.gguf replaced by a symbolic link: status 1, no response, and count.bin has
#                not grown. cp_prove opens the files after ClosedFront::close()
#                (prover/prover/closed_front.h), which limits its output to fixed lines: the last
#                names the rejection, without a reason: "p258 end 1 cp-weights-files"
#   prove-again  the file restored: as prove-plain
id=$(sed -n "s/^commitment ID: //p" "$W/plain.log" | tr -d ' ')
count() { stat -c %s "$W/isk.plain/leaf/0/count.bin" 2>/dev/null; }
files() { find "$1" -type f 2>/dev/null | sed "s#^$1/##" | LC_ALL=C sort | tr '\n' ' '; }
prove() { # <name>: sets got, out, grown
  local c; played=$((played + 1)); c=$(count)
  # shellcheck disable=SC2086
  ${GPU_RUN:-} "$S/cp_prove" -isk "$W/isk.plain" -in "$W/in" -out "$W/out.$1" >"$W/$1.screen" 2>&1; got=$?; case $got in 129 | 130 | 137 | 143) echo "NO END: [$1] a signal ended cp_prove (status $got): nothing is shown on this play or on those behind it; what it printed: $W/$1.screen"; exit 1 ;; esac # as in commit()
  out=$(files "$W/out.$1") grown=$(( $(count) - c ))
  echo "[$1] cp_prove status $got; in -out: ${out:-nothing}; the count grew by $grown; the last line: $(tail -n 1 "$W/$1.screen" | cut -c1-140)"
}
verified() { "$S/cp_verify" "$W/config" "$W/ivk.plain" "$W/in/example.req" "$W/out.$1/example.y" "$W/out.$1/example.proof" -id "$id" >"$W/$1.verify.log" 2>&1 || wrong "$1: cp_verify does not accept the response: $(tail -n 1 "$W/$1.verify.log" | cut -c1-160)"; } # <name>
# If count.bin cannot be read, `grown` is 0. So prove-plain and prove-again require `grown` > 0:
# otherwise prove-link's "the count unmoved" would also hold with the file missing.
prove prove-plain; [ "$got" = 0 ] && [ "$out" = "example.proof example.y " ] && [ "$grown" -gt 0 ] || wrong "prove-plain: wanted status 0, proof and reply, the count grown"
verified prove-plain
aside plain "$shard" && ln -s "$W/aside/plain.$shard" "$W/m.plain/$shard"
prove prove-link; [ "$got" = 1 ] && [ -z "$out" ] && [ "$grown" = 0 ] && [ "$(tail -n 1 "$W/prove-link.screen")" = "p258 end 1 cp-weights-files" ] || wrong "prove-link: wanted status 1, «p258 end 1 cp-weights-files», no response, the count unmoved"
rm "$W/m.plain/$shard" && mv "$W/aside/plain.$shard" "$W/m.plain/$shard" # the link goes, the true file comes back
prove prove-again; [ "$got" = 0 ] && [ "$out" = "example.proof example.y " ] && [ "$grown" -gt 0 ] || wrong "prove-again: wanted status 0, proof and reply, the count grown"
verified prove-again
if [ $bad = 0 ]; then echo "serving_links, version $V: $played plays, all as expected"; else echo "serving_links, version $V: FAILED: see the lines NOT AS EXPECTED above; the logs lie in $W"; exit 1; fi
