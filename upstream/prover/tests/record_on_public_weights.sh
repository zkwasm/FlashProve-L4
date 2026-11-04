#!/bin/bash
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.

# Started by the runner; by hand: below a directory of your own. (Runner: run_suite.py.)
# GPU test: cp_commit (the program that commits to a model's weights) makes a commitment on a
# model's files, the prover cp_prove serves a request from it, and the verifier
# cp_verify accepts the response. All three are run twice: under a seed that cp_commit draws, and
# with the option -public-weights, under which nothing is hidden (prover/common/public_weights.h).
# cp_prove serves a commitment that was made with the option only with the option.
# With the line `record 1` in the parameter file (policy.txt), a weight table publishes a weighted
# sum of each row's weights, each hidden by a random pad of its own.
# cp_commit treats every directory of weights as a model's files, these synthetic ones too.
#
# Not shown: whether a record hides anything; the full size; real weights; more than one leaf
# table, request or batch; cp_prove's service on a spool; an altered file of such a commitment
# (call 7 alters the ID); the same bytes on another machine; that the published sums are the right
# ones, except as far as cp_verify's checks depend on them. A YES here rests on the theorem on
# codes (ePrint 2025/2055, Theorem 4.6), taken as given: what has been checked of it, and what has
# not, docs/limits.md says (what-the-theorem-assumes). Call 6 prints the two lines that count the
# rounds.
#
# usage: record_on_public_weights.sh <programs> <work> <configuration> <model>, four directories:
#   <programs>       cp_commit, cp_prove, cp_verify of the production build (prover/build.sh's
#                    default): no other build accepts model files or has the option
#   <work>           new; made with mode 700, as is each missing directory above it. Name it below
#                    directories that only you can write to, with no symbolic link on the way: the
#                    script does not check this, and cp_prove rejects a path with a link, or
#                    through a directory that others can write. Gets a copy of the model files
#                    (530 MB). The script writes only there and removes nothing
#   <configuration>  prover/examples/target_small_file_f2: policy.txt (a parameter file with
#                    `record 1`), KL.bin (quantization scales)
#   <model>          written by make_model (prover/tests/weights_blocks/make_model.cpp) for the
#                    configuration's model.txt. The script marks its copy as public: never name
#                    weights that must stay hidden
# GPU_RUN in the environment: a command prefix for each call of cp_commit and cp_prove.
# Needs one H100 that nothing else uses during these calls.
# Last line, except at status 2 or if an argument is missing (bash's own message, status 1):
#   "record_on_public_weights, version <the version word>: ...".
# Status 0: all as expected; 1: the line counts the differences; 2: the set-up failed (an argument,
# a directory or a file was unusable); 3: cp_verify accepted the response under another ID.
set -u
unset CDPATH
made_here() { # <a directory that this script has just made>: it must be this account's, closed to others (mode 700; 2700 below a set-group-ID directory) and empty
  # One cd, then three tests of ".", so that all three see the same directory. If stat or ls fails,
  # the directory is rejected too. -P: cd takes ".." after a symbolic link as mkdir does, not by the
  # path's text. The slash: cd does not take a directory named "-" for the previous directory.
  (cd -P -- "$1/" && [ -O . ] && [[ $(stat -c %a .) == *700 ]] && list=$(ls -A) && [ -z "$list" ]) && return
  echo "record_on_public_weights.sh: $1 is not the directory that was just made: another was put in its place: name a new directory below directories in which only you can write" >&2
  return 1
}
B=${1:?argument 1 is missing: the directory with cp_commit, cp_prove and cp_verify} W=${2:?argument 2 is missing: a new work directory} K=${3:?argument 3 is missing: the configuration with policy.txt} M=${4:?argument 4 is missing: the directory that make_model wrote}
here=$(cd -- "$(dirname "$0")" && pwd) || exit 2
case $W in /*) full=$W ;; *) full=$PWD/$W ;; esac; case /$full/ in */-*|*/../*|*[!A-Za-z0-9_./-]*) echo "record_on_public_weights: the work directory's path may contain only letters, digits and _ . / -, and no part of it may be \"..\" or begin with \"-\": name another: $full"; exit 2 ;; esac # cp_prove accepts HOME and TMPDIR, which lie below it, only with these characters
if [ -e "$W" ]; then echo "record_on_public_weights: the work directory $W exists: name a new one"; exit 2; fi
[ -f "$M/wsums.txt" ] && [ -f "$M/fold.txt" ] && ls "$M"/*.gguf >/dev/null 2>&1 || { echo "record_on_public_weights: $M has no model files (wsums.txt, fold.txt, *.gguf)"; exit 2; }
[ "$(grep -c '^record 1$' "$K/policy.txt" 2>/dev/null)" = 1 ] || { echo "record_on_public_weights: $K/policy.txt is missing or lacks the line \"record 1\""; exit 2; }
# The script will mark its copy of the model files as public. Against a mistake (any file can be
# given the name) it takes only files with make_model.cpp's pair general.name = "made-up" in the
# first 128 bytes. tr removes the type and length bytes between key and value.
for f in "$M"/*.gguf; do
  head -c 128 "$f" | LC_ALL=C tr -d '\000-\037' | grep -a -q 'general\.namemade-up' || { echo "record_on_public_weights: $f does not look like a file that make_model wrote: its first 128 bytes lack the name \"made-up\" (general.name). This script marks its copy of the model files as public, so it takes no others. Nothing was made: name a directory that make_model wrote"; exit 2; }
done
(umask 077 && mkdir -p -- "$(dirname -- "$W")") && mkdir -m 700 -- "$W" && made_here "$W" && mkdir -p -- "$W/config" "$W/model" "$W/in" || exit 2 # the mkdir without -p fails if anything has appeared at the name since the -e test on it above
full=$(cd -- "$W" && pwd) && [ "$full" -ef "$W" ] && W=$full || exit 2; M=$(cd -- "$M" && pwd) || exit 2 # -ef: stop unless cd reached the directory just made (a lone "-", or ".." after a symbolic link, can lead cd elsewhere)
export HOME=$W/home TMPDIR=$W/tmp; mkdir "$HOME" "$TMPDIR" || exit 2 # the production build's cp_prove needs both to name directories
V=$(awk '$1 == "version" { print $2; exit }' "$K/policy.txt") # the version word: H821 (p = 2^64 - 2^32 + 1)
bad=0 accepted=0
no() { echo "NOT AS EXPECTED: $*"; bad=$((bad + 1)); }
end() { if [ $accepted = 1 ]; then echo "record_on_public_weights, version $V: cp_verify accepted a response under another commitment ID"; exit 3; fi; if [ $bad = 0 ]; then echo "record_on_public_weights, version $V: all as expected"; exit 0; fi; echo "record_on_public_weights, version $V: $bad not as expected"; exit 1; }
has() { grep -a -q -- "$2" "$W/$1" || no "$W/$1 has no line that matches \"$2\""; } # <log> <pattern>
lacks() { grep -a -q -- "$2" "$W/$1" && no "$W/$1 has a line that matches \"$2\""; } # <log> <pattern>
cp "$K/policy.txt" "$W/config/policy.txt" && cp "$K/KL.bin" "$W/config/" || exit 2
"$B/cp_verify" -plan arrays "$W/config" 128 >"$W/arrays.txt" 2>&1 # a planning call: its status is 4
python3 -B "$here/../tools/pack_layout.py" --arrays "$W/arrays.txt" 17 4 >"$W/config/layout.txt" 2>"$W/pack.err" || { no "the layout file cannot be made: $(head -c 300 "$W/pack.err"); the last line of cp_verify -plan arrays: $(tail -n 1 "$W/arrays.txt" | cut -c1-200)"; end; }
# The model's copy gets a marker beside each model file, written as cp_commit's message tells an
# operator to: sha256sum's output, in a file that its group and others cannot write.
cp "$M"/*.gguf "$M/wsums.txt" "$M/fold.txt" "$W/model/" || exit 2
(cd -- "$W/model" && for f in *.gguf; do sha256sum "$f" >"$f.weights-are-public" && chmod go-w "$f.weights-are-public" || exit 1; done) || exit 2
head -c 32 /dev/urandom >"$W/public.32" || exit 2 # the option's file. This test treats it as public; no program can check that it is
python3 -I -B "$here/../tools/make_request.py" "$W/config/policy.txt" "$W/in/example.req" || exit 2 # synthetic tokens; the proof is about the next token only
# The nine calls. In 1 and 2, and in 4 and 5, only the option and the output directory differ; in
# 6 and 7, one digit of the ID.
# 1. cp_commit without the option, so under a seed that it draws. `-M 1`: one leaf table.
# 2. cp_commit with the option.
# 3. Call 2 again, into another directory, before cp_prove changes a file of the first: what a
#    record publishes follows, like every other byte, from the weights, the configuration and the
#    option's file.
# 4. cp_prove on call 2's commitment without the option: stopped at isk.txt's line `seed public`.
# 5. cp_prove with the option.
# 6. cp_verify on public files only, under the commitment ID. The two lines that end "the table has
#    a record" are not printed from the key file or the leaf file:
#    that these have a record is shown only by the YES.
# 7. Call 6 under the commitment ID with its first digit changed.
# 8. cp_prove on call 1's commitment, without the option.
# 9. cp_verify on call 1's public files and call 8's response, under call 1's commitment ID.
commit() { # <log> <directory for ivk and isk> [option and its file]
  local log=$1 at=$2; shift 2
  # shellcheck disable=SC2086 # GPU_RUN is a list of words, here and below
  ${GPU_RUN:-} "$B/cp_commit" -weights "$W/model" -config "$W/config" -M 1 -ivk "$at/ivk" -isk "$at/isk" "$@" >"$W/$log" 2>&1
}
mkdir -m 700 "$W/without" "$W/with" "$W/again" || exit 2 # cp_prove rejects a path to isk through a directory that its group or others can write
commit 1.cp_commit.log "$W/without"; rc=$?
drawn=$(sed -n "s/^commitment ID: //p" "$W/1.cp_commit.log" | tr -d ' ')
echo "1. cp_commit without -public-weights: status $rc; commitment ID: ${drawn:-none}"
[ $rc = 0 ] && [[ $drawn =~ ^[0-9a-f]{64}$ ]] || { no "1: status $rc, or no commitment ID: $(tail -n 2 "$W/1.cp_commit.log" | cut -c1-300 | tr '\n' '|')"; end; }
lacks 1.cp_commit.log "FOR PUBLIC WEIGHTS"
[ "$(grep -a -c "^cp_commit: note (record): for each stored table " "$W/1.cp_commit.log")" = 1 ] || no "1: cp_commit's note on a record is not printed once"
has 1.cp_commit.log "^cp_commit: done\. .* which is complete and can be handed out\. It holds a record (see the note above)\. "
grep -q "^seed " "$W/without/isk/isk.txt" && no "1: isk.txt has a line that begins \"seed \", which cp_commit writes only if it has not drawn the seed"
commit 2.cp_commit.log "$W/with" -public-weights "$W/public.32"; rc=$?
id=$(sed -n "s/^commitment ID: //p" "$W/2.cp_commit.log" | tr -d ' ')
echo "2. cp_commit with -public-weights: status $rc; commitment ID: ${id:-none}"
[ $rc = 0 ] && [[ $id =~ ^[0-9a-f]{64}$ ]] || { no "2: status $rc, or no commitment ID: $(tail -n 2 "$W/2.cp_commit.log" | cut -c1-300 | tr '\n' '|')"; end; }
[ "$(grep -a -c "^cp_commit: FOR PUBLIC WEIGHTS: " "$W/2.cp_commit.log")" = 1 ] || no "2: the option's warning is not printed once"
[ "$id" != "$drawn" ] || no "2: the commitment ID is call 1's, though the two seeds differ"
lacks 2.cp_commit.log "note (record)"; lacks 2.cp_commit.log "It holds a record"
has 2.cp_commit.log "^cp_commit: done\. .* which is complete, and hides nothing: it was made with -public-weights "
[ "$(tail -n 1 "$W/with/isk/isk.txt")" = "seed public" ] || no "2: isk.txt does not end with the line \"seed public\""
commit 3.cp_commit.log "$W/again" -public-weights "$W/public.32"; rc=$?
differ=$(diff -rq "$W/with" "$W/again" 2>&1 | wc -l)
echo "3. the same call again: status $rc; files of the two key directories that differ: $differ"
[ $rc = 0 ] && [ "$(sed -n "s/^commitment ID: //p" "$W/3.cp_commit.log" | tr -d ' ')" = "$id" ] || no "3: status $rc, or another commitment ID"
[ "$differ" = 0 ] && [ -s "$W/again/ivk/key.bin" ] || no "3: the two commitments' files differ: $(diff -rq "$W/with" "$W/again" 2>&1 | head -n 3 | cut -c1-200 | tr '\n' '|')"
prove() { # <log> <directory for the response> <secret key directory> [option and its file]
  local log=$1 out=$2 isk=$3; shift 3
  # shellcheck disable=SC2086
  ${GPU_RUN:-} "$B/cp_prove" -isk "$isk" -in "$W/in" -out "$out" "$@" >"$W/$log" 2>&1
}
prove 4.cp_prove.log "$W/out.without" "$W/with/isk"; rc=$?
left=$(find "$W/out.without" -type f ! -name '*.failed' 2>/dev/null | tr '\n' ' ')
echo "4. cp_prove without -public-weights: status $rc; files of a response: ${left:-none}"
[ $rc != 0 ] && [ -z "$left" ] || echo "WAS NOT REFUSED: cp_prove, a commitment to public weights without -public-weights"
[ $rc = 1 ] && [ -z "$left" ] || no "4: status $rc, not 1, or a response"
has 4.cp_prove.log "refused (cp-isk-seed-public): isk.txt says \"seed public\"\. "
[ "$(tail -n 1 "$W/4.cp_prove.log")" = "p258 end 1 cp-isk-seed-public" ] || no "4: the last line is not \"p258 end 1 cp-isk-seed-public\""
prove 5.cp_prove.log "$W/out" "$W/with/isk" -public-weights "$W/public.32"; rc=$?
made=$(ls "$W/out" 2>/dev/null | tr '\n' ' ')
echo "5. cp_prove with -public-weights: status $rc; in its output directory: ${made:-nothing}"
[ $rc = 0 ] && [ -s "$W/out/example.proof" ] && [ -s "$W/out/example.y" ] || { no "5: status $rc, or no proof or no output token: $(tail -n 2 "$W/5.cp_prove.log" | cut -c1-300 | tr '\n' '|')"; end; }
[ "$(grep -a -c "^p258 FOR PUBLIC WEIGHTS: " "$W/5.cp_prove.log")" = 1 ] || no "5: the option's warning is not printed once"
"$B/cp_verify" "$W/config" "$W/with/ivk" "$W/in/example.req" "$W/out/example.y" "$W/out/example.proof" -id "$id" >"$W/6.cp_verify.log" 2>&1; rc=$?
echo "6. cp_verify: status $rc; last line: $(tail -n 1 "$W/6.cp_verify.log" | cut -c1-100)"
grep -a "^accounting: one-time files: " "$W/6.cp_verify.log"
[ $rc = 0 ] && [ "$(tail -n 1 "$W/6.cp_verify.log")" = YES ] || no "6: status $rc, not 0, or the last line is not YES"
says_the_record() { # <call's number>: cp_verify's log has one line on the key file and one on the leaf file that end "the table has a record"
  local file
  for file in "a key" "an entry"; do
    [ "$(grep -a -c "^accounting: one-time files: the plan of $file: .*; the table has a record\$" "$W/$1.cp_verify.log")" = 1 ] || no "$1: cp_verify did not print exactly one line on $file that ends \"the table has a record\""
  done
}
says_the_record 6
case $id in 0*) other=1${id#?} ;; *) other=0${id#?} ;; esac
"$B/cp_verify" "$W/config" "$W/with/ivk" "$W/in/example.req" "$W/out/example.y" "$W/out/example.proof" -id "$other" >"$W/7.cp_verify.log" 2>&1; rc=$?
echo "7. cp_verify under another ID: status $rc; last line: $(tail -n 1 "$W/7.cp_verify.log" | cut -c1-100)"
# If no test fails, a call on a request's files ends with status 0, or with 14 (NOT-A-VERIFICATION).
case $rc in 0 | 14) accepted=1 ;; esac; grep -a -q "^YES" "$W/7.cp_verify.log" && accepted=1
[ $accepted = 0 ] || echo "WAS NOT REFUSED: cp_verify passed a response under another commitment ID: status $rc; the files are kept in $W"
[ $rc = 1 ] && [ "$(tail -n 1 "$W/7.cp_verify.log")" = "NO: model-statement" ] || no "7: status $rc, not 1, or the last line is not \"NO: model-statement\""
prove 8.cp_prove.log "$W/out.drawn" "$W/without/isk"; rc=$?
made=$(ls "$W/out.drawn" 2>/dev/null | tr '\n' ' ')
echo "8. cp_prove on call 1's commitment: status $rc; in its output directory: ${made:-nothing}"
[ $rc = 0 ] && [ -s "$W/out.drawn/example.proof" ] && [ -s "$W/out.drawn/example.y" ] || { no "8: status $rc, or no proof or no output token: $(tail -n 2 "$W/8.cp_prove.log" | cut -c1-300 | tr '\n' '|')"; end; }
lacks 8.cp_prove.log "FOR PUBLIC WEIGHTS"
"$B/cp_verify" "$W/config" "$W/without/ivk" "$W/in/example.req" "$W/out.drawn/example.y" "$W/out.drawn/example.proof" -id "$drawn" >"$W/9.cp_verify.log" 2>&1; rc=$?
echo "9. cp_verify on call 1's commitment: status $rc; last line: $(tail -n 1 "$W/9.cp_verify.log" | cut -c1-100)"
[ $rc = 0 ] && [ "$(tail -n 1 "$W/9.cp_verify.log")" = YES ] || no "9: status $rc, not 0, or the last line is not YES"
says_the_record 9
end
