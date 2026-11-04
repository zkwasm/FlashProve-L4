#!/bin/bash
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.

# Tests what the prover program cp_prove rejects at its start, or once its first batch has begun, on
# a commitment made without -test-seed: an altered secret key directory (`isk`), and the mistakes
# of a first run, such as a wrong path. Each must get the expected message or rejection name. A case
# (`scene`) is one call of cp_prove. On a commitment made with -test-seed a test build does not ask
# for isk.txt's line `ranges`, so such a commitment cannot show these rejections.
# usage: refused_at_the_start.sh <programs> <configuration> <work directory> [<record file>]
#   <programs>: a directory with cp_commit (the committer), the cp_verify that it needs beside it and
#   cp_prove, of one build, as an absolute path: the case `isk-relative` calls cp_prove from another
#   working directory.
#   <configuration> (`kit`): a directory with policy.txt, KL.bin and layout.txt whose
#   scales pass cp_commit's worst-case check, such as prover/examples/target_small_f2_roomy.
#   <work directory> (messages say `folder` for a directory): must not exist; created here with
#   mode 700, with any missing directories above it (these under your umask). Its absolute path may
#   have only letters, digits and _ . / - and no part that is ".." or begins with "-". The script
#   itself deletes nothing.
#   <record file>, optional, adds the cases `answers-two`, `record-made-up` and `record-public`: a
#   parameter file with the line "record 1" and the line "answers 1" (else status 2) and other bytes
#   than <configuration>'s policy.txt: with prover/examples/target_small_f2_roomy,
#   prover/examples/target_small_f2's.
# Started by the runner; by hand: below a directory of your own, since the script does not check
# who can write to the directories above the work directory. (Runner: run_suite.py.) Name the work
# directory without symbolic links: cp_prove rejects an `isk` whose path has one, and the test
# would stop with status 2.
# GPU_RUN, if set, is a command prefix for cp_commit and cp_prove. The test needs one H100 that
# nothing else uses and takes about two minutes.
# Last line: "refused_at_the_start: <n> scenes, all as expected", status 0, or
# "refused_at_the_start: FAILED; the screens are in <work directory>", status 1. Status 2: the
# set-up or `control` (the call on the unaltered `isk`) failed. A missing argument gives bash's own
# message and status.
set -u
unset CDPATH
made_here() { # <a folder that this script has just made>: it must be this account's, closed to others (mode 700; 2700 below a set-group-ID directory) and empty
  # One cd, then three tests of ".", so that all three look at the same directory. If stat or ls
  # fails, the directory is rejected too. With -P, cd resolves ".." as mkdir does: after symbolic
  # links, not by the path's text. The slash keeps cd from reading "-" as the previous directory.
  (cd -P -- "$1/" && [ -O . ] && [[ $(stat -c %a .) == *700 ]] && list=$(ls -A) && [ -z "$list" ]) && return
  echo "refused_at_the_start.sh: $1 is not the folder that was just made: another was put in its place: name a new folder below folders in which only you can write" >&2
  return 1
}
B=${1:?the programs\' folder} K=${2:?a kit} W=${3:?a work folder} R=${4:-}
here=$(cd -- "$(dirname "$0")" && pwd) || exit 2
case $W in /*) full=$W ;; *) full=$PWD/$W ;; esac; case /$full/ in */-*|*/../*|*[!A-Za-z0-9_./-]*) echo "refused_at_the_start: the work folder's path may hold letters, digits and _ . / - only, and no part of it may be \"..\" or begin with \"-\": name another: $full"; exit 2 ;; esac # the path goes into sed commands below
if [ -e "$W" ]; then echo "refused_at_the_start: the work folder $W exists: name a new one"; exit 2; fi
for f in "$B/cp_commit" "$B/cp_prove" "$B/cp_verify" "$K/policy.txt" "$K/KL.bin" "$K/layout.txt"; do [ -f "$f" ] || { echo "refused_at_the_start: there is no $f"; exit 2; }; done
if [ -n "$R" ] && ! grep -q '^record 1$' "$R"; then echo "refused_at_the_start: $R cannot be read or has no line «record 1»"; exit 2; fi
mkdir -p -- "$(dirname -- "$W")" && mkdir -m 700 -- "$W" && made_here "$W" && mkdir -p "$W/in" "$W/home" "$W/tmp" || exit 2 # made new here: the look above is no guard against a folder made since
full=$(cd -- "$W" && pwd) && [ "$full" -ef "$W" ] && W=$full || exit 2 # -ef: stop unless cd reached the directory just made (a lone "-", or ".." after a symbolic link, can lead cd elsewhere)
export HOME=$W/home TMPDIR=$W/tmp # the closed front wants both to name folders that stand
read -r -a gpu_run <<<"${GPU_RUN:-}" # after TMPDIR is set: bash may write a here-string to a file in TMPDIR
bad=0 played=0
wrong() { echo "   NOT AS EXPECTED: $*"; bad=1; }
closed=no; "$B/cp_prove" --setting 2>&1 | grep -q 'a serving build' && closed=yes
early=2; [ $closed = yes ] && early=1 # a refusal where isk.txt is read stands before the program has armed its own endings: status 2; behind a closed front every refusal ends with 1
echo "the build's front is closed: $closed"
python3 -B "$here/../tools/make_request.py" "$K/policy.txt" "$W/in/example.req" || exit 2
for c in isk other; do
  ${gpu_run[@]+"${gpu_run[@]}"} "$B/cp_commit" -weights made-up-blocks -config "$K" -M 1 -ivk "$W/$c.ivk" -isk "$W/$c" >"$W/$c.commit.log" 2>&1 || { echo "refused_at_the_start: cp_commit failed: $W/$c.commit.log"; exit 2; }
done
[ -s "$W/isk/seed.bin" ] && [ -s "$W/other/seed.bin" ] && ! cmp -s "$W/isk/seed.bin" "$W/other/seed.bin" || { echo "refused_at_the_start: the two commitments' seeds are missing or the same"; exit 2; }
grep -q '^ranges [0-9a-f]' "$W/isk/isk.txt" && ! grep -q '^seed test$' "$W/isk/isk.txt" || { echo "refused_at_the_start: isk.txt is not spelled as this script expects"; exit 2; }
files() { find "$1" -type f 2>/dev/null | sed "s#^$1/##" | LC_ALL=C sort | tr '\n' ' '; }
# $closed, set above: yes if cp_prove is a production build, which restricts its output.
# $early, set above: the expected status of a rejection in read_isk() (prover/prover/isk.cuh): 2 in
# a test build, because read_isk() runs before arm_ends() (prover/prover/ends.h) installs the exit
# handlers. Every other rejection tested here, and every one in a production build, has status 1.
# judged <case> <status> <name> <where> <stands> [<end name>]: checks the exit status in $got and
# the output (stdout and stderr) in $W/<case>.screen.
# A test build reports a rejection as <where> says. `refused`: a line "refused (<name>)". `ended`:
# the last line "ended: .. by <name>". `line`: some line that contains the words <name>.
# The production build ($closed = yes; "a serving build" in its own output) restricts its output
# (its "closed front": prover/prover/closed_front.h). Its last line must be
# "p258 end <status> <name>", with <end name> as the name if <where> is `line`. Just before it reads
# the seed it prints "p258 closed: ..".
# <stands> = `open`: the rejection comes before that line, which must be absent, and the message
# must be printed as in a test build. <stands> = `behind`: it comes after that line, which must
# occur once, and from there on every line must match prover/tests/closed_front_lines.txt: the
# name is printed, but no message. Each line of that file is an extended regular expression for one
# whole line of output.
judged() {
  local n=$1 status=$2 name=$3 where=$4 stands=$5 last closes
  last=$(tail -n 1 "$W/$n.screen" | cut -c1-100)
  [ "$got" = "$status" ] || wrong "$n: wanted status $status"
  if [ $closed = yes ]; then
    closes=$(grep -a -c '^p258 closed: ' "$W/$n.screen")
    if [ "$where" = line ]; then grep -a -q -F -- "$name" "$W/$n.screen" || wrong "$n: wanted a line with the words «$name»"; name=$6; fi
    [ "$last" = "p258 end $status $name" ] || wrong "$n: wanted the last line «p258 end $status $name»"
    if [ "$stands" = open ]; then
      [ "$closes" = 0 ] || wrong "$n: the front has closed before the refusal"
      [ "$where" = line ] || grep -a -q "^\(cp_prove: \)\?refused ($name)" "$W/$n.screen" || wrong "$n: wanted the refusal's sentence on the screen: refused ($name)"
    else
      [ "$closes" = 1 ] && [ "$(sed -n '/^p258 closed: /,$p' "$W/$n.screen" | grep -a -c -v -x -E -f "$here/closed_front_lines.txt")" = 0 ] || wrong "$n: wanted the front to close once, and from then on lines of its list alone"
    fi
  else
    case $where in
    refused) grep -a -q "^\(cp_prove: \)\?refused ($name)" "$W/$n.screen" || wrong "$n: wanted the refusal by its name: $name" ;;
    ended) case $last in "ended: "*", by $name") ;; *) wrong "$n: wanted a last line «ended: .. by $name»" ;; esac ;;
    line) grep -a -q -F -- "$name" "$W/$n.screen" || wrong "$n: wanted a line with the words «$name»" ;;
    esac
  fi
}
# scene <case> <status> <name> <where> <stands> <out> <count> [<option> ..]: runs cp_prove on
# $W/s.<case>, which the caller has made with lay(). The first five arguments are judged()'s; its
# <end name> is `usage`.
#   <out>    the files expected in -out, each name followed by a space
#   <count>  stays | moves: whether leaf/0/count.bin must be unchanged or changed after the call
scene() {
  local n=$1 status=$2 name=$3 where=$4 stands=$5 out=$6 count=$7 before last moved=stays; shift 7
  played=$((played + 1)); before=$(sha256sum "$W/s.$n/leaf/0/count.bin" | cut -c1-16)
  ${gpu_run[@]+"${gpu_run[@]}"} "$B/cp_prove" -isk "$W/s.$n" -in "$W/in" -out "$W/out.$n" "$@" >"$W/$n.screen" 2>&1; got=$?
  last=$(tail -n 1 "$W/$n.screen" | cut -c1-100)
  [ "$(sha256sum "$W/s.$n/leaf/0/count.bin" | cut -c1-16)" = "$before" ] || moved=moves
  echo "[$n] status $got; in -out: $(files "$W/out.$n"); the count $moved; refusals named: $(grep -a -o '^\(cp_prove: \)\?refused ([a-z-]*)' "$W/$n.screen" | sed 's/.*(//; s/)//' | tr '\n' ' '); the last line: $last"
  [ "$(files "$W/out.$n")" = "$out" ] || wrong "$n: wanted «$out» in -out"
  [ "$moved" = "$count" ] || wrong "$n: wanted: the count $count"
  judged "$n" "$status" "$name" "$where" "$stands" usage
}
# Cases on a copy of `isk` (the lines `lay <case>; ..` below `control`), each with one thing changed
# in the copy or in the call; `other` is a second commitment under the same configuration. In every
# case of this script but `control` and `audit-empty`, -out must hold no proof file and no token
# file (.y): nothing, or only a `.failed` file.
# These rejections protect hiding and what a proof is bound to:
#   ranges-*: isk.txt's line `ranges` states that cp_commit's worst-case check passed for these
#     configuration files. Without that check, a prompt might drive a value of the forward pass out
#     of its range. Whether it does depends on the weights, and the request's `.failed` file would
#     tell that.
#   config-changed-*: one byte of KL.bin, of layout.txt or of policy.txt is changed, so the line
#     `ranges` does not hold the digest of the configuration files: the run ends before the spool
#     is opened. No key protects that line: this check does not find a writer who changes it too.
#     layout.txt is read whole here and parsed later, so any byte will do; a changed policy.txt must
#     still pass the parameter file's reader, so one digit of a number of proof of work changes.
#   seed-test-added: anyone who can write into `isk` can add the line "seed test". A production
#     build must refuse a commitment with that line. A test build serves it, so the case is played
#     on a production build alone.
#   audit-owed: cp_commit -keep-for-the-audit stores random bytes in `isk`/audit for the audit of
#     the commitment (tools/audit/README.md). With the seed, they open every mask in key.bin and
#     the leaf files (entry.bin). The audit removes them once it has passed in a run without
#     --rehearsal. While anything is in that directory, cp_prove must make no proof.
#   audit-empty: the directory is empty, as that run of the audit leaves it. This is a second
#     control: status 0, a proof file and a token file.
#   other-seed, other-entry, fresh-short: a seed, a leaf file (entry.bin) or a fresh.bin (the 32
#     bytes that cp_commit drew for the leaf table) that is not this commitment's must let nothing
#     out. The last two files belong to the leaf table (leaf/0), the committed weight table against
#     which the batch is proved.
#   answers-two, record-made-up, record-public: <record file> replaces config/policy.txt in the
#     copy.
#     answers-two: the line `answers` says 2: the parameter file's reader rejects it, in both builds.
#     record-made-up: isk.txt is unchanged. The run prints the line on the conditions for hiding, and
#     is then rejected as ranges-not-guarded: the line `ranges` holds the digest of other
#     configuration files.
#     record-public: a directory with an empty .gguf file as isk.txt's line `weights`, "seed public"
#     in isk.txt and the option -public-weights. The production build prints that the seed is public
#     and ends like record-made-up, before it reads seed.bin, which here is not the seed derived from
#     the option's file. A test build does not know the option.
# other-seed, other-entry and fresh-short are rejected only after the batch has begun. By then
# cp_prove has raised the leaf table's column count (count.bin: the columns shown by all openings
# begun so far), which a batch does before it reads anything else of the leaf table, and the request
# gets a `.failed` file.
# hc-own-node: a tree node recomputed under this seed differs from the stored one. hc-nodes-root:
# the stored nodes do not hash to the root in this leaf file.
# seed-open: seed.bin must have no permission for group or others; only group read is tried.
# entry-missing: read_isk() requires every live leaf table's hcnodes.bin, fresh.bin and entry.bin
# as readable files; only a missing entry.bin is tried.
# The expected statuses and count changes of the cases on a copy of `isk` made by lay(), the two
# audit-*, config-changed-*, answers-two, record-made-up and record-public excepted, were taken from
# a run of both builds, not from the sources alone.
# Not covered by this test: HOME or TMPDIR unset, which the production build rejects
# (prover/tests/serving_first_run.sh tests that); a parameter file that misses a condition for
# hiding; a GPU that is in use; a service on a spool directory; a real model's files
# (record-public has an empty stand-in); cp_commit's own rejections
# (prover/hash_commitment/tests/cp_commit_small.sh); full size.
# slip <case> <status> <name> <where> <end name> <program> [<argument> ..]: a mistake of a first
# run. Runs <program> with the arguments; slip() itself makes no copy of `isk`. Expects no file in
# $W/out.<case>, an unchanged $W/isk/leaf/0/count.bin, and a rejection before "p258 closed"
# (`open`). <where> is refused | line.
slip() {
  local n=$1 status=$2 name=$3 where=$4 ends=$5 program=$6 before; shift 6
  played=$((played + 1)); before=$(sha256sum "$W/isk/leaf/0/count.bin" | cut -c1-16)
  ${gpu_run[@]+"${gpu_run[@]}"} "$program" "$@" >"$W/$n.screen" 2>&1; got=$?
  echo "[$n] status $got; the last line: $(tail -n 1 "$W/$n.screen" | cut -c1-100)"
  [ "$(sha256sum "$W/isk/leaf/0/count.bin" | cut -c1-16)" = "$before" ] || wrong "$n: the count has moved"
  [ -z "$(files "$W/out.$n")" ] || wrong "$n: a file lies in -out"
  judged "$n" "$status" "$name" "$where" open "$ends"
}
lay() { cp -a "$W/isk" "$W/s.$1" || exit 2; }
${gpu_run[@]+"${gpu_run[@]}"} "$B/cp_prove" -isk "$W/isk" -in "$W/in" -out "$W/out.control" >"$W/control.screen" 2>&1; got=$?
echo "[control] status $got; in -out: $(files "$W/out.control")"
[ "$got" = 0 ] && [ "$(files "$W/out.control")" = "example.proof example.y " ] || { echo "refused_at_the_start: the control failed: $W/control.screen"; exit 2; }
lay ranges-missing; sed -i '/^ranges /d' "$W/s.ranges-missing/isk.txt"; scene ranges-missing 1 ranges-not-guarded refused open "" stays
lay ranges-bent; python3 -I -B - "$W/isk/isk.txt" "$W/s.ranges-bent/isk.txt" <<'END'
import sys
lines = open(sys.argv[1]).read().split("\n")
k = [i for i, l in enumerate(lines) if l.startswith("ranges ")][0]
word = lines[k].split()[1]; lines[k] = "ranges " + ("0" if word[0] != "0" else "1") + word[1:]
open(sys.argv[2], "w").write("\n".join(lines))
END
scene ranges-bent 1 ranges-not-guarded refused open "" stays
flipped() { python3 -I -B -c 'import sys; p, at = sys.argv[1], int(sys.argv[2]); b = bytearray(open(p, "rb").read()); b[at] ^= 1; open(p, "wb").write(b)' "$1" "$2"; } # <file> <offset, from the end if negative>
lay config-changed-kl; flipped "$W/s.config-changed-kl/config/KL.bin" -4; scene config-changed-kl 1 ranges-not-guarded refused open "" stays # the lowest bit of the last 32-bit word: a shift of 15 becomes 14 (a byte higher up gives an unlawful constant, which the constants' reader refuses first)
lay config-changed-layout; flipped "$W/s.config-changed-layout/config/layout.txt" 2; scene config-changed-layout 1 ranges-not-guarded refused open "" stays
lay config-changed-policy; sed -i 's/^work positions 20 /work positions 21 /' "$W/s.config-changed-policy/config/policy.txt"; cmp -s "$W/isk/config/policy.txt" "$W/s.config-changed-policy/config/policy.txt" && { echo "refused_at_the_start: the kit's policy.txt has no line «work positions 20 ..»: no byte was changed"; exit 2; }; scene config-changed-policy 1 ranges-not-guarded refused open "" stays
if [ $closed = yes ]; then lay seed-test-added; echo "seed test" >>"$W/s.seed-test-added/isk.txt"; scene seed-test-added 1 cp-isk-seed-test refused open "" stays; fi
lay version-word; awk '$1 == "version" { $2 = $2 + 1 } { print }' "$W/isk/isk.txt" >"$W/s.version-word/isk.txt"; scene version-word "$early" cp-isk refused open "" stays
lay seed-open; chmod 640 "$W/s.seed-open/seed.bin"; scene seed-open 1 hc-key-file refused behind "" stays
lay other-seed; cp -p "$W/other/seed.bin" "$W/s.other-seed/seed.bin"; scene other-seed 1 hc-own-node ended behind "example.failed " moves
lay entry-missing; mv "$W/s.entry-missing/leaf/0/entry.bin" "$W/entry.taken-out"; scene entry-missing "$early" cp-isk-leaves refused open "" stays
lay audit-owed; (umask 077 && mkdir "$W/s.audit-owed/audit" && head -c 32 /dev/urandom >"$W/s.audit-owed/audit/key.fresh") || exit 2; scene audit-owed "$early" cp-kept-for-the-audit refused open "" stays
lay audit-empty; (umask 077 && mkdir "$W/s.audit-empty/audit") || exit 2; played=$((played + 1))
${gpu_run[@]+"${gpu_run[@]}"} "$B/cp_prove" -isk "$W/s.audit-empty" -in "$W/in" -out "$W/out.audit-empty" >"$W/audit-empty.screen" 2>&1; got=$?
echo "[audit-empty] status $got; in -out: $(files "$W/out.audit-empty")"
[ "$got" = 0 ] && [ "$(files "$W/out.audit-empty")" = "example.proof example.y " ] || wrong "audit-empty: wanted status 0, a proof and a reply"
lay fresh-short; head -c 31 "$W/isk/leaf/0/fresh.bin" >"$W/s.fresh-short/leaf/0/fresh.bin"; scene fresh-short 1 cp-leaf-record refused behind "example.failed " moves
lay other-entry; cp -p "$W/other/leaf/0/entry.bin" "$W/s.other-entry/leaf/0/entry.bin"; scene other-entry 1 hc-nodes-root ended behind "example.failed " moves
# The cases on <record file>. $W/model stands for a model's files: cp_prove lists the names in it.
if [ -n "$R" ]; then
  mkdir "$W/model" && : >"$W/model/empty.gguf" || exit 2
  lay answers-two; cp "$R" "$W/s.answers-two/config/policy.txt"; sed -i 's/^answers 1$/answers 2/' "$W/s.answers-two/config/policy.txt"
  grep -q '^answers 2$' "$W/s.answers-two/config/policy.txt" || { echo "refused_at_the_start: the record file is not spelled as this script expects"; exit 2; }
  played=$((played + 1)); ${gpu_run[@]+"${gpu_run[@]}"} "$B/cp_prove" -isk "$W/s.answers-two" -in "$W/in" -out "$W/out.answers-two" >"$W/answers-two.screen" 2>&1; got=$?
  echo "[answers-two] status $got; in -out: $(files "$W/out.answers-two"); the last line: $(tail -n 1 "$W/answers-two.screen" | cut -c1-100)"
  [ "$got" = 1 ] && [ -z "$(files "$W/out.answers-two")" ] && grep -a -q '^policy: parameter-range: .*answers' "$W/answers-two.screen" || wrong "answers-two: wanted status 1, nothing in -out and the reader's line on the parameter's range"
  lay record-made-up; cp "$R" "$W/s.record-made-up/config/policy.txt"; scene record-made-up 1 ranges-not-guarded refused open "" stays
  grep -a -q '^p258 the parameter file m' "$W/record-made-up.screen" || wrong "record-made-up: wanted the line on the conditions for hiding, which comes before the check of the line ranges: the run has ended before it"
  lay record-public; head -c 32 /dev/urandom >"$W/public.32" || exit 2; cp "$R" "$W/s.record-public/config/policy.txt"; sed -i "s#^weights made-up-blocks\$#weights $W/model#" "$W/s.record-public/isk.txt"; echo "seed public" >>"$W/s.record-public/isk.txt"
  grep -q "^weights $W/model\$" "$W/s.record-public/isk.txt" || { echo "refused_at_the_start: isk.txt is not spelled as this script expects"; exit 2; }
  if [ $closed = yes ]; then
    scene record-public 1 ranges-not-guarded refused open "" stays -public-weights "$W/public.32"
    grep -a -q '^p258 FOR PUBLIC WEIGHTS: ' "$W/record-public.screen" && grep -a -q '^p258 the parameter file m' "$W/record-public.screen" || wrong "record-public: wanted the line that the seed is public, and the line on the conditions for hiding"
  else scene record-public 1 "cp_prove: unknown option -public-weights" line open "" stays -public-weights "$W/public.32"; fi
fi
# Mistakes of a first run.
# How -isk may be named. read_isk() opens the path part by part, so that no other account can have
# put another directory under a name on the way:
#   isk-relative: the path must begin with /. A relative path starts at the working directory, and
#     cp_prove cannot tell whether a link led the caller there (cd follows links).
#   isk-link, isk-link-above: no part may be a symbolic link, the last ($W/isk.link) or one above.
#   isk-below-open: in a directory of mode 0777 another account could replace `isk`.
#   isk-open: `isk` itself must have no permission for group or others; only mode 0750 is tried.
# slip() does not look at count.bin in $W/open/isk, the copy of `isk` that isk-below-open and
# isk-open use.
slip no-argument 1 "cp_prove: -isk is required" line usage "$B/cp_prove"
grep -a -q '^usage: cp_prove ' "$W/no-argument.screen" || wrong "no-argument: wanted the usage text"
for h in -h --help; do
  played=$((played + 1)); "$B/cp_prove" $h >"$W/help$h.screen" 2>&1; got=$?
  echo "[help $h] status $got; lines: $(wc -l <"$W/help$h.screen")"
  [ "$got" = 0 ] && grep -a -q '^usage: cp_prove ' "$W/help$h.screen" && ! grep -a -q -E '^(p258 end|ended:) ' "$W/help$h.screen" || wrong "help $h: wanted status 0, the usage text and no last line of an end"
  [ $closed = no ] || grep -a -q -F "HOME AND TMPDIR MUST BE SET" "$W/help$h.screen" || wrong "help $h: wanted the usage text to name HOME and TMPDIR"
done
slip isk-alone 1 "cp_prove: -isk needs a value" line usage "$B/cp_prove" -isk
slip no-such-isk "$early" cp-isk refused - "$B/cp_prove" -isk "$W/none" -in "$W/in" -out "$W/out.no-such-isk"
(cd -- "$W" && slip isk-relative "$early" cp-isk refused - "$B/cp_prove" -isk isk -in "$W/in" -out "$W/out.isk-relative"; echo "$bad $played" >"$W/isk-relative.tally") && read -r bad played <"$W/isk-relative.tally" || exit 2 # (a subshell for the cd: its two counts come back by a file)
ln -s "$W/isk" "$W/isk.link" && ln -s "$W" "$W/up" || exit 2
slip isk-link "$early" cp-isk refused - "$B/cp_prove" -isk "$W/isk.link" -in "$W/in" -out "$W/out.isk-link"
slip isk-link-above "$early" cp-isk refused - "$B/cp_prove" -isk "$W/up/isk" -in "$W/in" -out "$W/out.isk-link-above"
mkdir "$W/open" && cp -a "$W/isk" "$W/open/isk" && chmod 0777 "$W/open" || exit 2
slip isk-below-open "$early" cp-isk refused - "$B/cp_prove" -isk "$W/open/isk" -in "$W/in" -out "$W/out.isk-below-open"
chmod 0700 "$W/open" && chmod 0750 "$W/open/isk" || exit 2
slip isk-open "$early" cp-isk refused - "$B/cp_prove" -isk "$W/open/isk" -in "$W/in" -out "$W/out.isk-open"
slip no-such-in "$early" cp-batch refused - "$B/cp_prove" -isk "$W/isk" -in "$W/none" -out "$W/out.no-such-in"
slip no-such-spool 1 cp-spool refused - "$B/cp_prove" -isk "$W/isk" -spool "$W/none"
grep -a -q -F "the path given with -spool is not there or cannot be opened" "$W/no-such-spool.screen" || wrong "no-such-spool: wanted the sentence on the path given with -spool"
mkdir "$W/out.out-exists" || exit 2
slip out-exists 1 cp-spool refused - "$B/cp_prove" -isk "$W/isk" -in "$W/in" -out "$W/out.out-exists"
grep -a -q -F "the folder named with -out exists already" "$W/out-exists.screen" || wrong "out-exists: wanted the sentence on the folder named with -out"
if [ $bad = 0 ]; then echo "refused_at_the_start: $played scenes, all as expected"; exit 0; fi
echo "refused_at_the_start: FAILED; the screens are in $W"; exit 1
