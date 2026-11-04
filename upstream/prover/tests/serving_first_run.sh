#!/bin/bash
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.

# Runs the production build (`serving` in names and output; prover/build.sh's default target) of the
# programs cp_commit (committer) and cp_prove (prover) on synthetic weights (`made-up-blocks`), with
# seeds that cp_commit draws. It must commit, prove one request, serve six from a spool folder,
# stop before a weight table's limit of opened columns is passed, reject test options and a
# commitment made with -test-seed, and ignore two of the test build's environment variables. The
# verifier program cp_verify must reject an altered leaf file. Most GPU tests run the
# test build (prover/build.sh tests): their results carry over only as far as the two builds are the
# same (the production build is compiled with the macro SERVING_BUILD set: prover/build.sh), and
# only if the production build runs. Which tests of weight hiding ran on a production build:
# docs/how-hiding-is-checked.md. Not tested: other test variables;
# cp_prove's -overlap and -keep-for-the-audit; a model's files (tested by
# prover/tests/one_changed_weight.sh and prover/tests/serving_links.sh); full size.
# usage: serving_first_run.sh <production build: folder with cp_commit, cp_prove, cp_verify>
#        <test build: folder with cp_commit and cp_verify> <work folder, new> <small configuration
#        (`kit`, such as prover/examples/target_small_f2_roomy): policy.txt, KL.bin, layout.txt; its scales must pass cp_commit's worst-case check>
# Needs one H100 that nothing else uses. Takes about four minutes. Status 0, last line
# "serving_first_run: all as expected"; 1, "serving_first_run: FAILED"; 2, the set-up failed (a work
# folder that exists is left untouched; a call without exactly four arguments, none empty, prints the usage above); 130, stopped by Ctrl-C.
# The script creates the work folder (mode 700) and any missing folders above it (these under your
# umask), and does not remove it. The work folder's absolute path may have only letters, digits and
# _ . / - and no part that is ".." or begins with "-" (else status 2). Name it without symbolic
# links and below folders that only you or root can write to (a sticky folder such as /tmp will do):
# cp_prove requires both of the path of a commitment's secret folder (isk; here inside the work
# folder), and the test fails without them.
# Started by the runner; by hand: below a directory of your own: the script does not examine the directories above the work folder; cp_prove does. (Runner: run_suite.py. The work folder goes there.)
set -u
unset CDPATH; trap 'echo "serving_first_run: stopped by Ctrl-C" >&2; exit 130' INT # a production cp_prove catches INT, removes its secrets and ends by itself: without this the run would go on to the next step
made_here() { # <a folder that this script has just made>: it must be this account's, closed to others (mode 700; 2700 below a set-group-ID directory) and empty
  # One cd, then three tests of ".", so that all three see the same folder. If stat or ls fails, the
  # folder is rejected too. With -P, cd resolves symbolic links before "..", as mkdir does. The
  # slash after $1 keeps cd from taking a folder named "-" for the previous directory.
  (cd -P -- "$1/" && [ -O . ] && [[ $(stat -c %a .) == *700 ]] && list=$(ls -A) && [ -z "$list" ]) && return
  echo "serving_first_run.sh: $1 is not the folder that was just made: another was put in its place: name a new folder below folders in which only you can write" >&2
  return 1
}
[ $# = 4 ] && [ -n "$1" ] && [ -n "$2" ] && [ -n "$3" ] && [ -n "$4" ] || { sed -n '19,21s/^# \{0,1\}//p' "$0" >&2; exit 2; }; S=$1 P=$2 W=$3 K=$4; for f in "$S/cp_commit" "$S/cp_prove" "$S/cp_verify" "$P/cp_commit" "$P/cp_verify"; do [ -f "$f" ] && [ -x "$f" ] || { echo "serving_first_run: $f is missing or cannot be run: name the production build's folder first, then the test build's" >&2; exit 2; }; done; for f in policy.txt KL.bin layout.txt; do [ -f "$K/$f" ] || { echo "serving_first_run: the kit $K has no $f" >&2; exit 2; }; done; bears=$(awk '$1 == "table" { print $5 }' "$K/policy.txt"); [[ $bears =~ ^(0|[123456789][0123456789]{0,8})$ ]] || { echo "serving_first_run: the fifth word of the line table in $K/policy.txt must be a whole number from 0 to 999999999, with no leading zero" >&2; exit 2; }
here=$(cd -- "$(dirname "$0")" && pwd) || exit 2
case $W in /*) full=$W ;; *) full=$PWD/$W ;; esac; case /$full/ in */-*|*/../*|*[!A-Za-z0-9_./-]*) echo "serving_first_run: the work folder's path may hold letters, digits and _ . / - only, and no part of it may be \"..\" or begin with \"-\": name another: $full"; exit 2 ;; esac # the path goes into sed commands below
if [ -e "$W" ]; then echo "serving_first_run: the work folder $W exists: name a new one"; exit 2; fi
mkdir -p -- "$(dirname -- "$W")" && mkdir -m 700 -- "$W" && made_here "$W" && mkdir -p -- "$W/config" "$W/in" "$W/home" "$W/tmp" "$W/put" || exit 2 # the mkdir without -p fails if anything has appeared at the name since the -e test on it above
full=$(cd -- "$W" && pwd) && [ "$full" -ef "$W" ] && W=$full || exit 2 # -ef: stop unless cd reached the directory just made (a lone "-", or ".." after a symbolic link, can lead cd elsewhere)
bad=0
wrong() { echo "   NOT AS EXPECTED: $*"; bad=1; }
cp "$K/policy.txt" "$K/KL.bin" "$K/layout.txt" "$W/config/" || exit 2
python3 -B "$here/../tools/make_request.py" "$W/config/policy.txt" "$W/in/example.req" || exit 2
(umask 077 && head -c 32 /dev/urandom >"$W/seed.test" && head -c 32 /dev/urandom >"$W/coins.test")
export HOME=$W/home TMPDIR=$W/tmp
files() { find "$1" -type f 2>/dev/null | sed "s#^$1/##" | LC_ALL=C sort | tr '\n' ' '; }
count() { stat -c %s "$W/isk/leaf/0/count.bin" 2>/dev/null; }
# From its line "p258 closed: the program reads its secrets now. .." on, cp_prove must print only
# lines of a fixed list (its "closed front": prover/prover/closed_front.h).
# prover/tests/closed_front_lines.txt has a pattern for each. Earlier lines are not compared with
# the list.
# listed <screen: a file with cp_prove's output>: prints how many lines begin "p258 closed: ", and
# how many lines, from the first of these on, no pattern matches whole.
listed() { echo "$(grep -a -c '^p258 closed: ' "$1") $(sed -n '/^p258 closed: /,$p' "$1" | grep -a -c -v -x -E -f "$here/closed_front_lines.txt")"; }
says() { grep -a -q -F -- "$2" "$W/$1.screen" || wrong "$1: wanted a line with the words «$2»"; }
# The steps that follow prove(), by the label that each prints. isk is a commitment's secret folder,
# ivk its public one.
# First, --setting. Only the programs' own output says that they are a production build:
# prover/build.sh's records are not read.
# [commit] cp_commit's worst-case check asks whether a prompt can drive a value out of its range at
# the sites that it bounds. isk.txt gets its line "ranges" only when the check has passed; a line
# "seed test" would mark a test seed.
# [kept] cp_commit -keep-for-the-audit also writes isk/audit/key.fresh and entry.0.fresh, the fresh
# random bytes (32 each) used to create the key file (key.bin) and the leaf file (leaf/0.bin;
# `entry` in file names), for their owner only. The audit (tools/audit/audit_commitment.sh) needs
# them. With the option, cp_commit must print that ivk is handed out only when the audit has said
# yes (docs/operators-duties.md, "The audit with the key"). Only this step's last three checks would
# catch a program that used zeros as fresh bytes or a fixed seed.
# [kept-owed] cp_prove must reject ("refused" in the output) a secret folder while anything is in
# isk/audit. The run must not lengthen count.bin, which grows with every batch that is begun: the
# audit says NO if count.bin shows that a prover has run. [kept-audited] On a copy whose isk/audit
# is empty, as the audit leaves it at its YES, cp_prove must prove. cp_prove sees only that the
# folder is empty, not that the audit has said YES.
# [front-..], [options..] More calls that must be rejected: cp_prove without TMPDIR or HOME; both
# programs with test options; cp_prove -measure. Nothing may be written to -out, and count.bin must
# keep its length. prover/tests/refused_at_the_start.sh tests more mistakes at cp_prove's start.
# [prove] `one`: the columns that one batch opens.
# [verify] The call that a client makes. A yes has status 0 whether or not it counts proof of work.
# [variables] Two environment variables that only a test build reads must change nothing. The second
# names a folder to write into. No other test variable is tried: prover/build.sh checks that a
# production program contains no line of prover/tests/serving_build_words.txt. Only this step's last
# check would catch a cp_prove that used the same random values (`coins` in names) in every run.
# [copy spent] The length of count.bin is the number of columns of the openings begun on
# leaf table 0, the leaf file's weight table (a "stored copy" in the output), its LINK proof (last
# step) included; defined in docs/operators-duties.md, "Every day".
# The table's limit is read from policy.txt: the fourth number of the line `table`, the number of
# masks (random entries) in each of the table's rows. On a copy of isk whose count leaves room for
# one column fewer than a batch opens, cp_prove must begin no batch and leave the count as it is.
# [spool] cp_prove as a service (-spool: requests in in/, responses in out/) proves six requests,
# submitted before it starts, in three batches of two, and exits (-end-when-empty). Like
# example.req, each asks about the next token only (format: prover/tools/make_request.py).
# [seed-test] cp_prove must reject a commitment that the test build made with -test-seed.
# [bent] Last step. isk/leaf/0/entry.bin is the secret folder's copy of the leaf file. Its last byte
# belongs to the LINK proof, which ties leaf table 0 to the commitment's first weight table (the
# anchor). cp_prove reads only the file's first part: with one bit of that byte flipped it ends with
# status 0, and cp_verify accepts the response under the published leaf file. Given the altered file
# as the leaf file, cp_verify checks the LINK proof and must reject the same response.
#
# prove <label> <expected status> <expected files in -out> <expected last line> <isk> <arguments>;
# the last line is "p258 end <status>", then the cause's name if one was set
# (prover/prover/ends.h); cp_prove inherits the caller's environment variables
prove() {
  local n=$1 status=$2 out=$3 last=$4 isk=$5 got; shift 5
  "$S/cp_prove" -isk "$isk" -in "$W/in" -out "$W/out.$n" "$@" >"$W/$n.screen" 2>&1; got=$?
  echo "[$n] cp_prove status $got; in -out: $(files "$W/out.$n"); the last line: $(tail -n 1 "$W/$n.screen" | cut -c1-100)"
  [ "$got" = "$status" ] && [ "$(files "$W/out.$n")" = "$out" ] && [ "$(tail -n 1 "$W/$n.screen")" = "$last" ] || wrong "$n: wanted status $status, «$out» in -out, the last line «$last»"
}
for p in cp_commit cp_prove; do "$S/$p" --setting 2>&1 | grep -q 'a serving build' || wrong "settings: $p does not call itself a serving build"; done
"$S/cp_commit" -weights made-up-blocks -config "$W/config" -M 1 -ivk "$W/ivk" -isk "$W/isk" >"$W/commit.log" 2>&1; got=$?
echo "[commit] cp_commit status $got"
[ "$got" = 0 ] && grep -a -q 'check: every bounded site fits' "$W/commit.log" && grep -a -q 'which is complete and can be handed out\. ' "$W/commit.log" || wrong "commit: status, the check's line or the last line"
[ "$(grep -c '^ranges ' "$W/isk/isk.txt" 2>/dev/null)" = 1 ] && ! grep -q '^seed test$' "$W/isk/isk.txt" || wrong "commit: isk.txt: wanted a line ranges and no line «seed test»"
id=$(sed -n "s/^commitment ID: //p" "$W/commit.log" | tr -d ' ')
[ ${#id} = 64 ] || wrong "commit: the log names no ID"
"$S/cp_commit" -weights made-up-blocks -config "$W/config" -M 1 -ivk "$W/kept.ivk" -isk "$W/kept.isk" -keep-for-the-audit >"$W/kept.log" 2>&1; got=$?
echo "[kept] cp_commit status $got; isk/audit: $(cd -- "$W/kept.isk/audit" 2>/dev/null && stat -c '%n %a %s %h' -- * | LC_ALL=C sort | tr '\n' ';')"
[ "$got" = 0 ] && [ "$(cd -- "$W/kept.isk/audit" 2>/dev/null && stat -c '%n %a %s %h' -- * | LC_ALL=C sort | tr '\n' ';')" = "entry.0.fresh 600 32 1;key.fresh 600 32 1;" ] || wrong "kept: status, or what isk/audit holds"
grep -a -q 'which is complete, AND IS HANDED OUT ONLY WHEN THE AUDIT WITH THE KEY HAS SAID YES: ' "$W/kept.log" && ! grep -a -q 'can be handed out' "$W/kept.log" || wrong "kept: the last line"
for f in ivk/key.bin ivk/leaf/0.bin isk/seed.bin; do [ -s "$W/$f" ] && [ -s "$W/kept.$f" ] && ! cmp -s "$W/$f" "$W/kept.$f" || wrong "kept: $f is missing, or has the same bytes in the two commitments"; done
for f in key.fresh entry.0.fresh; do ! cmp -s "$W/kept.isk/audit/$f" <(head -c 32 /dev/zero) || wrong "kept: $f holds zeros"; done
! cmp -s "$W/kept.isk/audit/key.fresh" "$W/kept.isk/audit/entry.0.fresh" || wrong "kept: the two kept files have the same bytes"
kept_count=$(stat -c %s "$W/kept.isk/leaf/0/count.bin")
prove kept-owed 1 "" "p258 end 1 cp-kept-for-the-audit" "$W/kept.isk"; says kept-owed "refused (cp-kept-for-the-audit): "
[ -n "$kept_count" ] && [ "$(stat -c %s "$W/kept.isk/leaf/0/count.bin")" = "$kept_count" ] || wrong "kept, owed: the count of the kept commitment has moved in a run that was refused: the audit would say NO of it"
cp -a "$W/kept.isk" "$W/kept.audited.isk" && rm -f -- "$W/kept.audited.isk/audit/key.fresh" "$W/kept.audited.isk/audit/entry.0.fresh" && [ -d "$W/kept.audited.isk/audit" ] || wrong "kept, owed: the copy without the kept files cannot be laid"
prove kept-audited 0 "example.proof example.y " "p258 end 0" "$W/kept.audited.isk"
begun=$(count)
for v in TMPDIR HOME; do ( unset $v; prove front-$v 1 "" "p258 end 1 cp-prove-environment" "$W/isk"; says front-$v "cp_prove: refused (cp-prove-environment): $v: must be set and name a folder that exists"; exit $bad ) || bad=1; done
for o in "-test-seed $W/seed.test" "-test-bits 0"; do
  # shellcheck disable=SC2086 # an option and its value
  "$S/cp_commit" -weights made-up-blocks -config "$W/config" -M 1 -ivk "$W/ivk.o" -isk "$W/isk.o" $o >"$W/o.log" 2>&1; got=$?
  echo "[options] cp_commit ${o%% *}: status $got"
  [ "$got" = 2 ] && grep -a -q '^usage: cp_commit' "$W/o.log" && [ -z "$(ls -d "$W/ivk.o" "$W/isk.o" "$W/ivk.o.making" "$W/isk.o.making" 2>/dev/null)" ] || wrong "options: cp_commit ${o%% *}: wanted the usage text, status 2, no folder"
done
prove options-coins 1 "" "p258 end 1 usage" "$W/isk" -test-coins "$W/coins.test"; says options-coins "cp_prove: unknown option -test-coins"
prove options-measure 1 "" "p258 end 1 usage" "$W/isk" -measure; says options-measure "refused (serving-build)"
for n in coins measure; do says options-$n "usage: cp_prove "; says options-$n "HOME AND TMPDIR MUST BE SET"; done
[ "$(count)" = "$begun" ] || wrong "the count has moved in a run that was refused"
prove prove 0 "example.proof example.y " "p258 end 0" "$W/isk"
one=$(($(count) - begun))
[ "$(listed "$W/prove.screen")" = "1 0" ] || wrong "prove: wanted the front to close once, and from then on lines of the front's list alone: closes and other lines: $(listed "$W/prove.screen")"
"$S/cp_verify" "$W/config" "$W/ivk" "$W/in/example.req" "$W/out.prove/example.y" "$W/out.prove/example.proof" -id "$id" >"$W/verify.log" 2>&1; got=$?
echo "[verify] cp_verify status $got: $(tail -n 1 "$W/verify.log" | cut -c1-80)"
[ "$got" = 0 ] || wrong "verify: wanted status 0"
CHAIN_TIMES=1 CHAIN_PUT_VECTORS="$W/put" prove variables 0 "example.proof example.y " "p258 end 0" "$W/isk"
"$S/cp_verify" "$W/config" "$W/ivk" "$W/in/example.req" "$W/out.variables/example.y" "$W/out.variables/example.proof" -id "$id" >"$W/variables.verify.log" 2>&1 || wrong "variables: cp_verify does not accept the proof: $(tail -n 1 "$W/variables.verify.log" | cut -c1-100)"
[ -z "$(files "$W/put")" ] || wrong "variables: the door's folder holds a file"
# The step tests nothing unless a test build reads both names. This is checked in the sources' text,
# not by a run: each name must be an argument of TESTENV() (prover/common/setting.h), the macro that
# reads a test variable.
for v in CHAIN_TIMES CHAIN_PUT_VECTORS; do grep -r -q -F "TESTENV(\"$v\")" "$here/../prover" "$here/../hiding" || wrong "variables: no source under prover/prover or prover/hiding reads $v: the scene sets a name that no build knows"; done
diff <(sed 's/out\.prove/OUT/g; s/[0-9][0-9.]*/N/g' "$W/prove.screen") <(sed 's/out\.variables/OUT/g; s/[0-9][0-9.]*/N/g' "$W/variables.screen") >/dev/null || wrong "variables: the screen is another, numbers and the name of -out apart"
cmp -s "$W/out.prove/example.y" "$W/out.variables/example.y" && ! cmp -s "$W/out.prove/example.proof" "$W/out.variables/example.proof" || wrong "variables: two runs on one request: expected the same reply and two different proofs"
bears=$(awk '$1 == "table" { print $5 }' "$W/config/policy.txt"); [[ $bears =~ ^(0|[123456789][0123456789]{0,8})$ ]] || { echo "serving_first_run.sh: the fifth word of the line table in $W/config/policy.txt must be a whole number from 0 to 999999999, with no leading zero" >&2; exit 2; } # only a plain decimal number may reach the arithmetic below: bash runs a command written inside a value such as x[$(command)]
echo "[copy spent] a stored copy allows $bears opened columns; a batch opens $one"
[ "$one" -gt 0 ] && cp -a "$W/isk" "$W/isk.spent" && truncate -s $((bears - one + 1)) "$W/isk.spent/leaf/0/count.bin" || wrong "copy spent: the copy of the secret folder could not be prepared"
prove copy-spent 1 "" "p258 end 1 cp-copy-spent" "$W/isk.spent"
[ "$(stat -c %s "$W/isk.spent/leaf/0/count.bin")" = $((bears - one + 1)) ] || wrong "copy spent: the count changed"
mkdir "$W/req" && (umask 077 && mkdir "$W/spool") || exit 2
python3 -I -B - "$W/config/policy.txt" "$W/req" <<'END' || exit 2
import struct, sys
word = dict(line.split() for line in open(sys.argv[1]) if len(line.split()) == 2)
T, V = 1 << int(word["lt"]), int(word["V"])
for n in range(6):  # six different windows of made-up tokens; the question: which token comes next
    tokens = [(2654435761 * (t + 1 + 1000 * n) >> 7) % V for t in range(T)]
    open("%s/%d.req" % (sys.argv[2], n), "wb").write(struct.pack("<%dI" % (T + 2), *tokens, T - 1, T))
END
for n in 0 1 2 3 4 5; do python3 -B "$here/../../driver/spool.py" lay "$W/spool" "$W/req/$n.req" >"$W/req/$n.id" || wrong "spool: request $n cannot be laid"; done
"$S/cp_prove" -isk "$W/isk" -spool "$W/spool" -batch 2 -in-step 2 -end-when-empty >"$W/spool.screen" 2>&1; got=$?
passed=0
for n in 0 1 2 3 4 5; do i=$(cat "$W/req/$n.id"); "$S/cp_verify" "$W/config" "$W/ivk" "$W/req/$n.req" "$W/spool/out/$i.y" "$W/spool/out/$i.proof" -id "$id" >"$W/spool.verify.$n.log" 2>&1; v=$?; if [ $v = 0 ]; then passed=$((passed + 1)); fi; done
seen="batches of two: $(grep -c -E '^batch [0-9]+ leaf [0-9]+ n 2$' "$W/spool.screen"); the front closes, other lines behind: $(listed "$W/spool.screen"); proofs: $(find "$W/spool/out" -maxdepth 1 -name '*.proof' | wc -l); failed: $(find "$W/spool/out" -maxdepth 1 -name '*.failed' | wc -l); left in in/ and work/: $(find "$W/spool/in" "$W/spool/work" -maxdepth 1 -name '*.req' 2>/dev/null | wc -l); left staged: $(find "$W/spool/out/.tmp" -type f ! -name mark.txt 2>/dev/null | wc -l); tags: $(find "$W/spool/out" -maxdepth 1 -type f -name '*.tag' | wc -l); in out/ beside whole responses: $(find "$W/spool/out" -maxdepth 1 -type f ! -name '*.proof' ! -name '*.y' ! -name '*.tag' | wc -l); accepted: $passed"
echo "[spool] cp_prove status $got; the last line: $(tail -n 1 "$W/spool.screen" | cut -c1-100); $seen"
[ "$got" = 0 ] && [ "$(tail -n 1 "$W/spool.screen")" = "p258 end 0" ] && [ "$seen" = "batches of two: 3; the front closes, other lines behind: 1 0; proofs: 6; failed: 0; left in in/ and work/: 0; left staged: 0; tags: 6; in out/ beside whole responses: 0; accepted: 6" ] || wrong "spool: wanted status 0, «p258 end 0», three batches of two, six proofs, nothing staged, nothing else in out/, all accepted"
"$P/cp_commit" -weights made-up-blocks -config "$W/config" -M 1 -ivk "$W/ivk.t" -isk "$W/isk.t" -test-seed "$W/seed.test" >"$W/commit.t.log" 2>&1 || wrong "seed test: the tests' build's cp_commit: status $?"
prove seed-test 1 "" "p258 end 1 cp-isk-seed-test" "$W/isk.t"; says seed-test "refused (cp-isk-seed-test): "
cp -a "$W/isk" "$W/isk.bent" && python3 -I -c 'import sys; p = sys.argv[1]; b = bytearray(open(p, "rb").read()); b[-1] ^= 1; open(p, "wb").write(b)' "$W/isk.bent/leaf/0/entry.bin" || wrong "the bent copy cannot be laid"
prove bent 0 "example.proof example.y " "p258 end 0" "$W/isk.bent"
[ "$(listed "$W/bent.screen")" = "1 0" ] || wrong "bent: wanted the front to close once, and from then on lines of the front's list alone"
"$S/cp_verify" "$W/config" "$W/ivk" "$W/in/example.req" "$W/out.bent/example.y" "$W/out.bent/example.proof" -id "$id" >"$W/bent.verify.log" 2>&1; got=$?
echo "[bent] cp_verify under the published leaf file: status $got: $(tail -n 1 "$W/bent.verify.log" | cut -c1-80)"
[ "$got" = 0 ] || wrong "bent: wanted cp_verify to accept the response under the published leaf file"
cp -a "$W/ivk" "$W/ivk.bent" && cp -f "$W/isk.bent/leaf/0/entry.bin" "$W/ivk.bent/leaf/0.bin" || wrong "bent: the public folder with the altered file cannot be laid"
"$S/cp_verify" "$W/config" "$W/ivk.bent" "$W/in/example.req" "$W/out.bent/example.y" "$W/out.bent/example.proof" -id "$id" >"$W/bent.verify.altered.log" 2>&1; got=$?
echo "[bent] cp_verify under the altered file: status $got: $(tail -n 1 "$W/bent.verify.altered.log" | cut -c1-80)"
[ "$got" = 1 ] && [[ $(tail -n 1 "$W/bent.verify.altered.log") == "NO: "* ]] || wrong "bent: wanted cp_verify to reject the altered leaf file: status 1, a last line that begins «NO: »"
if [ $bad = 0 ]; then echo "serving_first_run: all as expected"; else echo "serving_first_run: FAILED"; exit 1; fi
