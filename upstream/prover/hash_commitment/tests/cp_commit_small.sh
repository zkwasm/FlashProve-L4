#!/bin/bash
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.

# Runs the committer program cp_commit (a test build: prover/build.sh tests) on a GPU in a series of
# named cases and writes one line for each observation to <work>/res/summary.txt, which it prints at
# the end. It compares nothing with expected values: its exit status is 0 whatever the lines say
# (2 for unusable arguments). The weights are synthetic (`made-up-blocks`); the configurations are
# prover/tests/configurations/pinned_no_proof_of_work and
# prover/tests/configurations/pinned_no_proof_of_work_roomy (small; neither asks for proof of work).
# Printed words: `ivk`, `isk` = the public and the secret key directory; anchor = a commitment's
# first weight table; `copy` = leaf table, a further table of the same weights; `entry` = leaf file,
# ivk/leaf/<k>.bin; drawn seed = one that cp_commit draws itself (no -test-seed);
# `refused` = rejected; folder = directory. Expected lines: SUITE.md, test `gpu.cp_commit.small`.
# In short: the cases `refuse-*`, `kept-part`, `second-form` and `table-there` end with exit status
# 2, `short` and `table-short` with 7, all others with 0; `SAME FILES`, `SAME KEY`; `OTHER BYTES` in
# key.bin and in the leaf file for the two drawn seeds; rejected calls make no directory
# (`folders made: 0`, `folders left: 0`), and `refuse-there` changes no file.
# Nothing here shows that the programs are sound or hide anything.
# usage: cp_commit_small.sh <bin> <work> [<field: 1 | 2>]
#   <bin>   directory with a test build's cp_commit and the verifier program cp_verify
#   <work>  a directory that must not exist yet; the case <name> logs to <work>/res/<name>.log
#   <field> must be 1 or 2 and has no other effect: both values run the same configurations
#   GL      (environment, optional) a command prefix for every cp_commit call, such as a GPU lock
# Started by the runner; by hand: below a directory of your own. (Runner: run_suite.py.)
# Absolute paths (the repository's too): letters, digits and / . _ - only. <work>'s path is checked
# before anything is made, and no part of it may be ".." or begin with "-". CDPATH is unset: with it
# set, a cd could end in a directory of the same name elsewhere. <work> is made with mode 700. Name
# it below directories in which only you can write: the script does not check them. If the new
# <work> fails made_here(), or cd does not lead to it, the script stops and leaves <work> in place.
# The script itself deletes nothing but its own empty <work>, when it rejects a path or the field
# after making it. Missing directories above <work> are made too (mkdir -p) and are never removed.
# The first cases. `whole`, `again`: -M 2 makes the anchor and two leaf tables. `key-alone`:
# -leaves none makes the anchor and the key file (ivk/key.bin), but no leaf table. `second-form`:
# the call's second form (no -weights: add leaf tables to an existing commitment) is rejected.
# verdict() counts two forms of cp_commit's last line. A test build says that `ivk` is NOT to be
# handed out, whatever the seed and the options. Only the production build (prover/build.sh's
# default) says "can be handed out" or, with -keep-for-the-audit, that the audit must pass first
# (tested by prover/tests/serving_first_run.sh).
set -u
unset CDPATH
made_here() { # <a folder that this script has just made>: it must be this account's, closed to others (mode 700; 2700 below a set-group-ID directory) and empty
  # One cd, then three tests of ".", so that all three see the same directory. If stat or ls fails,
  # the check fails too. With -P, cd resolves symbolic links before "..", as mkdir does. The slash
  # after $1 keeps cd from taking a directory named "-" for the previous directory.
  (cd -P -- "$1/" && [ -O . ] && [[ $(stat -c %a .) == *700 ]] && list=$(ls -A) && [ -z "$list" ]) && return
  echo "cp_commit_small.sh: $1 is not the folder that was just made: another was put in its place: name a new folder below folders in which only you can write" >&2
  return 1
}
[ $# = 2 ] || [ $# = 3 ] || { echo "usage: cp_commit_small.sh <bin> <work> [<field>]" >&2; exit 2; }
B=$(cd -- "$1" && pwd) || exit 2
case $2 in /*) J=$2 ;; *) J=$PWD/$2 ;; esac; case /$J/ in */-*|*/../*|*[!A-Za-z0-9_./-]*) echo "cp_commit_small.sh: the work folder's whole path (the current directory included) may hold letters, digits" \
  "and _ . / - only, and no part of it may be \"..\" or begin with \"-\": name another" >&2; exit 2 ;; esac
mkdir -p -- "$(dirname -- "$2")" && mkdir -m 700 -- "$2" && made_here "$2" || exit 2
J=$(cd -P -- "$2" && pwd) && [ "$J" -ef "$2" ] || { echo "cp_commit_small.sh: cd does not lead to the folder just made: name another" >&2; exit 2; } # -P: the name as the kernel reads it, which is what cp_commit prints for a relative name; and a removal just below takes it
H=$(cd -- "$(dirname "$0")/../../.." && pwd) || exit 2
case "$B$J$H" in *[!A-Za-z0-9_./-]*) # these names go unquoted, and into program text: other signs would send its calls elsewhere
  echo "cp_commit_small.sh: a folder's path may hold letters, digits and / . _ - only (the folder made for this run is removed again)" >&2
  rmdir -- "$J"
  exit 2 ;;
esac
F=${3:-1}
case $F in 1 | 2) ;; *) # the field is only checked: it goes into no configuration's name
  echo "cp_commit_small.sh: the field is 1 or 2 (the folder made for this run is removed again)" >&2
  rmdir -- "$J"
  exit 2 ;;
esac
GL=${GL:-}
K=$H/prover/tests/configurations/pinned_no_proof_of_work
mkdir -p $J/res
python3 -I -c "open('$J/seed.bin','wb').write(bytes(range(1, 33)))" # made up, and none of the seeds that are written in the source
run() { local name=$1; shift; $GL $B/cp_commit "$@" > $J/res/$name.log 2>&1; echo "exit status $?" >> $J/res/$name.log; echo "$name: $(tail -1 $J/res/$name.log)" >> $J/res/summary.txt; }
sums() { ( cd -- $1 && find . -type f | sort | xargs sha256sum ); }
same() { cmp -s res/$1.ivk.sums res/$2.ivk.sums && cmp -s res/$1.isk.sums res/$2.isk.sums && echo "$3: SAME FILES" >> res/summary.txt || echo "$3: OTHER FILES" >> res/summary.txt; }
cd -- "$J" || exit 2
run whole -weights made-up-blocks -config $K -M 2 -ivk $J/a.ivk -isk $J/a.isk -test-seed $J/seed.bin
run again -weights made-up-blocks -config $K -M 2 -ivk $J/b.ivk -isk $J/b.isk -test-seed $J/seed.bin
for f in a b; do sums $f.ivk > res/$f.ivk.sums; sums $f.isk > res/$f.isk.sums; done
same a b "the whole run, twice under one test's seed"
verdict() { echo "$1: the closing line: $(grep -c '^cp_commit: done\..* is NOT to be handed out\. ' res/$1.log) says NOT to be handed out, $(grep -c '^cp_commit: done\..* is complete and can be handed out\. ' res/$1.log) says can be handed out" >> res/summary.txt; }
verdict whole
run key-alone -weights made-up-blocks -config $K -M 2 -ivk k.ivk -isk k.isk -test-seed $J/seed.bin -leaves none # relative names (the working folder is $J): see "from the root" below
cmp -s a.ivk/key.bin k.ivk/key.bin && echo "the whole run's key and the key alone: SAME KEY" >> res/summary.txt || echo "the whole run's key and the key alone: OTHER KEYS" >> res/summary.txt
echo "the key alone: entries in ivk: $(find k.ivk -path '*leaf*' -type f | wc -l)" >> res/summary.txt
$B/cp_verify -key $K a.ivk/key.bin -q > res/verify-key.log 2>&1; echo "cp_verify -key: status $?: $(tail -1 res/verify-key.log)" >> res/summary.txt
for n in 0 1; do $B/cp_verify -leaf $K a.ivk/key.bin a.ivk/leaf/$n.bin -q > res/verify-leaf-$n.log 2>&1; echo "cp_verify -leaf, copy $n: status $?: $(tail -1 res/verify-leaf-$n.log)" >> res/summary.txt; done
cmp -s a.ivk/leaf/0.bin a.ivk/leaf/1.bin && echo "the two copies' entries: SAME ENTRY" >> res/summary.txt || echo "the two copies' entries: OTHER ENTRIES" >> res/summary.txt
run second-form -isk $J/k.isk -ivk $J/k.ivk -leaves 0..1 -test-seed $J/seed.bin
echo "second-form: $(grep -c 'refused (cp-commit-first-form-alone)' res/second-form.log) line with its refusal's name; entries in ivk: $(find k.ivk -path '*leaf*' -type f | wc -l)" >> res/summary.txt
# With a drawn seed cp_commit always makes its worst-case check (can a prompt drive an inner value
# out of its range?), before it draws the seed. The plain configuration's synthetic scales fail it:
# exit status 7, and nothing is written. The _roomy configuration passes.
# In isk.txt the line `seed test` marks a test seed, the line `ranges` a passed worst-case check.
# Of `drawn-c` and `drawn-d`, key.bin and leaf/0.bin are compared one by one: isk/seed.bin differs
# anyway, so one comparison of all files would always report a difference.
run short -weights made-up-blocks -config $K -M 1 -ivk $J/s.ivk -isk $J/s.isk
echo "short: files written: $(find s.ivk s.isk s.ivk.making s.isk.making -type f 2>/dev/null | wc -l)" >> res/summary.txt
for d in c d; do
  run drawn-$d -weights made-up-blocks -config ${K}_roomy -M 1 -ivk $J/$d.ivk -isk $J/$d.isk
  echo "drawn-$d: $(grep -c 'check: every bounded site fits' res/drawn-$d.log) line every bounded site fits; isk.txt: $(grep -c '^seed test$' $d.isk/isk.txt) line seed test, $(grep -c '^ranges ' $d.isk/isk.txt) line ranges; the seed's file: $(stat -c '%a %s' $d.isk/seed.bin)" >> res/summary.txt
  sums $d.ivk > res/$d.ivk.sums; sums $d.isk > res/$d.isk.sums
done
for f in key.bin leaf/0.bin; do if [ ! -s c.ivk/$f ] || [ ! -s d.ivk/$f ]; then echo "two runs under drawn seeds: $f: MISSING"; elif cmp -s c.ivk/$f d.ivk/$f; then echo "two runs under drawn seeds: $f: THE SAME BYTES"; else echo "two runs under drawn seeds: $f: OTHER BYTES"; fi >> res/summary.txt; done
verdict drawn-c
# cp_commit has cp_verify check the key file and each leaf file that it makes, and requests the bits
# that a verification needs; only this case requests fewer (-test-bits 0).
run drawn-bits -weights made-up-blocks -config ${K}_roomy -M 1 -ivk $J/t.ivk -isk $J/t.isk -test-bits 0
verdict drawn-bits
# -keep-for-the-audit. The keys for the private randomness in key.bin and in each leaf file are each
# derived from the seed and 32 fresh bytes of the operating system (zeros under a test seed). The
# option keeps these bytes in isk/audit, one file for each key, and must change no other file.
# kept() prints each file's name, mode, length and hard links (expected: 600 32 1). Of the
# directory's mode only the nine permission bits are printed: a parent directory's set-group-ID bit
# is inherited. `kept-test` (the seed of `whole`): zeros, and every other file as in `whole`.
# `kept-drawn`: not zeros, and different from file to file. That the kept bytes are those that went
# into the keys cannot be seen from the files: only a passed audit of such a directory shows it
# (described in tools/audit/README.md).
kept() { ( cd -- $1.isk/audit 2>/dev/null && for f in $(ls | LC_ALL=C sort); do printf '%s %s %s; ' "$f" "$(stat -c '%a %s %h' "$f")" "$(cmp -s "$f" <(head -c 32 /dev/zero) && echo zeros || echo other)"; done ); }
run kept-test -weights made-up-blocks -config $K -M 2 -ivk $J/u.ivk -isk $J/u.isk -test-seed $J/seed.bin -keep-for-the-audit
echo "kept-test: isk/audit ($(stat -c %a u.isk/audit 2>/dev/null | tail -c 4)): $(kept u)" >> res/summary.txt
sums u.ivk | cmp -s - res/a.ivk.sums && sums u.isk | grep -v ' \./audit/' | cmp -s - res/a.isk.sums && echo "kept-test against whole, isk/audit apart: SAME FILES" >> res/summary.txt || echo "kept-test against whole, isk/audit apart: OTHER FILES" >> res/summary.txt
verdict kept-test
run kept-drawn -weights made-up-blocks -config ${K}_roomy -M 2 -ivk $J/w.ivk -isk $J/w.isk -keep-for-the-audit
echo "kept-drawn: isk/audit ($(stat -c %a w.isk/audit 2>/dev/null | tail -c 4)): $(kept w)different files among them: $(sha256sum w.isk/audit/* 2>/dev/null | cut -c1-64 | sort -u | wc -l)" >> res/summary.txt
verdict kept-drawn
echo "kept-drawn: the head's line on the option: $(grep -c '^cp_commit: -keep-for-the-audit: ' res/kept-drawn.log)" >> res/summary.txt
run kept-part -weights made-up-blocks -config $K -M 2 -leaves 0..1 -ivk $J/x.ivk -isk $J/x.isk -test-seed $J/seed.bin -keep-for-the-audit
echo "kept-part: the option with -leaves: $(grep -c '^usage: cp_commit' res/kept-part.log) usage text; folders made: $(ls -d x.ivk x.isk x.ivk.making x.isk.making 2>/dev/null | wc -l)" >> res/summary.txt
echo "without the option: folders audit in the other runs' isk: $(ls -d a.isk/audit c.isk/audit t.isk/audit 2>/dev/null | wc -l)" >> res/summary.txt
cmp -s c.isk/seed.bin d.isk/seed.bin && echo "the two drawn seeds: EQUAL" >> res/summary.txt || echo "the two drawn seeds: DIFFER" >> res/summary.txt
run refuse-there -weights made-up-blocks -config ${K}_roomy -M 1 -ivk $J/c.ivk -isk $J/c.isk
echo "refuse-there: $(grep -c 'refused (file-there-already)' res/refuse-there.log) line with its refusal's name" >> res/summary.txt
sums c.ivk | cmp -s - res/c.ivk.sums && sums c.isk | cmp -s - res/c.isk.sums && echo "after refuse-there: NOTHING HAS CHANGED" >> res/summary.txt || echo "after refuse-there: SOMETHING HAS CHANGED" >> res/summary.txt
# Working names: cp_commit works in <isk>.making and <ivk>.making and renames both when the run has
# succeeded. If either name exists already, a new call fails and changes nothing. The two
# directories made here are not a killed call's: each holds a file, `theirs`, that cp_commit never
# makes. Under a umask that leaves them open to group or others, that mode alone makes cp_commit
# call them not its own, and it never lists them. What a killed call leaves is not tested here.
mkdir left.isk.making other.ivk.making && : > left.isk.making/theirs && : > other.ivk.making/theirs
run refuse-left-isk -weights made-up-blocks -config ${K}_roomy -M 1 -ivk left.ivk -isk left.isk # relative names
run refuse-left-ivk -weights made-up-blocks -config ${K}_roomy -M 1 -ivk $J/other.ivk -isk $J/other.isk
run refuse-a-working-name -weights made-up-blocks -config ${K}_roomy -M 1 -ivk $J/x.making -isk $J/x
run refuse-one-name -weights made-up-blocks -config ${K}_roomy -M 1 -ivk $J/y -isk $J/y
echo "one name for both: $(grep -c 'refused (usage): -isk and -ivk name the same folder' res/refuse-one-name.log) line; folders made: $(ls -d y y.making 2>/dev/null | wc -l)" >> res/summary.txt
echo "a left working name: $(cat res/refuse-left-isk.log res/refuse-left-ivk.log | grep -c 'refused (file-there-already): .*\.making exists already') lines with the refusal's name; the foreign files kept: $(ls left.isk.making/theirs other.ivk.making/theirs 2>/dev/null | wc -l); folders made: $(ls -d left.isk left.ivk left.ivk.making other.isk other.ivk other.isk.making 2>/dev/null | wc -l)" >> res/summary.txt
echo "a given name that ends in .making: $(grep -c 'refused (usage): .*x\.making: a name that ends in \.making' res/refuse-a-working-name.log) line; folders made: $(ls -d x x.making x.making.making 2>/dev/null | wc -l)" >> res/summary.txt
echo "working names left by all the runs above: $(ls -d ./*.isk.making ./*.ivk.making 2>/dev/null | grep -c -v '^./left\.isk\.making$\|^./other\.ivk\.making$'); the head's line in the whole run: $(grep -c '^cp_commit: the two folders are made as .*a\.isk\.making and .*a\.ivk\.making and get their names' res/whole.log)" >> res/summary.txt
# cp_commit prints a relative -ivk or -isk as an absolute path ("from the root"): its lines on what
# an interrupted call leaves may be read later, in another working directory.
# A printed "can be removed" is as dangerous as a removal: cp_commit says it only of a directory
# that looks like a killed call's own, so it must not say it of the two made above (expected: 0).
echo "relative names are printed from the root: $(grep -c -F "cp_commit: the two folders are made as $J/k.isk.making and $J/k.ivk.making and get their names, $J/k.isk and $J/k.ivk, at the run's good end" res/key-alone.log) line at the start of a making, $(grep -c -F "$J/left.isk.making exists already. No call of cp_commit by this user has made this, or not this alone: " res/refuse-left-isk.log) in the refusal at a left working name" >> res/summary.txt
echo "a left working name that cp_commit has not made: lines that tell the reader to remove something: $(cat res/refuse-left-isk.log res/refuse-left-ivk.log | grep -c -e 'can be removed')" >> res/summary.txt
run refuse-budget -weights made-up-blocks -config $K -M 4000000 -ivk $J/e.ivk -isk $J/e.isk -test-seed $J/seed.bin
echo "refuse-budget: $(grep -c 'refused (cp-commit-anchor-budget)' res/refuse-budget.log) line with its refusal's name; folders made: $(ls -d e.ivk e.isk e.ivk.making e.isk.making 2>/dev/null | wc -l)" >> res/summary.txt
# The edge of the limit that `refuse-budget` exceeds. The anchor's random entries allow `bears` of
# its columns to be shown in all; the key file's one-time test (the committer's own test on the
# anchor) shows `one` of them, and each leaf table's LINK proof, which ties that table to the
# anchor, shows `link` (numbers from the rejection's message). The largest M with
# one + M x link <= bears is accepted (-leaves none: no leaf table is made), and M + 1 is
# rejected. This catches a wrong comparison (a doubled bound, a lost term, an off-by-one), not a
# wrong number that the message and the comparison share.
read -r one link bears <<<"$(sed -n "s/.*refused (cp-commit-anchor-budget): \([0-9]*\) + M x \([0-9]*\) = [0-9]* columns of the anchor would be shown, and a row's random entries bear \([0-9]*\):.*/\1 \2 \3/p" res/refuse-budget.log)"
if [ -n "${bears:-}" ] && [ "$link" -gt 0 ] && [ "$bears" -ge "$one" ]; then
  edge=$(( (bears - one) / link ))
  run budget-edge -weights made-up-blocks -config $K -M $edge -leaves none -ivk $J/g.ivk -isk $J/g.isk -test-seed $J/seed.bin
  run refuse-budget-beyond -weights made-up-blocks -config $K -M $((edge + 1)) -leaves none -ivk $J/h.ivk -isk $J/h.isk -test-seed $J/seed.bin
  echo "the budget's edge: the greatest M is above 2 and below 4000000: $([ $edge -gt 2 ] && [ $edge -lt 4000000 ] && echo yes || echo no); one more: $(grep -c 'refused (cp-commit-anchor-budget)' res/refuse-budget-beyond.log) line with the refusal's name; folders made: $(ls -d h.ivk h.isk h.ivk.making h.isk.making 2>/dev/null | wc -l)" >> res/summary.txt
else echo "the budget's edge: NOT PLAYED: the refusal's line does not give its three numbers" >> res/summary.txt; fi
# -worst-case-table <file> writes the worst-case check's whole table into a new file, and makes
# cp_commit run the check under a test seed too. The table's numbers are computed from the weights:
# the file is for its owner alone (expected mode: 600), and a call that fails after the check
# removes it, which is not tested here. cp_commit keeps the file at exit status 7 (`table-short`),
# where it is the answer: it says which inner values (`sites`) do not fit; and after a successful
# run (`table-good`).
# `table-there`: a file name that exists already is rejected, and the file keeps its bytes.
# Also not tested here: an end by a signal; an end before cp_commit's cleanup is armed, where exit()
# or a signal leaves the file (`TableGoes` in prover/hash_commitment/hf_commit.cu).
table() { if [ -e "$1" ]; then echo "there, mode $(stat -c %a "$1"), $([ -s "$1" ] && echo with bytes || echo empty)"; else echo "not there"; fi; }
run table-short -weights made-up-blocks -config $K -M 1 -ivk $J/ts.ivk -isk $J/ts.isk -worst-case-table $J/ts.table
echo "table-short: the table's file: $(table ts.table)" >> res/summary.txt
run table-good -weights made-up-blocks -config ${K}_roomy -M 1 -leaves none -ivk $J/tg.ivk -isk $J/tg.isk -test-seed $J/seed.bin -worst-case-table $J/tg.table
echo "table-good: the table's file: $(table tg.table)" >> res/summary.txt
sha256sum tg.table > res/tg.table.sum
run table-there -weights made-up-blocks -config ${K}_roomy -M 1 -leaves none -ivk $J/tt.ivk -isk $J/tt.isk -test-seed $J/seed.bin -worst-case-table $J/tg.table
echo "table-there: $(grep -c 'refused (no-table-file)' res/table-there.log) line with its refusal's name; the file that stood: $(sha256sum -c --quiet res/tg.table.sum > /dev/null 2>&1 && echo THE SAME BYTES || echo OTHER BYTES OR GONE); folders left: $(ls -d tt.ivk tt.isk tt.ivk.making tt.isk.making 2>/dev/null | wc -l)" >> res/summary.txt
cat res/summary.txt
