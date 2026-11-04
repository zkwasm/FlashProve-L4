#!/usr/bin/env bash
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# tools/accepts/accepts.sh: one command that asks whether Lean's checks behind the parser accept a
# stored example's files as the C++ prover wrote them, and whether they refuse changed ones.
# A test, not a theorem: see "What a yes does not mean" in lean/Ask/README.md.
#
# usage: tools/accepts/accepts.sh small|full [honest|controls|all]        (default: all)
#   small: examples/accepted-proof. full: examples/full-size. The table that comes with this script
#   has calls for small alone: full stops with exit code 2.
#   Run it from anywhere. It needs no sudo: run it as an ordinary user. It needs what step 4 of
#   lean/CHECKING.md and tools/parser/README.md need: the library built in lean/ with its packages
#   in place, lake, bash, GNU coreutils, findutils, grep and sed, awk, g++ 13 or later, python3.
#   It calls no network tool, and stops if lean/.lake/packages is missing. Lake itself may still go
#   to the network if a package there is not as lean/lake-manifest.json has it, and elan if this
#   account lacks the Lean version of lean/lean-toolchain (under sudo it may).
#   ACCEPTS_WORK: an existing empty directory of yours to work in, closed to your group and to
#   others (chmod go-w). Each directory above it must be yours or root's, and closed likewise or
#   have the sticky bit, as /tmp has: else the script stops before it writes (default: mktemp -d).
#   ACCEPTS_TIME_LIMIT: seconds after which one call of Lean is stopped (default: 14400; not 0).
#   ACCEPTS_JOBS: how many calls of Lean run side by side (default: 1). Each takes one core and some
#   10 GiB with mapped files (measured at the small size).
#   ACCEPTS_SHEETS: another table of asking files than tools/accepts/sheets.tsv.
# Which asking files it calls stands in that table, one call a line, fields parted by blanks or
# tabs:
#   size  part  name  file-below-lean/Ask  parsed-file|-  controls-that-the-file-plays
#   part: honest | controls | both (a file that plays the unchanged files and then controls, and
#   is to be called whichever part is asked for).
#   Calls whose names begin with "head-" (the table that comes with this script has none) feed four
#   lines at the end of the print, for a run in which a check is being rewritten step by step. The
#   lines' words stand in the table ("#four-1: ..." to "#four-4: ..."); the script fills in what the
#   run printed: "own steps: k of n; written anew: a; pure: p; alone: b", the answer at the
#   unchanged files, "the unchanged files took: n ms", and for each control whether some test said
#   NO, and whether one of these was a step (the list behind "said no:" then names a "step ..."), or
#   something behind the steps (the list then begins with the words of the table's line
#   "#four-behind: ..."). "#four-note <control>: ..." gives words for one control that is known not
#   to be refused; they are printed only if just that control was not refused.
# Beside its print it writes RECORD.txt into the work directory: one fact a line, for a program.
# It writes only below the work directory; Lake also writes below lean/.lake, and the compilers use
# the system's temporary directory. It deletes nothing: the work directory stays, delete it when
# you are done.
# Exit code: 0 all that was asked for ended as expected. 1 a wrong answer: the honest files got a
# NO, the parser refused an honest file, a control did not end as expected, or Lean panicked.
# 2 something was not tested: lean/ is not built, a tool or a file is missing, a stored file is not
# as SHA256SUMS has it, a wrong call or a faulty table, a work directory that another account could
# change or swap, the library or the parser could not be made ready, Lean stopped before it ran a
# test, or a run played fewer or more controls than the table says. 3 a run did not end: it was
# killed, ran out of memory or stack, passed the time limit, or a signal stopped the script. If
# calls end differently, 1 goes before 3, and 3 before 2. All calls are run even after a wrong
# answer.
set -u
LC_ALL=C; export LC_ALL
unset CDPATH
umask go-w # what is written below the work directory is for this account alone to change
unset ASK_K # the runs then take what the rounds need (lean/Ask/README.md)

say() { printf '%s\n' "$*"; }
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
W=; size=${1:-}; part=${2:-all}; ask=${size:-nothing}${size:+ $part}
# One fact a line, into the record; nothing before the work directory stands.
rec() { if [ -n "$W" ]; then printf '%s\n' "$*" >> "$W/RECORD.txt"; fi; }
end_line() { say "END: asked: $ask; $1"; } # not $part: a trap may run inside open_above
stop() { code=$1; shift; rec "exit $code"; rec "end $*"; end_line "$*"; exit "$code"; }
# Ctrl-C or a TERM ends the script with its END line. In step 4 it is also passed on to each call of
# Lean, which timeout puts into a process group of its own.
# The calls' sub-shells are signalled as jobs, not by number: bash sends nothing to a job whose
# process it has already collected, while a number may by then be another process's.
# shellcheck disable=SC2154
trap 'for j in $(jobs -r | sed -n "s/^\[\([0-9]*\)\].*/\1/p"); do kill -TERM "%$j" 2>/dev/null; done
  stop 3 "a run did not end: a signal stopped this script"' INT TERM HUP

usage="nothing was tested: usage: accepts.sh small|full [honest|controls|all]"
top=$(cd -- "$(dirname -- "$0")/../.." && pwd -P) || stop 2 "nothing was tested: no top directory"
case "$size" in
  small) ex=$top/examples/accepted-proof; which=small ;;
  full) ex=$top/examples/full-size; which=full-size ;;
  *) stop 2 "$usage" ;;
esac
case "$part" in honest|controls|all) ;; *) stop 2 "$usage" ;; esac
[ $# -le 2 ] || stop 2 "$usage"

limit=${ACCEPTS_TIME_LIMIT:-14400}
case "$limit" in ''|*[!0-9]*|0*)
  stop 2 "nothing was tested: ACCEPTS_TIME_LIMIT must be a whole number of seconds above 0" ;; esac
jobs=${ACCEPTS_JOBS:-1}
case "$jobs" in ''|*[!0-9]*|0*)
  stop 2 "nothing was tested: ACCEPTS_JOBS must be a whole number above 0" ;; esac
sheets=${ACCEPTS_SHEETS:-$top/tools/accepts/sheets.tsv}
case "$sheets" in /*) ;; *) sheets=./$sheets ;; esac # awk would read a=b.tsv as an assignment
[ -f "$sheets" ] || stop 2 "nothing was tested: the table of asking files is missing: $sheets"
# g++ is called only by tools/parser/parse_files.sh; without it, step 3 stops with exit 2.
for t in lake python3 sha256sum timeout find awk comm grep sed sort xargs stat readlink id; do
  command -v "$t" >/dev/null 2>&1 || stop 2 "nothing was tested: $t is not on the PATH"
done
[ -d "$top/lean/.lake/build/lib" ] ||
  stop 2 "nothing was tested: lean/ is not built (lean/CHECKING.md, step 4)"
# Without its packages Lake would go to the network before it builds anything.
[ -d "$top/lean/.lake/packages" ] ||
  stop 2 "nothing was tested: lean/.lake/packages is missing (lean/CHECKING.md, step 3)"

# The calls that were asked for, out of the table. A line that begins with # is a comment.
bad=$(awk '$0 !~ /^#/ && NF != 0 && NF != 6' "$sheets" | head -n 1)
[ -z "$bad" ] || stop 2 "nothing was tested: a line of the table does not have six fields: $bad"
bad=$(awk '$0 !~ /^#/ && NF == 6 && ($1 !~ /^(small|full)$/ ||
  $2 !~ /^(honest|controls|both)$/ || $5 !~ /^(key|leaf-0|proof|-)$/)' "$sheets" | head -n 1)
[ -z "$bad" ] ||
  stop 2 "nothing was tested: a line of the table has a wrong first, second or fifth field: $bad"
calls=$(awk -v s="$size" -v p="$part" \
  '$0 !~ /^#/ && NF == 6 && $1 == s && (p == "all" || $2 == p || $2 == "both")' "$sheets")
[ -n "$calls" ] || stop 2 "nothing was tested: the table names no asking file for: $size $part"
for p in honest controls; do
  case "$part" in all|"$p")
    printf '%s\n' "$calls" | awk -v p="$p" '$2 == p || $2 == "both" { f = 1 } END { exit !f }' ||
      stop 2 "nothing was tested: the table names no asking file for: $size $p" ;;
  esac
done
heads=0
if printf '%s\n' "$calls" | awk '$3 ~ /^head-/ { f = 1 } END { exit !f }'; then
  heads=1
  for i in 1 2 3 4; do
    grep -q "^#four-$i: " "$sheets" ||
      stop 2 "nothing was tested: the table has calls named head-... and no line #four-$i:"
  done
  # What the run found must come from the run: a line that lacks its places in braces could say
  # anything.
  for b in '1 {k}' '1 {n}' '2 {answer}' '2 {seconds}' '3 {refused}' '3 {played}' '3 {note}'; do
    sed -n "s/^#four-${b%% *}: *//p" "$sheets" | head -n 1 | grep -q -F "${b##* }" ||
      stop 2 "nothing was tested: the table's line #four-${b%% *}: lacks ${b##* }"
  done
fi
twice=$(printf '%s\n' "$calls" | awk '{ print $3 }' | sort | uniq -d | head -n 1)
[ -z "$twice" ] || stop 2 "nothing was tested: two lines of the table have the same name: $twice"
while read -r _ kind name file _ want; do
  case "$name" in ''|*[!A-Za-z0-9_-]*)
    stop 2 "nothing was tested: a name in the table has a character other than letters, digits," \
      "- and _: $name" ;; esac
  case "$file" in ''|*[!A-Za-z0-9_./-]*|*..*|/*)
    stop 2 "nothing was tested: the table names a file that is not a plain path: $file" ;; esac
  [ -f "$top/lean/Ask/$file" ] || stop 2 "nothing was tested: lean/Ask/$file is missing"
  case "$kind:$want" in
    honest:0|controls:[1-9]|both:[1-9]|controls:[1-9][0-9]|both:[1-9][0-9]) ;;
    *) stop 2 "nothing was tested: the table's line on $name is wrong: its second field must be" \
      "honest, with 0 controls, or controls or both, with 1 to 99: $kind $want" ;;
  esac
done <<EOF
$calls
EOF

if [ -n "${ACCEPTS_WORK:-}" ]; then w=$ACCEPTS_WORK
else w=$(mktemp -d) || stop 2 "nothing was tested: mktemp -d failed"; fi
wd=$(cd -- "$w" 2>/dev/null && pwd -P) || # W stays empty, so nothing is written, till the looks end
  stop 2 "nothing was tested: the work directory cannot be entered"
# All looks are at wd, the path that is used. What cannot be listed does not count as empty.
if [ ! -d "$wd" ] || ! l=$(ls -A -- "$wd" 2>/dev/null) || [ -n "$l" ]; then
  stop 2 "nothing was tested: set ACCEPTS_WORK to an existing empty directory that you can list," \
    "or leave it unset"
fi
# Lean loads a library from there: nobody else may be able to write there.
if [ ! -O "$wd" ] || [ -n "$(find "$wd" -maxdepth 0 -perm /022)" ]; then
  stop 2 "nothing was tested: the work directory must be yours, and closed to your group and" \
    "to others (chmod go-w)"
fi
# Lean loads a library from W: no other account may be able to swap W or a directory above it.
if open_above "$wd/"; then
  stop 2 "nothing was tested: the work directory lies below $place: $why$cure"
fi
W=$real
say "work directory: $W"
sum() { sha256sum < "$1" | cut -d' ' -f1; }
rec "command tools/accepts/accepts.sh $size $part"
rec "script $(sum "$0")"
rec "sheets $(sum "$sheets")"
rec "size $size"
rec "jobs $jobs"
rec "hash BLAKE2b as Lean's library writes it (Fmt.oracle)"
rec "tree $(cd "$top/lean" && find . -name '*.lean' -not -path './.lake/*' -type f | sort |
  xargs sha256sum | sha256sum | cut -d' ' -f1)"
say "WHAT IS RUN: the asking files that $sheets names for: $size $part."
say "A run may print which description of the protocol it ran, in a line that begins"
say "\"description:\". What a yes of these runs means, and what it does not: lean/Ask/README.md."
if [ "$size" = full ]; then
  say "AT THE FULL SIZE THE CHECK OF EACH FILE IS ASKED ALONE: a yes here is one yes for each file,"
  say "not the joined check's ten tests, unless the table names a joined asking file."
fi

# 1. The stored files are as the list beside them has them.
( cd "$ex" && sha256sum -c SHA256SUMS ) > "$W/sums.txt" 2>&1 ||
  stop 2 "nothing was tested: a stored file is not as SHA256SUMS has it ($W/sums.txt)"
say "stored files as SHA256SUMS has them: $(grep -c ': OK$' "$W/sums.txt")" \
  "of $(wc -l < "$W/sums.txt")"
while read -r h f; do rec "file $f $h"; done < "$ex/SHA256SUMS"

# 2. The linked library, built again each time: an old one loads without an error (Ask/README.md).
# It holds OnBytes.Main and the modules of lean/ beneath it, found by their sources' import lines.
t0=$(date +%s)
targets=(); objects=(); seen=' '; todo=(OnBytes.Main)
while [ "${#todo[@]}" -gt 0 ]; do
  m=${todo[0]}; todo=("${todo[@]:1}"); f=$top/lean/${m//.//}.lean
  case $seen in *" $m "*) continue ;; esac
  seen="$seen$m "
  if [ -f "$f" ]; then
    targets+=("+$m:c.o.export"); objects+=(".lake/build/ir/${m//.//}.c.o.export")
    while read -r i || [ -n "$i" ]; do todo+=("$i"); done < <(
      sed -n -E 's/^(public |meta )*import (all )?([^ ]+).*/\3/p' "$f")
  else
    case $m in
      Init | Std | Lean | Lake | Init.* | Std.* | Lean.* | Lake.*) ;;
      *) stop 2 "nothing was tested: lean/ has no source for the module $m" ;;
    esac
  fi
done
( cd "$top/lean" && LEAN_ABORT_ON_PANIC=1 lake build "${targets[@]}" ) > "$W/lib-build.txt" 2>&1 ||
  stop 2 "nothing was tested: lake build failed on OnBytes.Main or a module that it imports" \
    "($W/lib-build.txt)"
printf '%s\n' "${objects[@]}" | sort > "$W/libvob.objects" ||
  stop 2 "nothing was tested: $W/libvob.objects cannot be written"
while read -r o; do c=${o%.o.export}; l=${c%.c}.olean
  if [ ! "$top/lean/$o" -nt "$top/lean/$c" ] ||
    [ ! "$top/lean/$o" -nt "$top/lean/.lake/build/lib/lean/${l#.lake/build/ir/}" ]; then
    stop 2 "nothing was tested: an object is not newer than its .c and .olean: $o" \
      "(the cure: lean/Ask/README.md, The faster way)"
  fi
done < "$W/libvob.objects"
# The objects' paths hold no blank: they are module names.
# shellcheck disable=SC2046
( cd "$top/lean" && lake env leanc -shared -o "$W/libvob.so" $(cat "$W/libvob.objects") ) \
  > "$W/lib-link.txt" 2>&1 ||
  stop 2 "nothing was tested: the library was not linked ($W/lib-link.txt)"
say "linked library: $(wc -l < "$W/libvob.objects") objects; sha256 $(sum "$W/libvob.so");" \
  "$(($(date +%s) - t0)) s"
rec "by Lean's evaluator, with compiled code loaded"
rec "loaded libvob.so $(sum "$W/libvob.so")"

# 3. The parser. Its refusal of an honest file is a wrong answer, not a skipped test.
# It also refuses a work directory below a directory that another account could change.
t0=$(date +%s)
"$top/tools/parser/parse_files.sh" "$ex/config" "$ex/ivk" "$ex/example.req" "$ex/example.proof" \
  "$W/work" > "$W/parser.txt" 2>&1
rc=$?
for f in "$top"/tools/parser/*; do
  if [ -f "$f" ]; then rec "parser ${f##*/} $(sum "$f")"; fi
done
case $rc in
  0) ;;
  1) stop 1 "A WRONG ANSWER: the parser refused an honest file ($W/parser.txt)" ;;
  *) stop 2 "nothing was tested: the parser stopped before it parsed: its exit code $rc" \
       "($W/parser.txt)" ;;
esac
P=$W/work/parsed
for f in key leaf-0 proof; do
  [ -f "$P/$f.parsed.bin" ] || stop 2 "nothing was tested: $P/$f.parsed.bin is missing"
  say "parsed file: $f.parsed.bin: $(wc -c < "$P/$f.parsed.bin") bytes;" \
    "sha256 $(sum "$P/$f.parsed.bin")"
  rec "parsed $f $(sum "$P/$f.parsed.bin")"
done
say "the parser: $(($(date +%s) - t0)) s"

# 4. One call of Lean for each line of the table, ACCEPTS_JOBS of them side by side.
wrong=0; unended=0; untested=0
behind_words=$(sed -n 's/^#four-behind: *//p' "$sheets" | head -n 1)
head_own=; head_ans=; head_s=; head_a=0; head_b=0; head_c=0; head_d=0
head_bad2=; head_bad3=; head_not=
head_anew=; head_alone=; head_pure=
played=0; asked=0; refused_all=0
# A panic of Lean's runtime does not always print the word: a read of an array out of bounds does
# not. Without LEAN_ABORT_ON_PANIC=1 Lean would go on with a default value and end with 0.
panic_words='PANIC|^Error: index out of bounds|lean_panic_fn|lean_array_get_panic|backtrace:'
start() { # <name> <asking file> <parsed file, or ->
  case "$3" in -) one= ;; *) one=$P/$3.parsed.bin ;; esac
  date +%s > "$W/$1.t0"
  # A shell of its own waits for the call, so that the call's end is timed when it ends, not when
  # its turn to be judged comes. It hands a TERM on to the call, and a HUP as a TERM: when a
  # terminal closes, this shell gets the HUP; the call, in a process group of its own, does not.
  ( cd "$top/lean" || exit 126
    c=; trap 'if [ -n "$c" ]; then kill -TERM %% 2>/dev/null; fi' TERM HUP
    ASK_SIZE=$size ASK_EXAMPLE=$ex ASK_PARSED=$one ASK_PARSED_KEY=$P/key.parsed.bin \
    ASK_PARSED_LEAF=$P/leaf-0.parsed.bin ASK_PARSED_PROOF=$P/proof.parsed.bin \
    LEAN_ABORT_ON_PANIC=1 timeout "$limit" lake env lean -j1 "--load-dynlib=$W/libvob.so" \
      --tstack=4000000 "Ask/$2" > "$W/$1.out" 2> "$W/$1.err" &
    c=$!; wait "$c"; s=$?
    if [ "$s" -gt 128 ] && kill -0 "$c" 2>/dev/null; then wait "$c"; s=$?; fi
    date +%s > "$W/$1.t1"; exit "$s" ) &
}
judge() { # <name> <asking file> <how many controls the file plays> <Lean's exit code>
  name=$1; file=$2; want=$3; st=$4; t0=$(cat "$W/$name.t0")
  t1=$(cat "$W/$name.t1" 2>/dev/null) || t1=$(date +%s)
  nomem=$(cat "$W/$name.out" "$W/$name.err" | grep -c 'INTERNAL PANIC: out of memory')
  panics=$(cat "$W/$name.out" "$W/$name.err" | grep -c -E "$panic_words")
  panics=$((panics - nomem))
  # Each control's closing line begins "CONTROL <its number or name>:". Every count is of distinct
  # controls: a line printed twice counts once.
  ids() { sed -n 's/^CONTROL \([A-Za-z0-9_-]*\):.*/\1/p' | sort -u | wc -l; }
  seen=$(ids < "$W/$name.err")
  # "AS EXPECTED" also stands in "NOT AS EXPECTED" and in "... WAS EXPECTED": these are taken off.
  notexp=$(grep '^CONTROL [A-Za-z0-9_-]*:' "$W/$name.err" |
    grep -e 'NOT AS EXPECTED' -e 'WAS EXPECTED' -e ALARM | ids)
  asexp=$(grep '^CONTROL [A-Za-z0-9_-]*:.*AS EXPECTED' "$W/$name.err" |
    grep -v -e 'NOT AS EXPECTED' -e 'WAS EXPECTED' -e ALARM | ids)
  # Refused: a control at which some test said NO, the answer is not yes and no alarm was raised.
  refused=$(grep '^CONTROL [A-Za-z0-9_-]*:.*said no: \[[^]]' "$W/$name.err" |
    grep -v -e ALARM -e ': answer yes' | ids)
  # The unchanged files' answer: of the ANSWER lines that stand before every control the first
  # that says NO, else the first.
  first=$(sed 's/ *\[[0-9]* ms\] *$//' "$W/$name.err" | awk '/^(=== )?CONTROL / { exit }
    /^ANSWER/ { if (f == "") f = $0; if (n == "" && $0 ~ /: NO$/) n = $0 }
    END { if (n != "") print n; else print f }')
  desc=$(grep -m1 '^description:' "$W/$name.err" | sed 's/ *\[[0-9]* ms\] *$//')
  if [ -n "$desc" ]; then say "$name: $desc"; fi
  say "$name: Lean's exit code $st; $((t1 - t0)) s; $(wc -l < "$W/$name.err") lines in $name.err;" \
    "panic lines $panics; the unchanged files: ${first:-no ANSWER line};" \
    "controls seen $seen; as expected $asexp of $want; not as expected $notexp; refused $refused"
  played=$((played + want)); asked=$((asked + 1))
  rec "call $name lean/Ask/$file $(sum "$top/lean/Ask/$file"): exit $st; seconds $((t1 - t0));" \
    "lines $(wc -l < "$W/$name.err"); panic lines $panics; out of memory lines $nomem"
  rec "ran $name: ${desc:-the run prints no line \"description:\"}"
  rec "verdict $name: ${first:-no ANSWER line}"
  # Each control's two lines: what was changed and what was expected; what said NO.
  grep '^\(=== \)\{0,1\}CONTROL [A-Za-z0-9_-]*:' "$W/$name.err" |
    sed -e 's/ *\[[0-9]* ms\] *$//' -e 's/^=== //' -e "s/^/control $name: /" >> "$W/RECORD.txt"
  # This call's own verdict.
  # "touch": whether what went amiss touches the run of the unchanged files, or controls alone.
  v=ok; touch=run
  case "$first" in ANSWER*': yes') yes=1 ;; *) yes=0 ;; esac
  if [ "$panics" -ne 0 ]; then v=wrong
  elif [ "$nomem" -ne 0 ]; then v=unended
  elif [ "$st" -eq 0 ] || [ "$st" -eq 1 ]; then
    # Lean's exit code alone is not believed: the yes and every control must stand in the print.
    # Each run's first line begins "read:". Without it Lean stopped before it ran anything.
    if ! grep -q '^read:' "$W/$name.err"; then v=untested
    elif [ "$yes" -eq 0 ]; then
      case "$first" in ANSWER*': NO') v=wrong ;; *) v=untested ;; esac
    elif [ "$notexp" -ne 0 ]; then v=wrong; touch=controls
    elif [ "$asexp" -ne "$want" ] || [ "$seen" -ne "$want" ]; then v=untested; touch=controls
    elif [ "$st" -eq 1 ]; then v=wrong
    fi
  elif [ "$st" -ge 125 ] && [ "$st" -le 127 ]; then v=untested
  else v=unended; fi
  case $v in wrong) wrong=1 ;; unended) unended=1 ;; untested) untested=1 ;; esac
  refused_all=$((refused_all + refused))
  # The calls whose names begin with "head-" feed the four lines at the end of the print. The
  # first of them that prints "own steps:" gives lines 1 and 2. The four lines are quoted without
  # the END line: so they say themselves where a call that feeds them did not end as expected.
  case "$name" in head-*)
    if [ "$v" != ok ]; then
      if [ "$touch" = run ]; then
        if [ "$panics" -ne 0 ]; then r="Lean panicked"
        elif [ "$nomem" -ne 0 ]; then r="Lean ran out of memory"
        elif [ "$v" = unended ]; then r="the run was stopped before its end"
        elif [ "$yes" -eq 0 ]; then r="the unchanged files got no yes"
        else r="Lean's exit code was $st"; fi
        head_bad2=" [THE RUN DID NOT END AS EXPECTED: $r]"; head_bad3=$head_bad2
      elif [ "$notexp" -ne 0 ]; then
        head_bad3=" [altered inputs that did not end as expected: $notexp]"
      elif [ "$seen" -ne "$want" ]; then
        head_bad3=" [the run played $seen altered inputs where $want were to be played]"
      else
        head_bad3=" [altered inputs of which the run does not say AS EXPECTED: $((seen - asexp))]"
      fi
    fi
    # Which controls were not refused, by their numbers or names.
    sed -n 's/^CONTROL \([A-Za-z0-9_-]*\):.*/\1/p' "$W/$name.err" | sort -u > "$W/$name.ids"
    grep '^CONTROL [A-Za-z0-9_-]*:.*said no: \[[^]]' "$W/$name.err" |
      grep -v -e ALARM -e ': answer yes' |
      sed -n 's/^CONTROL \([A-Za-z0-9_-]*\):.*/\1/p' | sort -u > "$W/$name.refused"
    head_not="$head_not $(comm -23 "$W/$name.ids" "$W/$name.refused" | tr '\n' ' ')"
    own=$(sed -n 's/^own steps: *\([0-9]* of [0-9]*\).*/\1/p' "$W/$name.err" | head -n 1)
    if [ -n "$own" ] && [ -z "$head_own" ]; then
      head_own=$own; head_s=$((t1 - t0))
      ownline=$(grep -m1 '^own steps:' "$W/$name.err")
      head_anew=$(printf '%s\n' "$ownline" | sed -n 's/.*written anew: *\([0-9]*\).*/\1/p')
      head_alone=$(printf '%s\n' "$ownline" | sed -n 's/.*alone: *\([0-9]*\).*/\1/p')
      head_pure=$(printf '%s\n' "$ownline" | sed -n 's/.*pure: *\([0-9]*\).*/\1/p')
      # A file that plays controls behind the unchanged files says itself how long these took.
      ms=$(sed -n 's/^the unchanged files took: *\([0-9][0-9]*\) ms.*/\1/p' "$W/$name.err" |
        head -n 1)
      if [ -n "$ms" ] && [ "$(((ms + 500) / 1000))" -le "$head_s" ]; then
        head_s=$(((ms + 500) / 1000))
      elif [ "$want" -ne 0 ]; then head_s="at most $head_s"; fi
      case "$first" in ANSWER*': yes') head_ans=yes ;; ANSWER*': NO') head_ans=no ;; esac
    fi
    # "of b": the controls that the table promises or, if the run played more, those.
    if [ "$seen" -gt "$want" ]; then head_b=$((head_b + seen)); else head_b=$((head_b + want)); fi
    head_a=$((head_a + refused))
    bystep=$(grep '^CONTROL [A-Za-z0-9_-]*:.*said no: \[\([^]]*, \)\{0,1\}step ' "$W/$name.err" |
      grep -v -e ALARM -e ': answer yes' | ids)
    head_c=$((head_c + bystep))
    # Behind the steps: the list begins with the words of the table's line "#four-behind: ...".
    if [ -n "$behind_words" ]; then
      behind=$(grep '^CONTROL [A-Za-z0-9_-]*:' "$W/$name.err" |
        grep -v -e ALARM -e ': answer yes' |
        grep -F "said no: [$behind_words" | ids)
      head_d=$((head_d + behind))
    fi ;;
  esac
}
# A round: up to ACCEPTS_JOBS calls are started five seconds apart, then each is waited for.
round=; n=0
flush() {
  for r in $round; do
    IFS=, read -r pid name file want <<EOF
$r
EOF
    wait "$pid"; judge "$name" "$file" "$want" "$?"
  done
  round=; n=0
}
while read -r _ _ name file one want; do
  [ "$n" -eq 0 ] || sleep 5
  start "$name" "$file" "$one"
  round="$round $!,$name,$file,$want"; n=$((n + 1))
  [ "$n" -lt "$jobs" ] || flush
done <<EOF
$calls
EOF
flush

# The four lines, where the table names calls that feed them. Their words stand in the table, in
# lines "#four-1: ..." to "#four-4: ...", beside the calls that they describe; the script fills in
# {k} {n} {anew} {pure} {alone} {which} {answer} {seconds} {refused} {played} {bystep} {behind}
# {note}.
if [ "$heads" -eq 1 ]; then
  [ -n "$head_own" ] || untested=1 # one of these calls must print the line "own steps:"
  # {note}: nothing where every control was refused. Where just the control that the table's line
  # "#four-note <control>: ..." names was not refused: that line's words. Else the controls' names.
  set -f
  # shellcheck disable=SC2086
  set -- $head_not
  set +f
  note=
  if [ $# -gt 0 ]; then
    told=$(sed -n 's/^#four-note \([A-Za-z0-9_-]*\): *.*/\1/p' "$sheets" | head -n 1)
    if [ $# -eq 1 ] && [ "$1" = "$told" ]; then
      note="; $(sed -n 's/^#four-note [A-Za-z0-9_-]*: *//p' "$sheets" | head -n 1)"
    else note="; NOT REFUSED: $*"; fi
  fi
  say "FOUR LINES:"
  for i in 1 2 3 4; do
    l=$(sed -n "s/^#four-$i: *//p" "$sheets" | head -n 1)
    l=${l//'{k}'/"${head_own%% *}"}; l=${l//'{n}'/"${head_own##* }"}
    l=${l//'{anew}'/"${head_anew:-?}"}; l=${l//'{alone}'/"${head_alone:-?}"}
    l=${l//'{which}'/"$which"}; l=${l//'{answer}'/"${head_ans:-no answer}"}
    l=${l//'{seconds}'/"${head_s:-no}"}; l=${l//'{refused}'/"$head_a"}
    l=${l//'{played}'/"$head_b"}; l=${l//'{bystep}'/"$head_c"}; l=${l//'{note}'/"$note"}
    l=${l//'{behind}'/"$head_d"}; l=${l//'{pure}'/"${head_pure:-?}"}
    case $i in 2) l=$l$head_bad2 ;; 3) l=$l$head_bad3 ;; esac
    [ -n "$head_own" ] || case $i in 1) l="$l [THE RUN PRINTED NO LINE \"own steps:\"]" ;; esac
    say "($i) $l"; rec "four ($i) $l"
  done
fi

see="see the lines above and the files .out and .err in $W"
[ "$asked" -eq "$(printf '%s\n' "$calls" | wc -l)" ] ||
  stop 2 "not every call of Lean was judged: $see"
[ "$wrong" -eq 0 ] || stop 1 "A WRONG ANSWER: $see"
[ "$unended" -eq 0 ] || stop 3 "a run did not end: $see"
[ "$untested" -eq 0 ] || stop 2 "a call of Lean tested nothing, or less than it should: $see"
stop 0 "all ended as expected: calls of Lean $asked; controls played $played, refused" \
  "$refused_all." \
  "A yes here is the result of a test on stored files, not a verification (lean/Ask/README.md)"
