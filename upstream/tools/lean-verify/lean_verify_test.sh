#!/bin/bash
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Tests tools/lean-verify/lean_verify.sh without Lean: its refusals, what it hands to Lean, and how
# it reads Lean's status and lines; with it read_claim.py, which runs as it is; and check_list.sh.
# The test copies them into a small stand-in tree, in which stand-ins for lake and for the parser
# play each ending; in three calls a stand-in for grep or sed stages a moment that cannot be timed
# with the real one, and one call runs below a small Python parent that adopts orphans. It also looks
# whether the real LeanVerify.lean and, where it lies in the tree, Program.lean still hold the
# statuses and the fixed words that the script and read_claim.py expect.
# No proof is judged and no Lean code runs: a pass says nothing about the verifier, and nothing on
# whether the real lake, lean and parser behave as the stand-ins do.
#
# usage: tools/lean-verify/lean_verify_test.sh [NEW_WORK_DIR]
#   Started by the runner; by hand: below a directory of your own.
#   NEW_WORK_DIR must not exist (default: a fresh directory that mktemp makes). The test writes only
#   there, if no other account can change the path to it: TMPDIR places the work directories of the
#   script under test below it. It removes nothing: delete the work directory when you are done.
# Needs: bash, GNU coreutils, diffutils, findutils, grep and sed, util-linux's prlimit, python3. Run
# it as an ordinary user, not with sudo. Two calls are left out either way: as root those on an
# unreadable file, since root can read every file; else those on a tree of another account's, since
# only root can make one.
# Inside its work directory it renames files, and makes a FIFO, two sparse files of 256 MiB, a file
# that nobody may read, symbolic links, and names that hold a blank, a newline or a leading "-"; as
# root it gives one directory there to another account and takes it back.
# The test itself sends no signal. In twenty calls "timeout" sends one to the script under test, which
# hands it on to its stand-ins and to what these have started. In four calls the stand-in for lake
# aborts itself, as Lean does at a panic: your system's crash reporter may log that; no core file is
# written.
# Exit code 0: the last line ends "all as expected". 1: a call did not end as expected, a step of the
# test's own failed, or the number of calls is not the one that the test's end names. 2: a wrong
# call, or the stand-in tree could not be made.
set -u
umask 077
LC_ALL=C
export LC_ALL
# The variables by which this test steers its stand-ins are not taken from the caller's environment:
# PLAY_SWAP names a file that a stand-in overwrites.
unset CDPATH POSIXLY_CORRECT AT PLAY_SWAP PLAY_PARSER PLAY_LEAF PLAY_FITS PLAY_BUILD PLAY_LINK PLAY_WALK PLAY_CLAIM
[ $# -le 1 ] || { echo "usage: tools/lean-verify/lean_verify_test.sh [NEW_WORK_DIR]" >&2; exit 2; }
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P) || exit 2
# As in the two scripts: else this test could copy, and test, another tree's scripts than the reader thinks.
[ "$here/lean_verify_test.sh" -ef "${BASH_SOURCE[0]}" ] ||
  { echo "lean_verify_test.sh: call it by its own path, without \"..\" and not through a link" >&2; exit 2; }
if [ $# -eq 1 ]; then
  # "$(cd ..)" drops newlines at a name's end, and "cd -- -" enters the directory before: either way W
  # would be another directory than the one just made.
  case $1 in
    - | *$'\n'*) echo "lean_verify_test.sh: name a directory without a newline, and not \"-\"" >&2; exit 2 ;;
  esac
  mkdir -- "$1" || { echo "lean_verify_test.sh: name a directory that does not exist yet" >&2; exit 2; }
  W=$(cd -- "$1" && pwd -P) || exit 2
  # .. as it would with ".." behind a link: mkdir follows the link, "cd" cuts both away as text.
  [ "$W" -ef "$1" ] ||
    { echo "lean_verify_test.sh: $1 leads elsewhere than it reads (it was made, and is left empty): name a path without \"..\"" >&2; exit 2; }
else
  W=$(mktemp -d) || exit 2
  W=$(cd -- "$W" && pwd -P) || exit 2
fi
echo "work directory: $W"

# <numbers>: each as a word of four bytes, the lowest first.
words() {
  local w
  for w in "$@"; do
    # shellcheck disable=SC2059 # the format is made here, of hex digits alone
    printf "$(printf '\\x%02x\\x%02x\\x%02x\\x%02x' $((w & 255)) $((w >> 8 & 255)) $((w >> 16 & 255)) $((w >> 24 & 255)))"
  done
}
sum() { sha256sum < "$1" | cut -d' ' -f1; }

# The stand-in tree; a blank stands in its path. Its request has T = 4 tokens, then r = 2 and e = 4:
# a prompt of three tokens and an output of two, the last of them the token file's.
T="$W/a tree"
L=$T/lean/.lake
I=$W/in
h=00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff
{ mkdir -p -- "$T/tools/lean-verify" "$T/tools/parser" "$L/packages/pkg" "$W/bin" "$W/tmp" "$I/config" "$I/ivk" \
  "$T/examples/full-size/config" "$L/build/lib/lean/Johnson/PVR" "$W/calls" &&
  cp -- "$here/lean_verify.sh" "$here/check_list.sh" "$here/read_claim.py" "$T/tools/lean-verify/" &&
  echo '-- a stand-in' > "$T/tools/lean-verify/LeanVerify.lean" &&
  echo '-- a stand-in' > "$T/tools/lean-verify/LeanVerifyWalk.lean" &&
  printf '%s\n' '# a stand-in list' 'Ours.First' '' "Pkg.Second'" > "$W/list" &&
  cp -- "$W/list" "$T/tools/lean-verify/native-modules.txt" &&
  : > "$L/build/lib/lean/Johnson/PVR/ProgramGuard.olean" &&
  mkdir -p -- "$L/build/ir/Ours" "$L/packages/pkg/.lake/build/ir/Pkg" &&
  : > "$L/build/ir/Ours/First.c" && : > "$L/packages/pkg/.lake/build/ir/Pkg/Second'.c" &&
  for f in config/layout.txt config/KL.bin proof; do echo "$f" > "$I/$f"; done &&
  printf '%s\n' 'lt 2' 'V 100' 'C 256' 'theta 256' 'cap 0' 's 15' 'sh 8' 'sT 4' 'F 24' 'Fx 1' > "$I/config/policy.txt" &&
  words 1 2 3 4 2 4 > "$I/req" && words 5 > "$I/y" && echo "$h" > "$I/id" &&
  cp -- "$I/config/policy.txt" "$I/config/layout.txt" "$I/config/KL.bin" "$T/examples/full-size/config/" &&
  (cd -- "$I" && sha256sum config/policy.txt config/layout.txt config/KL.bin req y proof id) > "$W/sums-at-the-start"; } ||
  { echo "lean_verify_test.sh: the stand-in tree could not be made" >&2; exit 2; }

cat > "$T/tools/parser/parse_files.sh" << 'EOF'
#!/bin/bash
# A stand-in for the parser. PLAY_PARSER says how it ends.
# Sleeps in a child of its own, as the parser waits for a compiler: both process numbers are recorded.
sleeps() { sleep 60 & printf '%s\n' $$ $! > "$PLAY_RECORD.pid"; wait; exit 99; }
printf '%s\n' "$@" > "$PLAY_RECORD.parser"
mkdir -p -- "$5/parsed" || exit 3
d=$(cd -- "$5" && pwd -P) || exit 3 # the path without links, as the parser prints it
[ -z "${PLAY_SWAP:-}" ] || echo swapped > "$PLAY_SWAP" # a file of the caller's changes while the parser runs
for f in key "leaf-${PLAY_LEAF:-0}" proof; do echo "$f" > "$d/parsed/$f.parsed.bin"; echo "$f seeds" > "$d/parsed/$f.parsed.bin.seeds"; done
case ${PLAY_PARSER:-ok} in
  ok) echo "PARSED AND READ BACK: $d/parsed" ;;
  refuses) exit 1 ;;
  tool) exit 3 ;;
  line-then-fails) echo "PARSED AND READ BACK: $d/parsed"; exit 1 ;;
  sleeps) sleeps ;;
  other-line) echo "PARSED AND READ BACK: $d/parsed " ;;
  silent) ;;
  two-leaves) echo x > "$d/parsed/leaf-7.parsed.bin"; echo "PARSED AND READ BACK: $d/parsed" ;;
  no-leaf) mv -- "$d/parsed/leaf-0.parsed.bin" "$d/parsed/other"; echo "PARSED AND READ BACK: $d/parsed" ;;
esac
EOF

cat > "$W/bin/lake" << 'EOF'
#!/bin/bash
# A stand-in for lake. PLAY_FITS, PLAY_BUILD, PLAY_LINK and PLAY say how its four kinds of call end.
core=$(ulimit -c)
ulimit -c 0
accept='VERDICT: ACCEPT'
reject='VERDICT: REJECT'
sum() { sha256sum < "$1" | cut -d' ' -f1; }
# Sleeps in a child of its own, as Lake waits for Lean or a compiler: both process numbers are recorded.
sleeps() { sleep 60 & printf '%s\n' $$ $! > "$PLAY_RECORD.pid"; wait; exit 99; }
# <lines>: that many lines of 50 bytes.
flood() { yes ' BK IS READ: a string behind the escape is hashed' | command head -n "$1"; }
case "$1 ${2:-}" in
  'env lean')
    case ${!#} in
      */LeanVerifyWalk.lean) # the walk, for check_list.sh. PLAY_WALK says how it ends.
        { echo "cwd $PWD"; echo "panic ${LEAN_ABORT_ON_PANIC:-unset}"; echo "core $core"; echo "umask $(umask)"
          echo "paths ${LEAN_PATH:-unset} ${LEAN_SRC_PATH:-unset}"; echo "locale ${LC_ALL:-unset}"
          echo "posix ${POSIXLY_CORRECT:-unset}"; } > "$PLAY_RECORD"
        case ${PLAY_WALK:-same} in
          sleeps) sleeps ;;
          same) printf '%s\n' Ours.First "Pkg.Second'" > "$LEAN_VERIFY_LIST" ;;
          one-more) printf '%s\n' Ours.First Ours.Other "Pkg.Second'" > "$LEAN_VERIFY_LIST" ;;
          one-less) printf '%s\n' Ours.First > "$LEAN_VERIFY_LIST" ;;
          other-order) printf '%s\n' "Pkg.Second'" Ours.First > "$LEAN_VERIFY_LIST" ;;
          empty) : > "$LEAN_VERIFY_LIST" ;;
          silent) ;;
          fails) printf '%s\n' Ours.First "Pkg.Second'" > "$LEAN_VERIFY_LIST"; exit 1 ;;
        esac
        echo 'declarations reached: ..'
        exit 0 ;;
    esac
    mapfile -t a <<< "${LEAN_VERIFY_ARGS:-}"
    { printf 'arg %s\n' "$@"
      echo "panic ${LEAN_ABORT_ON_PANIC:-unset}"; echo "cwd $PWD"; echo "arguments ${#a[@]}"; echo "core $core"
      echo "posix ${POSIXLY_CORRECT:-unset}"; echo "file size $(ulimit -f)"
      echo "paths ${LEAN_PATH:-unset} ${LEAN_SRC_PATH:-unset}"; echo "locale ${LC_ALL:-unset}"
      for f in policy.txt layout.txt KL.bin; do echo "claim config/$f $(sum "${a[0]:-}/$f")"; done
      echo "claim ID.txt $(sum "${a[1]:-}")"; echo "claim example.req $(sum "${a[2]:-}")"; echo "claim example.y $(sum "${a[3]:-}")"
      echo "parsed key $(cat -- "${a[4]:-}")"; echo "parsed leaf $(cat -- "${a[5]:-}")"; echo "parsed proof $(cat -- "${a[6]:-}")"
      echo "parsed key $(cat -- "${a[4]:-}.seeds")"; echo "parsed leaf $(cat -- "${a[5]:-}.seeds")"
    } > "$PLAY_RECORD" 2>&1
    # What Lean prints before its last line. The claim's lines are played by the second reader itself,
    # and PLAY_CLAIM says how they are then bent.
    head() {
      echo 'read: the parameter file: 1 bytes'
      python3 -I -B "$PLAY_READER" "${a[0]}" "${a[2]}" "${a[3]}" "${a[1]}" | case ${PLAY_CLAIM:-as-read} in
        as-read) cat ;;
        other-token) sed 's/^\(claim: output, .*\)5\]$/\16]/' ;;
        other-id) sed 's/^\(claim: commitment ID: \)0/\11/' ;;
        other-policy) sed 's/theta 256/theta 255/' ;;
        a-line-less) sed '/^claim: prompt/d' ;;
        a-line-more) sed 's/^\(claim: prompt.*\)$/\1\n\1/' ;;
        other-order) sed -n '1!G;h;$p' ;;
        none) : ;;
      esac
      echo 'steps: buildOk yes; ..'
    }
    case $PLAY in
      accept) { head; echo "$accept"; } >&2; exit 10 ;;
      accept-status-*) { head; echo "$accept"; } >&2; exit "${PLAY#accept-status-}" ;;
      accept-on-stdout) head >&2; echo "$accept"; exit 10 ;;
      all-on-stdout) head; echo "$accept"; exit 10 ;;
      accept-blank) { head; echo "$accept "; } >&2; exit 10 ;;
      accept-cr) { head; printf '%s\r\n' "$accept"; } >&2; exit 10 ;;
      accept-nul) { head; printf '%s\0\n' "$accept"; } >&2; exit 10 ;;
      accept-cut) { head; printf '%s' "$accept"; } >&2; exit 10 ;;
      accept-short) { head; echo "${accept%T}"; } >&2; exit 10 ;;
      accept-lower) { head; echo 'VERDICT: accept'; } >&2; exit 10 ;;
      accept-indented) { head; echo " $accept"; } >&2; exit 10 ;;
      accept-inside) { head; echo "public files: the acceptance policy, by the parameter file's own lines: [C $accept]"; } >&2; exit 10 ;;
      accept-not-last) { head; echo "$accept"; echo 'one line more'; } >&2; exit 10 ;;
      accept-then-empty) { head; echo "$accept"; echo; } >&2; exit 10 ;;
      accept-twice) { head; echo "$accept"; echo "$accept"; } >&2; exit 10 ;;
      reject-then-accept) { head; echo "$reject"; echo "$accept"; } >&2; exit 10 ;;
      none-then-accept) { head; echo 'NO VERDICT: the verifier gave none'; echo "$accept"; } >&2; exit 10 ;;
      rejected-then-accept) { head; echo 'VERDICT: REJECTED'; echo "$accept"; } >&2; exit 10 ;;
      accept-then-reject) { head; echo "$accept"; echo "$reject"; } >&2; exit 11 ;;
      reject) { head; echo "$reject"; } >&2; exit 11 ;;
      reject-status-*) { head; echo "$reject"; } >&2; exit "${PLAY#reject-status-}" ;;
      reject-blank) { head; echo "$reject "; } >&2; exit 11 ;;
      reject-on-stdout) head >&2; echo "$reject"; exit 11 ;;
      reject-not-last) { head; echo "$reject"; echo 'one line more'; } >&2; exit 11 ;;
      no-verdict) { head; echo 'NO VERDICT: the verifier gave none'; } >&2; exit 12 ;;
      no-verdict-status-*) { head; echo 'NO VERDICT: the verifier gave none'; } >&2; exit "${PLAY#no-verdict-status-}" ;;
      silent-*) exit "${PLAY#silent-}" ;;
      error) echo 'LeanVerify.lean:1:0: error: unknown constant'; exit 1 ;; # a made-up line of Lean's
      aborted) { head; echo 'PANIC at ..'; } >&2; exec python3 -c 'import os; os.abort()' ;;
      aborted-after-accept) { head; echo "$accept"; } >&2; exec python3 -c 'import os; os.abort()' ;;
      sleeps) head >&2; sleeps ;;
      # Three million bytes, then an accept's end. As with real Lean, the system ends whoever passes the bound.
      floods) { head; flood 60000; echo "$accept"; } >&2; exit 10 ;;
      # .. a stand-in that passes over the system's signal and goes on to its end.
      floods-and-goes-on) trap '' XFSZ; { head; flood 60000; echo "$accept"; } >&2; exit 10 ;;
      floods-standard-output) trap '' XFSZ; flood 60000; { head; echo "$accept"; } >&2; exit 10 ;;
      much) { head; flood 17000; echo "$accept"; } >&2; exit 10 ;; # under the bound
      # An accept's line that ends on the bound's last byte; a reject's line behind it is lost.
      accept-at-the-bound-then-more) trap '' XFSZ
        h=$(head)
        { printf '%s\n' "$h"; printf '%*s\n' $((1048576 - ${#h} - ${#accept} - 3)) '' | tr ' ' x
          echo "$accept"; echo "$reject"; echo 'one line more'; } >&2
        exit 10 ;;
    esac
    exit 99 ;;
  'env leanc')
    [ "${PLAY_LINK:-ok}" != sleeps ] || sleeps
    [ "${PLAY_LINK:-ok}" = ok ] || exit 1
    while [ $# -gt 0 ]; do if [ "$1" = -o ]; then echo library > "$2"; fi; shift; done
    exit 0 ;;
  'build --no-build')
    shift 2
    printf '%s\n' "$@" > "$PLAY_RECORD.fits"
    [ "${PLAY_FITS:-ok}" != sleeps ] || sleeps
    [ "${PLAY_FITS:-ok}" = ok ] || exit 3
    exit 0 ;;
  'build '*)
    shift
    printf '%s\n' "$@" > "$PLAY_RECORD.build"
    umask > "$PLAY_RECORD.umask"
    [ "${PLAY_BUILD:-ok}" != fails ] || exit 1
    [ "${PLAY_BUILD:-ok}" != sleeps ] || sleeps
    for t in "$@"; do
      case $t in +*:c.o.export) ;; *) continue ;; esac
      m=${t#+} m=${m%:c.o.export} p=${m//.//} b=.lake/build
      case $m in Pkg.*) b=.lake/packages/pkg/.lake/build ;; esac
      mkdir -p -- "$b/ir/${p%/*}" "$b/lib/lean/${p%/*}"
      touch -d 2001-01-01 -- "$b/ir/$p.c" "$b/lib/lean/$p.olean"
      touch -d 2002-01-01 -- "$b/ir/$p.c.o.export"
      case ${PLAY_BUILD:-ok}:$m in
        no-object:Pkg.*) mv -- "$b/ir/$p.c.o.export" "$b/ir/$p.moved" ;;
        old-c:Pkg.*) touch -d 2003-01-01 -- "$b/ir/$p.c" ;;
        old-olean:Ours.*) touch -d 2003-01-01 -- "$b/lib/lean/$p.olean" ;;
        same-time:Ours.*) touch -d 2002-01-01 -- "$b/ir/$p.c" ;;
        no-c:Ours.*) mv -- "$b/ir/$p.c" "$b/ir/$p.moved" ;;
        no-olean:Pkg.*) mv -- "$b/lib/lean/$p.olean" "$b/lib/lean/$p.moved" ;;
      esac
    done
    exit 0 ;;
esac
exit 98
EOF
chmod +x -- "$T/tools/parser/parse_files.sh" "$W/bin/lake" "$T/tools/lean-verify/lean_verify.sh" "$T/tools/lean-verify/check_list.sh" || exit 2

S=$T/tools/lean-verify/lean_verify.sh
six=("$I/config" "$I/ivk" "$I/req" "$I/y" "$I/proof" "$I/id")
calls=0 off=0
export TMPDIR=$W/tmp PATH=$W/bin:$PATH PLAY_READER=$T/tools/lean-verify/read_claim.py
wrong() {
  off=$((off + 1))
  echo "NOT AS EXPECTED: $*"
}
# <command ..>: a step of the test's own. If it fails, the calls that need it prove nothing.
step() { "$@" || wrong "a step of the test's own failed: $*"; }
# <name> <exit code> <the whole of standard output, without its newline; "" for none> <command ..>
# AT: the directory in which the command is started (default: the work directory).
call() {
  local name=$1 code=$2 line=$3 got
  shift 3
  calls=$((calls + 1))
  export PLAY_RECORD=$W/calls/$name.record
  (cd -- "${AT:-$W}" && exec "$@") > "$W/calls/$name.out" 2> "$W/calls/$name.err" < /dev/null
  got=$?
  if [ "$got" = "$code" ] && { if [ -n "$line" ]; then printf '%s\n' "$line"; fi; } | cmp -s -- - "$W/calls/$name.out"; then
    return 0
  fi
  wrong "$name: exit code $got where $code was due, or another standard output: see $W/calls/$name.out and .err"
}
# <name> <fixed string>: the call's error stream must hold it.
tells() { grep -a -q -F -e "$2" -- "$W/calls/$1.err" || wrong "$1: the error stream lacks: $2"; }
# <name> <whole line>: what the stand-in for Lean recorded must hold it.
got_line() { grep -a -q -x -F -e "$2" -- "$W/calls/$1.record" || wrong "$1: Lean was not handed: $2"; }
no_lean() { [ ! -e "$W/calls/$1.record" ] || wrong "$1: Lean was called"; }
# <name> <the record's ending> <how many> <who>: that many processes wrote their numbers into that record
# of the call, and all have ended.
ended() {
  local p n=0 ps=()
  [ ! -f "$W/calls/$1.record.$2" ] || mapfile -t ps < "$W/calls/$1.record.$2"
  [ ${#ps[@]} -eq "$3" ] || { wrong "$1: $4: not reached"; return; }
  for p in "${ps[@]}"; do
    while [ -d "/proc/$p" ] && ! grep -q '^State:[[:space:]]*Z' "/proc/$p/status" 2> /dev/null && [ "$n" -lt 50 ]; do
      sleep 0.1
      n=$((n + 1))
    done
  done
  [ "$n" -lt 50 ] || wrong "$1: $4: runs on behind the script's end"
}
# <name>: the stand-in that slept in the call, and the child that it had started, have both ended.
gone() { ended "$1" pid 2 "the sleeping stand-in with its child"; }
# <name> <command ..>: due "NO VERDICT" and code 2, and Lean is not called.
refused() {
  local name=$1
  shift
  call "$name" 2 'NO VERDICT' "$@"
  no_lean "$name"
}
N='NO VERDICT' A='VERDICT: ACCEPT (exact match)'

# 1. Wrong calls and files that will not do.
export PLAY=accept
refused no-argument "$S"; tells no-argument 'usage: '
refused five "$S" "${six[@]:0:5}"; tells five 'usage: '
refused seven "$S" "${six[@]}" more
refused option-and-five "$S" --no-native "${six[@]:0:5}"
refused option-last "$S" "${six[@]}" --no-native
refused other-option "$S" --native "${six[@]}"
for i in 0 1 2 3 4 5; do
  a=("${six[@]}"); a[i]=$I/absent
  refused "missing-$i" "$S" "${a[@]}"
  a[i]=''
  refused "empty-name-$i" "$S" "${a[@]}"
done
for i in 2 3 4 5; do
  a=("${six[@]}"); a[i]=$I/ivk
  refused "directory-$i" "$S" "${a[@]}"
done
refused file-as-config "$S" "$I/req" "${six[@]:1}"
refused file-as-keys "$S" "$I/config" "$I/req" "${six[@]:2}"
step mkfifo -- "$I/fifo"; refused fifo "$S" "${six[@]:0:3}" "$I/fifo" "${six[@]:4}"
step truncate -s 268435457 -- "$I/large"
refused proof-too-large "$S" "${six[@]:0:4}" "$I/large" "$I/id"
refused token-too-large "$S" "${six[@]:0:3}" "$I/large" "${six[@]:4}"
step truncate -s 268435456 -- "$I/at-the-limit"; call at-the-limit 0 "$A" "$S" "${six[@]:0:4}" "$I/at-the-limit" "$I/id"
if [ "$(id -u)" != 0 ]; then
  step cp -- "$I/id" "$I/closed"; step chmod 000 -- "$I/closed"; refused unreadable "$S" "${six[@]:0:5}" "$I/closed"
  refused unreadable-proof "$S" "${six[@]:0:4}" "$I/closed" "$I/id"
fi
step mkdir -- "$I/config-short"; step cp -- "$I/config/policy.txt" "$I/config/layout.txt" "$I/config-short/"
refused config-lacks-a-file "$S" "$I/config-short" "${six[@]:1}"
# A name that holds a newline: a line like a verdict in it may reach the error stream alone; and a
# newline at a name's end must not lead to the file of the name without it.
nl=$'\n'
refused name-poses-missing "$S" "${six[@]:0:5}" "$I/x${nl}VERDICT: ACCEPT"
step cp -- "$I/id" "$I/x${nl}VERDICT: ACCEPT"; refused name-poses "$S" "${six[@]:0:5}" "$I/x${nl}VERDICT: ACCEPT"
step cp -- "$I/id" "$I/id${nl}"; refused name-ends-in-a-newline "$S" "${six[@]:0:5}" "$I/id${nl}"
step cp -- "$I/id" "$I/-n"; AT=$I call name-like-an-option 0 "$A" "$S" "${six[@]:0:5}" -n
step cp -- "$I/id" "$I/an id"; call name-with-a-blank 0 "$A" "$S" "${six[@]:0:5}" "$I/an id"
AT=$I call relative-names 0 "$A" "$S" config ivk req y proof id
got_line relative-names "claim ID.txt $(sum "$I/id")"
# An argument without a newline that is a link to a name which ends in one.
step ln -s -- "$I/id${nl}" "$I/link-to-a-newline"; refused link-to-a-name-with-a-newline "$S" "${six[@]:0:5}" "$I/link-to-a-newline"
tells link-to-a-name-with-a-newline 'leads to a name that ends in a newline'
# .. or to a name below a folder whose name holds one: such a path must not reach the error stream.
step mkdir -- "$I/x${nl}VERDICT: ACCEPT (exact match)"; step cp -- "$I/id" "$I/x${nl}VERDICT: ACCEPT (exact match)/id"
step ln -s -- "$I/x${nl}VERDICT: ACCEPT (exact match)/id" "$I/link-into-a-newline"
refused link-into-a-name-with-a-newline "$S" "${six[@]:0:5}" "$I/link-into-a-newline"
tells link-into-a-name-with-a-newline 'leads to a name that holds a newline'
step ln -s -- "$I/id" "$I/link-to-the-id"; call link-to-a-file 0 "$A" "$S" "${six[@]:0:5}" "$I/link-to-the-id"
# The script is called through a link that lies elsewhere: it finds no tree around it.
step ln -s -- "$S" "$W/bin/linked.sh"; refused called-through-a-link "$W/bin/linked.sh" "${six[@]}"
# A function of the caller's environment under the name of a command that the script uses.
refused function-from-the-environment env 'BASH_FUNC_echo%%=() { builtin echo "VERDICT: ACCEPT"; }' "$S"

# 2. The format page's five tests on the caller's two files, and the ID file's form, as read_claim.py
# makes them.
claim() { # <name> <request file> <token file> <ID file>
  refused "$1" "$S" "$I/config" "$I/ivk" "$2" "$3" "$I/proof" "$4"
  tells "$1" 'read_claim.py refuses the request, token, ID or parameter file'
}
C=$I/claims
step mkdir -- "$C"
{ words 1 2 3 4 2 4; printf x; } > "$C/ragged.req"; claim request-ragged "$C/ragged.req" "$I/y" "$I/id"
: > "$C/empty"; claim request-empty "$C/empty" "$I/y" "$I/id"; claim token-empty "$I/req" "$C/empty" "$I/id"
{ words 5; printf xyz; } > "$C/ragged.y"; claim token-ragged "$I/req" "$C/ragged.y" "$I/id"
words 1 2 3 2 4 > "$C/short.req"; claim request-short "$C/short.req" "$I/y" "$I/id"
words 1 2 3 4 4 2 4 > "$C/long.req"; claim request-long "$C/long.req" "$I/y" "$I/id"
words 5 5 > "$C/two.y"; claim token-two-words "$I/req" "$C/two.y" "$I/id"
words 1 2 3 4 2 2 > "$C/r-is-e.req"; words 0 > "$C/zero.y"; claim r-is-e "$C/r-is-e.req" "$C/zero.y" "$I/id"
words 1 2 3 4 3 2 > "$C/r-above-e.req"; claim r-above-e "$C/r-above-e.req" "$C/zero.y" "$I/id"
words 1 2 3 4 2 5 > "$C/e-above-T.req"; claim e-above-T "$C/e-above-T.req" "$I/y" "$I/id"
words 1 2 3 4 1 3 > "$C/e-below-T.req"; claim token-of-which-nothing-is-said "$C/e-below-T.req" "$I/y" "$I/id"
words 1 100 3 4 2 4 > "$C/large-token.req"; claim token-not-below-V "$C/large-token.req" "$I/y" "$I/id"
words 100 > "$C/large.y"; claim next-token-not-below-V "$I/req" "$C/large.y" "$I/id"
echo "${h%f}" > "$C/63.id"; claim id-63-digits "$I/req" "$I/y" "$C/63.id"
echo "${h}0" > "$C/65.id"; claim id-65-digits "$I/req" "$I/y" "$C/65.id"
echo "${h%f}F" > "$C/upper.id"; claim id-upper-case "$I/req" "$I/y" "$C/upper.id"
echo "${h%f}g" > "$C/g.id"; claim id-other-sign "$I/req" "$I/y" "$C/g.id"
printf '%s\n\n' "$h" > "$C/two-newlines.id"; claim id-two-newlines "$I/req" "$I/y" "$C/two-newlines.id"
printf '%s\r\n' "$h" > "$C/cr.id"; claim id-cr "$I/req" "$I/y" "$C/cr.id"
printf '%s ' "$h" > "$C/blank.id"; claim id-blank "$I/req" "$I/y" "$C/blank.id"
printf '0x%s\n' "${h%ff}" > "$C/0x.id"; claim id-0x "$I/req" "$I/y" "$C/0x.id"
printf '%s' "$h" > "$C/no-newline.id"; call id-without-newline 0 "$A" "$S" "${six[@]:0:5}" "$C/no-newline.id"
# <name> <lines of a parameter file ..>
policy() {
  local name=$1
  shift
  step mkdir -- "$C/$name"; step cp -- "$I/config/layout.txt" "$I/config/KL.bin" "$C/$name/"
  printf '%s\n' "$@" > "$C/$name/policy.txt"
  refused "$name" "$S" "$C/$name" "${six[@]:1}"; tells "$name" 'read_claim.py refuses the request, token, ID or parameter file'
}
# Each file below is whole but for the one thing that its name tells.
seven=('C 256' 'theta 256' 'cap 0' 's 15' 'sh 8' 'sT 4' 'F 24')
policy policy-lacks-lt 'V 100' "${seven[@]}"
policy policy-lacks-V 'lt 2' "${seven[@]}"
policy policy-lt-twice 'lt 2' 'lt 2' 'V 100' "${seven[@]}"
policy policy-lt-no-number 'lt two' 'V 100' "${seven[@]}"
policy policy-lt-extra-word 'lt 2 3' 'V 100' "${seven[@]}"
policy policy-lt-huge 'lt 99999999999' 'V 100' "${seven[@]}"; tells policy-lt-huge "'lt' is above 26"
policy policy-lt-27 'lt 27' 'V 100' "${seven[@]}"; tells policy-lt-27 "'lt' is above 26"
policy policy-not-ascii $'lt 2\xff' 'V 100' "${seven[@]}"
# A second line of a name, hidden from a reader by words: a control character stands in the name.
policy policy-control-in-a-name 'lt 2' 'V 100' "${seven[@]}" $'C\001 255'; tells policy-control-in-a-name 'holds a control character'
policy policy-lacks-F 'lt 2' 'V 100' "${seven[@]:0:6}"
policy policy-a-name-twice 'lt 2' 'V 100' "${seven[@]}" 'C 1'
policy policy-tab-for-the-blank 'lt 2' 'V 100' "${seven[@]}" $'C\t9'
policy policy-carriage-return 'lt 2' 'V 100' $'C 0\rC 256' "${seven[@]:1}"
policy policy-escape 'lt 2' 'V 100' 'C 256' $'theta 0\033[2K' "${seven[@]:2}"
policy policy-del 'lt 2' 'V 100' "${seven[@]:0:2}" $'cap 0\177' "${seven[@]:3}"
policy policy-leading-zero 'lt 2' 'V 100' "${seven[@]:0:6}" 'F 024'
policy policy-blank-at-the-end 'lt 2' 'V 100' "${seven[@]:0:6}" 'F 24 '
policy policy-no-number 'lt 2' 'V 100' "${seven[@]:0:6}" 'F x'
# e < T with a token file that holds 0: the claim reads.
call e-below-T 0 "$A" "$S" "$I/config" "$I/ivk" "$C/e-below-T.req" "$C/zero.y" "$I/proof" "$I/id"
tells e-below-T 'claim: prompt, 2 tokens: [1, 2]'; tells e-below-T 'claim: output, 2 tokens: [3, 4]'
tells e-below-T 'not part of the claim: tokens of the request behind position e, which go to the verifier with the proof: 0'
words 1 2 3 4 0 1 > "$C/further.req"
call further-tokens 0 "$A" "$S" "$I/config" "$I/ivk" "$C/further.req" "$C/zero.y" "$I/proof" "$I/id"
tells further-tokens 'claim: prompt, 1 tokens: [1]'; tells further-tokens 'claim: output, 1 tokens: [2]'
tells further-tokens 'not part of the claim: tokens of the request behind position e, which go to the verifier with the proof: 2'

# 3. The tree is not ready, or a tool fails.
# <name> <path>: the call while the path is renamed.
without() {
  step mv -- "$2" "$2.away"
  refused "$1" "$S" "${six[@]}"
  step mv -- "$2.away" "$2"
}
without no-packages "$L/packages"
without not-built "$L/build/lib/lean/Johnson/PVR/ProgramGuard.olean"; tells not-built "the library is not built in $T/lean"
without lacks-the-asking-file "$T/tools/lean-verify/LeanVerify.lean"
without lacks-the-list "$T/tools/lean-verify/native-modules.txt"
without lacks-the-reader "$T/tools/lean-verify/read_claim.py"
# <name> <path>: the call while the path is a symbolic link to what lay there.
linked() {
  step mv -- "$2" "$2.real"; step ln -s -- "$2.real" "$2"
  refused "$1" "$S" "${six[@]}"; tells "$1" 'is a symbolic link: refused'
  step mv -T -- "$2" "$W/links-put-aside/$1"; step mv -T -- "$2.real" "$2"
}
step mkdir -- "$W/links-put-aside"
linked lean-is-a-link "$T/lean"
linked lake-directory-is-a-link "$L"
linked packages-is-a-link "$L/packages"
linked build-is-a-link "$L/build"
linked a-package-is-a-link "$L/packages/pkg"
linked a-packages-lake-directory-is-a-link "$L/packages/pkg/.lake"
linked a-packages-build-is-a-link "$L/packages/pkg/.lake/build"
# A link among the packages whose name holds a newline: the refusal must not print that name.
step ln -s -- "$L/packages/pkg" "$L/packages/x${nl}VERDICT: ACCEPT (exact match)"
refused a-package-with-a-newline "$S" "${six[@]}"; tells a-package-with-a-newline 'lean/.lake/packages holds a newline'
call list-a-package-with-a-newline 2 'native-modules.txt: not checked' "$T/tools/lean-verify/check_list.sh"
no_lean list-a-package-with-a-newline; tells list-a-package-with-a-newline 'lean/.lake/packages holds a newline'
step mv -T -- "$L/packages/x${nl}VERDICT: ACCEPT (exact match)" "$W/links-put-aside/a-package-with-a-newline"
# A listed module's .c file is missing: Lake is not told to build.
without no-c-file-of-ours "$L/build/ir/Ours/First.c"; tells no-c-file-of-ours 'no .c file for the module Ours.First'
without no-c-file-of-a-package "$L/packages/pkg/.lake/build/ir/Pkg/Second'.c"
for n in no-c-file-of-ours no-c-file-of-a-package; do
  [ ! -e "$W/calls/$n.record.build" ] || wrong "$n: Lake was told to build"
done
step mv -- "$L/build/ir/Ours/First.c" "$L/build/ir/Ours/First.c.away"
call no-c-file-no-native 0 "$A" "$S" --no-native "${six[@]}"
step mv -- "$L/build/ir/Ours/First.c.away" "$L/build/ir/Ours/First.c"
step mkdir -- "$W/tools-alone"
for t in bash dirname realpath stat mktemp sha256sum; do step ln -s -- "$(command -v "$t")" "$W/tools-alone/"; done
refused no-lake env "PATH=$W/tools-alone" "$S" "${six[@]}"; tells no-lake 'lake is not on the PATH'
step ln -s -- "$W/bin/lake" "$W/tools-alone/"
refused no-python3 env "PATH=$W/tools-alone" "$S" "${six[@]}"; tells no-python3 'python3 is not on the PATH'
step ln -s -- "$(command -v python3)" "$W/tools-alone/"
refused no-timeout env "PATH=$W/tools-alone" "$S" "${six[@]}"; tells no-timeout 'timeout is not on the PATH'
step cp -R -- "$W/tools-alone" "$W/tools-with-timeout"; step ln -s -- "$(command -v timeout)" "$W/tools-with-timeout/"
refused no-prlimit env "PATH=$W/tools-with-timeout" "$S" "${six[@]}"; tells no-prlimit 'prlimit is not on the PATH'
refused no-temporary-directory env "TMPDIR=$W/absent" "$S" "${six[@]}"
step mkdir -- "$W/tmp${nl}x"; refused temporary-directory-with-a-newline env "TMPDIR=$W/tmp${nl}x" "$S" "${six[@]}"
tells temporary-directory-with-a-newline "the work directory's path holds a newline"
# POSIXLY_CORRECT changes how GNU grep reads its arguments.
call posixly-correct 0 "$A" env POSIXLY_CORRECT=1 "$S" "${six[@]}"; got_line posixly-correct 'posix unset'
# CDPATH leads a relative "cd" astray: here to a decoy that has the folder's name and nothing in it.
step mkdir -p -- "$W/decoy/tools/lean-verify"
AT=$T call cdpath 0 "$A" env "CDPATH=$W/decoy" tools/lean-verify/lean_verify.sh "${six[@]}"
# .. also where the script is started as "bash <file>": without the "-p" of its first line, which by
# itself makes bash pass over a CDPATH of the environment.
AT=$T call cdpath-without-p 0 "$A" env "CDPATH=$W/decoy" bash tools/lean-verify/lean_verify.sh "${six[@]}"
# The tree's top directory has a name that ends in a newline, and a whole tree of the same name without
# it lies beside: neither script may work in that other tree.
step mkdir -p -- "$W/twin${nl}/tools"; step cp -R -- "$T/tools/lean-verify" "$W/twin${nl}/tools/"; step cp -R -- "$T" "$W/twin"
refused top-directory-ends-in-a-newline "$W/twin${nl}/tools/lean-verify/lean_verify.sh" "${six[@]}"
tells top-directory-ends-in-a-newline 'is not the one above this script'
[ ! -e "$W/calls/top-directory-ends-in-a-newline.record.fits" ] || wrong "top-directory-ends-in-a-newline: Lake was called"
# A whole tree below a folder whose name holds a newline: its path, which both scripts print, could pose as
# a line of theirs.
step mkdir -- "$W/y${nl}VERDICT: ACCEPT (exact match)"; step cp -R -- "$T" "$W/y${nl}VERDICT: ACCEPT (exact match)/tree"
refused path-holds-a-newline "$W/y${nl}VERDICT: ACCEPT (exact match)/tree/tools/lean-verify/lean_verify.sh" "${six[@]}"
tells path-holds-a-newline "the repository's path holds a newline"
# Standing in the twin, a path with ".." behind a link into the first tree: the system runs the first
# tree's file, while "cd" would lead into the twin's folder.
step ln -s -- "$T/lean" "$W/twin/l"
AT=$W/twin refused path-with-dots-behind-a-link l/../tools/lean-verify/lean_verify.sh "${six[@]}"
tells path-with-dots-behind-a-link 'or by a path in which ".." follows one'
# .. and standing inside that link.
AT=$W/twin/l refused standing-inside-a-link ../tools/lean-verify/lean_verify.sh "${six[@]}"
# lean/.lake belongs to another account, as it does for root under sudo.
if [ "$(id -u)" = 0 ]; then
  step chown 65534 -- "$L"; refused tree-of-another-account "$S" "${six[@]}"; step chown 0 -- "$L"
  tells tree-of-another-account "$L belongs to another account"
  step chown 65534 -- "$L"; call list-tree-of-another-account 2 'native-modules.txt: not checked' "$T/tools/lean-verify/check_list.sh"
  step chown 0 -- "$L"; no_lean list-tree-of-another-account; tells list-tree-of-another-account "$L belongs to another account"
fi
step ln -s -- "$W/tmp" "$W/tmp-by-a-link"; call temporary-directory-by-a-link 0 "$A" env "TMPDIR=$W/tmp-by-a-link" "$S" "${six[@]}"
AT=$W call temporary-directory-relative 0 "$A" env TMPDIR=tmp "$S" "${six[@]}"
step mkdir -- "$W/a tmp"; call temporary-directory-with-a-blank 0 "$A" env "TMPDIR=$W/a tmp" "$S" "${six[@]}"
for p in refuses tool line-then-fails other-line silent two-leaves no-leaf; do
  PLAY_PARSER=$p refused "parser-$p" "$S" "${six[@]}"
done
# The compiled library does not fit the sources: the script says so, and tells Lake to build nothing.
PLAY_FITS=stale refused library-stale "$S" "${six[@]}"; tells library-stale 'build it (tools/lean-verify/README.md, step 1)'
PLAY_FITS=stale refused library-stale-no-native "$S" --no-native "${six[@]}"
for n in library-stale library-stale-no-native; do
  [ ! -e "$W/calls/$n.record.build" ] || wrong "$n: Lake was told to build"
done
for b in fails no-object old-c old-olean same-time no-c no-olean; do
  PLAY_BUILD=$b refused "build-$b" "$S" "${six[@]}"
  [ -e "$W/calls/build-$b.record.build" ] || wrong "build-$b: the call ended before Lake was told to build"
  : > "$L/build/ir/Ours/First.c" # the play no-c has renamed it
done
PLAY_LINK=fails refused link-fails "$S" "${six[@]}"
k=0
# shellcheck disable=SC2016 # the text is to reach the list as it stands
for bad in 'Ours.First; echo' 'Ours/../First' '$(echo x)' 'Ours First' '-x' '+Ours.First:exe'; do
  k=$((k + 1))
  printf '%s\n' Ours.First "$bad" > "$T/tools/lean-verify/native-modules.txt"
  refused "list-$k" "$S" "${six[@]}"; tells "list-$k" 'a line that is not a module name'
done
echo '# nothing' > "$T/tools/lean-verify/native-modules.txt"; refused list-empty "$S" "${six[@]}"
step cp -- "$W/list" "$T/tools/lean-verify/native-modules.txt"

# 4. What Lean is handed, whatever the caller's environment says.
# The two holds "core 0" can go red only where this account may raise its limit on core files.
cores='the two holds on core files were made'
(ulimit -c 1000) 2> /dev/null || cores='the two holds on core files could not be made: the limit on core files cannot be raised here'
LEAN_VERIFY_ARGS=/nowhere LEAN_ABORT_ON_PANIC=0 LEAN_PATH=/nowhere LEAN_SRC_PATH=/nowhere \
  PLAY_LEAF=3 call handed 0 "$A" bash -c 'ulimit -c 1000 2> /dev/null; umask 000; exec "$@"' - \
  env LC_ALL=C.UTF-8 "$S" "${six[@]}"
for l in 'panic 1' 'arguments 7' 'paths unset unset' 'locale C' 'file size 1024' "cwd $T/lean" 'arg -j1' 'arg --tstack=4000000' \
  'arg interpreter.prefer_native=true' "arg $T/tools/lean-verify/LeanVerify.lean" \
  "claim config/policy.txt $(sum "$I/config/policy.txt")" "claim config/layout.txt $(sum "$I/config/layout.txt")" \
  "claim config/KL.bin $(sum "$I/config/KL.bin")" "claim ID.txt $(sum "$I/id")" "claim example.req $(sum "$I/req")" \
  "claim example.y $(sum "$I/y")" 'parsed key key' 'parsed leaf leaf-3' 'parsed proof proof' 'parsed key key seeds' \
  'parsed leaf leaf-3 seeds'; do got_line handed "$l"; done
grep -a -q -x -e 'arg --load-dynlib=.*/libnative\.so' -- "$W/calls/handed.record" || wrong "handed: no library"
asked() { printf '%s\n' Johnson.PVR.Program Johnson.PVR.ProgramGuard | cmp -s -- - "$W/calls/$1.record.fits"; }
asked handed || wrong "handed: Lake was not asked of the import and of the library's checks of it"
printf '%s\n' +Ours.First:c.o.export "+Pkg.Second':c.o.export" | cmp -s -- - "$W/calls/handed.record.build" ||
  wrong "handed: other targets for lake build"
echo 0022 | cmp -s -- - "$W/calls/handed.record.umask" || wrong "handed: Lake would write files that others can change"
# The parser is handed the copies of the public files and of the request; the key folder and the proof
# file in place.
work=$(sed -n 's/^work directory: //p' "$W/calls/handed.err")
printf '%s\n' "$work/claim/config" "$I/ivk" "$work/claim/example.req" "$I/proof" "$work/parser" |
  cmp -s -- - "$W/calls/handed.record.parser" || wrong "handed: other arguments for the parser"
for l in "$(sum "$I/id")  ID.txt" "$(sum "$I/config/KL.bin")  config/KL.bin" "claim: commitment ID: $h" \
  'claim: checked positions: from r = 2 up to, not including, e = 4' 'claim: prompt, 3 tokens: [1, 2, 3]' \
  'claim: output, 2 tokens: [4, 5]' "public files: the acceptance policy, by the parameter file's own lines: [C 256, theta 256, cap 0, s 15, sh 8, sT 4, F 24]" \
  'not part of the claim: tokens of the request behind position e, which go to the verifier with the proof: 0' 'the three public files are those of examples/full-size/config' \
  'the policy lines (C, theta, cap, sh, sT) read (256, 256, 0, 8, 4)' "read_claim.py: claim: commitment ID: $h" \
  "running the verifier's executable form inside Lean (about half an hour)"; do tells handed "$l"; done
got_line handed 'core 0'
tells handed "tree: $T"
step mkdir -- "$I/config-other"; step cp -- "$I/config/policy.txt" "$I/config/layout.txt" "$I/config-other/"
echo other > "$I/config-other/KL.bin"
call other-public-files 0 "$A" "$S" "$I/config-other" "${six[@]:1}"
tells other-public-files 'NOTE: the three public files are not those of examples/full-size/config'
# <name> <how the line on the public files begins> <how the line on the policy begins>: each is said
# twice, at the start and right before the verdict, where the two are the error stream's last lines.
twice() {
  local l
  for l in "$2" "$3"; do
    [ "$(grep -a -c -e "^$l" -- "$W/calls/$1.err")" = 2 ] || wrong "$1: this does not stand twice: $l"
  done
  tail -n 2 -- "$W/calls/$1.err" | head -n 1 | grep -a -q -e "^$2" || wrong "$1: the last line but one is not: $2"
  tail -n 1 -- "$W/calls/$1.err" | grep -a -q -e "^$3" || wrong "$1: the last line is not: $3"
}
other='NOTE: the three public files are not those'
exact='the policy lines (C, theta, cap, sh, sT) read (256, 256, 0, 8, 4)$'
twice other-public-files "$other" "$exact"
PLAY=reject call other-public-files-reject 1 'VERDICT: REJECT' "$S" "$I/config-other" "${six[@]:1}"
twice other-public-files-reject "$other" "$exact"
twice handed 'the three public files are those of examples/full-size/config$' "$exact"
# .. also where the layout file alone is another.
step mkdir -- "$I/config-other-layout"; step cp -- "$I/config/policy.txt" "$I/config/KL.bin" "$I/config-other-layout/"
echo other > "$I/config-other-layout/layout.txt"
call other-layout 0 "$A" "$S" "$I/config-other-layout" "${six[@]:1}"; twice other-layout "$other" "$exact"
# Another acceptance policy: the accept's own line says so and names the three numbers, whatever their
# order in the file; the error stream says it twice; a reject's line stays as it is.
# <name> <C> <theta> <cap> <sh> <sT> <these five lines of a parameter file ..>
tolerant() {
  local name=$1 c=$2 t=$3 p=$4 h=$5 u=$6
  shift 6
  step mkdir -- "$I/$name"; step cp -- "$I/config/layout.txt" "$I/config/KL.bin" "$I/$name/"
  printf '%s\n' 'lt 2' 'V 100' "$@" 's 15' 'F 24' > "$I/$name/policy.txt"
  call "$name" 0 "VERDICT: ACCEPT (under the files' policy C=$c theta=$t cap=$p; NOT exact match: see \"What an accept means\")" \
    "$S" "$I/$name" "${six[@]:1}"
  twice "$name" "$other" \
    "NOTE: the policy lines (C, theta, cap, sh, sT) read ($c, $t, $p, $h, $u), not (256, 256, 0, 8, 4): see "
}
tolerant policy-tolerant 25856 512 1 8 4 'C 25856' 'theta 512' 'cap 1' 'sh 8' 'sT 4'
tolerant policy-other-C 255 256 0 8 4 'C 255' 'theta 256' 'cap 0' 'sh 8' 'sT 4'
tolerant policy-other-theta 256 2560 0 8 4 'C 256' 'theta 2560' 'cap 0' 'sh 8' 'sT 4'
tolerant policy-other-cap 256 256 10 8 4 'C 256' 'theta 256' 'cap 10' 'sh 8' 'sT 4'
tolerant policy-other-sh 256 256 0 7 4 'C 256' 'theta 256' 'cap 0' 'sh 7' 'sT 4'
tolerant policy-other-sT 256 256 0 8 3 'C 256' 'theta 256' 'cap 0' 'sh 8' 'sT 3'
tolerant policy-numbers-changed-round 0 256 256 4 8 'sT 8' 'cap 256' 'sh 4' 'theta 256' 'C 0'
PLAY=reject call policy-tolerant-reject 1 'VERDICT: REJECT' "$S" "$I/policy-tolerant" "${six[@]:1}"
twice policy-tolerant-reject "$other" 'NOTE: the policy lines (C, theta, cap, sh, sT) read (25856, 512, 1, 8, 4), not (256, 256, 0, 8, 4): '
# Exact match with the lines in another order.
step mkdir -- "$I/policy-order"; step cp -- "$I/config/layout.txt" "$I/config/KL.bin" "$I/policy-order/"
printf '%s\n' 'F 24' 'cap 0' 'lt 2' 'theta 256' 'sT 4' 'V 100' 'sh 8' 'C 256' 's 15' > "$I/policy-order/policy.txt"
call policy-order 0 "$A" "$S" "$I/policy-order" "${six[@]:1}"; twice policy-order "$other" "$exact"
got_line other-public-files "claim config/KL.bin $(sum "$I/config-other/KL.bin")"
# A file of the caller's that changes after the start: Lean is handed what lay there at the start.
for i in 2 3 5; do
  a=("${six[@]}"); step cp -- "${six[i]}" "$I/swap-$i"; a[i]=$I/swap-$i
  PLAY_SWAP=$I/swap-$i call "swapped-$i" 0 "$A" "$S" "${a[@]}"
  echo swapped | cmp -s -- - "$I/swap-$i" || wrong "swapped-$i: the stand-in did not change the file"
done
got_line swapped-2 "claim example.req $(sum "$I/req")"; got_line swapped-3 "claim example.y $(sum "$I/y")"
got_line swapped-5 "claim ID.txt $(sum "$I/id")"
for f in policy.txt layout.txt KL.bin; do
  step mkdir -- "$I/config-swap-$f"; step cp -- "$I/config/policy.txt" "$I/config/layout.txt" "$I/config/KL.bin" "$I/config-swap-$f/"
  PLAY_SWAP=$I/config-swap-$f/$f call "swapped-$f" 0 "$A" "$S" "$I/config-swap-$f" "${six[@]:1}"
  echo swapped | cmp -s -- - "$I/config-swap-$f/$f" || wrong "swapped-$f: the stand-in did not change the file"
  got_line "swapped-$f" "claim config/$f $(sum "$I/config/$f")"
  tells "swapped-$f" 'the three public files are those of examples/full-size/config'
done
call handed-no-native 0 "$A" "$S" --no-native "${six[@]}"
for l in 'panic 1' 'arguments 7' 'arg -j1' 'arg --tstack=4000000'; do got_line handed-no-native "$l"; done
tells handed-no-native "running the verifier's executable form inside Lean (hours: Lean interprets all the code)"
if grep -a -q -e load-dynlib -e prefer_native -- "$W/calls/handed-no-native.record"; then wrong "handed-no-native: a library"; fi
asked handed-no-native || wrong "handed-no-native: Lake was not asked of the import and of the library's checks of it"
[ ! -e "$W/calls/handed-no-native.record.build" ] || wrong "handed-no-native: Lake was told to build"

# 5. How Lean's status and lines are read. 0 for one ending alone, 1 for one alone, 2 for the rest.
PLAY=accept call accept 0 "$A" "$S" "${six[@]}"
PLAY=reject call reject 1 'VERDICT: REJECT' "$S" "${six[@]}"
PLAY=reject call reject-no-native 1 'VERDICT: REJECT' "$S" --no-native "${six[@]}"
for p in accept-status-0 accept-status-1 accept-status-2 accept-status-11 accept-status-12 accept-status-134 \
  accept-status-137 accept-status-143 accept-status-255 accept-on-stdout all-on-stdout accept-blank accept-cr accept-nul \
  accept-cut accept-short accept-lower accept-indented accept-inside accept-not-last accept-then-empty accept-twice \
  reject-then-accept none-then-accept rejected-then-accept accept-then-reject reject-status-0 reject-status-1 reject-status-10 \
  reject-status-12 reject-status-134 reject-blank reject-on-stdout reject-not-last no-verdict no-verdict-status-0 \
  no-verdict-status-10 no-verdict-status-11 silent-0 silent-1 silent-10 silent-11 silent-12 silent-134 silent-137 error \
  aborted aborted-after-accept not-a-play; do
  PLAY=$p call "$p" 2 "$N" "$S" "${six[@]}"
  PLAY=$p call "$p-no-native" 2 "$N" "$S" --no-native "${six[@]}"
done
# Lean's lines on the claim are not the second reader's.
for c in other-token other-id other-policy a-line-less a-line-more other-order none; do
  PLAY=accept PLAY_CLAIM=$c call "accept-claim-$c" 2 "$N" "$S" "${six[@]}"
  case $c in
    none) tells "accept-claim-$c" 'before it printed the claim' ;;
    *) tells "accept-claim-$c" "Lean's reading of the claim differs from read_claim.py's" ;;
  esac
  PLAY=reject PLAY_CLAIM=$c call "reject-claim-$c" 2 "$N" "$S" "${six[@]}"
done
tells silent-12 'before it printed the claim'
tells accept-status-11 "Lean's status (11) and its last lines do not make one verdict"
# A second reader that prints nothing and ends with 0, and a Lean that prints no line on the claim: two
# empty readings are the same bytes, and still there is no verdict.
step mv -- "$T/tools/lean-verify/read_claim.py" "$T/tools/lean-verify/read_claim.py.away"
echo 'pass' > "$T/tools/lean-verify/read_claim.py"
PLAY=accept PLAY_CLAIM=none refused reader-silent "$S" "${six[@]}"; tells reader-silent 'did not print its six lines'
step mv -- "$T/tools/lean-verify/read_claim.py" "$T/tools/lean-verify/read_claim.py.silent"
# A second reader whose sixth line names no number: there is nothing to put into an accept's line.
printf '%s\n' 'for i in range(5): print("claim: x")' 'print("public files: none")' > "$T/tools/lean-verify/read_claim.py"
PLAY=accept refused reader-without-the-numbers "$S" "${six[@]}"
tells reader-without-the-numbers 'sixth line lacks C, theta, cap, sh or sT'
step mv -- "$T/tools/lean-verify/read_claim.py" "$T/tools/lean-verify/read_claim.py.numberless"
step mv -- "$T/tools/lean-verify/read_claim.py.away" "$T/tools/lean-verify/read_claim.py"
# The caller's variables for python3 do not reach the second reader.
step mkdir -- "$W/python"; echo 'raise SystemExit(0)' > "$W/python/struct.py"; step cp -- "$W/python/struct.py" "$W/python/start.py"
call python-variables 0 "$A" env "PYTHONPATH=$W/python" "PYTHONSTARTUP=$W/python/start.py" PYTHONINSPECT=1 "$S" "${six[@]}"
# A time limit around the call: timeout's own code, and what the script has started ends with it.
# The limit is five seconds, here and below, so that the script has reached its sleeping stand-in by
# then on a busy machine too. ("-k 5": were the script ever to hang at the signal, timeout would end
# it five seconds later, with another code than the one due, and this test would go on.)
PLAY=sleeps call time-limit 124 "$N" timeout -k 5 5 "$S" "${six[@]}"; gone time-limit
# A signal to the script alone ("--foreground": timeout then signals no other process): no verdict, and
# what the script has started does not run on: Lean, Lake's build, the parser.
for s in TERM HUP INT; do
  PLAY=sleeps call "signal-$s" 124 "$N" timeout --foreground -k 5 -s "$s" 5 "$S" "${six[@]}"; gone "signal-$s"
  tells "signal-$s" 'a signal stopped this script'
done
# .. and in each of the four phases before Lean: to the script alone, and to its whole group.
for ph in FITS:look PARSER:parser BUILD:build LINK:link; do
  call "signal-in-the-${ph#*:}" 124 "$N" env "PLAY_${ph%:*}=sleeps" timeout --foreground -k 5 5 "$S" "${six[@]}"
  gone "signal-in-the-${ph#*:}"; no_lean "signal-in-the-${ph#*:}"
  call "time-limit-in-the-${ph#*:}" 124 "$N" env "PLAY_${ph%:*}=sleeps" timeout -k 5 5 "$S" "${six[@]}"
  gone "time-limit-in-the-${ph#*:}"; no_lean "time-limit-in-the-${ph#*:}"
done
# The caller reads the error stream through a pipe, under a parent that adopts what the script leaves behind
# and collects none of it, as process 1 does in some containers. Whatever outlived the script with that
# stream open would keep the pipe open, and the caller would hang: the pipe must end with the script.
cat > "$W/adopt.py" << 'EOF'
"""usage: adopt.py FILE COMMAND ..: runs the command, its error stream through a pipe into the file.
Ends with the command's code; with 99 if the pipe is still open three seconds behind the command's end;
with 98 if this system does not let a process adopt orphans."""
import ctypes
import os
import subprocess
import sys
import time

if ctypes.CDLL(None).prctl(36, 1, 0, 0, 0) != 0:  # Linux: this process adopts its descendants' orphans
    sys.exit(98)
r, w = os.pipe()
p = subprocess.Popen(sys.argv[2:], stdin=subprocess.DEVNULL, stderr=w)
os.close(w)
os.set_blocking(r, False)
ended = None
with open(sys.argv[1], "wb") as out:
    while True:
        try:
            b = os.read(r, 65536)
        except BlockingIOError:
            b = None
        if b == b"":
            break
        if b:
            out.write(b)
        if ended is None and p.poll() is not None:
            ended = time.monotonic()
        if ended is not None and time.monotonic() - ended > 3:
            sys.exit(99)
        time.sleep(0.02)
sys.exit(p.wait())
EOF
PLAY=sleeps call signal-under-a-parent-that-collects-nobody 124 "$N" python3 -I -B "$W/adopt.py" "$W/calls/adopted.err" \
  timeout --foreground -k 5 5 "$S" "${six[@]}"; gone signal-under-a-parent-that-collects-nobody
grep -a -q -F 'a signal stopped this script' -- "$W/calls/adopted.err" || wrong "signal-under-a-parent-that-collects-nobody: the trap did not speak"
# A proof may be made to provoke output: Lean's job can write no more than the bound into one file, a run
# that meets the bound counts for nothing, whatever its status and last line, and what is shown of it is
# bounded with it (each line gains six bytes: with these lines of 50 bytes, well under twice the bound).
bound=1048576
for p in floods floods-and-goes-on floods-standard-output accept-at-the-bound-then-more; do
  PLAY=$p call "$p" 2 "$N" "$S" "${six[@]}"; tells "$p" "Lean's run wrote $bound bytes into"
  d=$(sed -n 's/^work directory: //p' -- "$W/calls/$p.err")
  for f in lean.err lean.out; do
    [ "$(stat -c %s -- "$d/$f")" -le "$bound" ] || wrong "$p: $f has more bytes than the bound"
  done
  [ "$(stat -c %s -- "$W/calls/$p.err")" -le $((2 * bound)) ] || wrong "$p: more is shown than the bound allows"
done
PLAY=much call much-under-the-bound 0 "$A" "$S" "${six[@]}"
# Standard output is closed: no code 0 and no code 1 without its line. The error stream is closed: the
# claim cannot be shown, so there is no verdict.
PLAY=accept call no-standard-output 2 '' bash -c '"$@" >&-' - "$S" "${six[@]}"
PLAY=reject call reject-no-standard-output 2 '' bash -c '"$@" >&-' - "$S" "${six[@]}"
PLAY=accept refused no-error-stream bash -c '"$@" 2>&-' - "$S" "${six[@]}"

# 6. check_list.sh: it judges no proof, and none of its lines is a verdict's.
K=$T/tools/lean-verify/check_list.sh
U='native-modules.txt: not checked'
LEAN_ABORT_ON_PANIC=0 LEAN_PATH=/nowhere LEAN_SRC_PATH=/nowhere call list-same 0 'native-modules.txt: the same as made anew' \
  bash -c 'ulimit -c 1000 2> /dev/null; umask 000; exec "$@"' - env LC_ALL=C.UTF-8 "$K"
for l in "cwd $T/lean" 'panic 1' 'core 0' 'umask 0022' 'paths unset unset' 'locale C'; do got_line list-same "$l"; done
tells list-same "tree: $T"
asked list-same || wrong "list-same: Lake was not asked of the import and of the library's checks of it"
for w in one-more one-less other-order; do
  PLAY_WALK=$w call "list-walk-$w" 1 'native-modules.txt: NOT the same as made anew' "$K"
done
for w in empty silent fails; do
  PLAY_WALK=$w call "list-walk-$w" 2 "$U" "$K"
done
call list-with-an-argument 2 "$U" "$K" more; no_lean list-with-an-argument
PLAY_FITS=stale call list-library-stale 2 "$U" "$K"; no_lean list-library-stale
step mv -- "$T/tools/lean-verify/LeanVerifyWalk.lean" "$T/tools/lean-verify/LeanVerifyWalk.lean.away"
call list-lacks-the-walk 2 "$U" "$K"; no_lean list-lacks-the-walk
step mv -- "$T/tools/lean-verify/LeanVerifyWalk.lean.away" "$T/tools/lean-verify/LeanVerifyWalk.lean"
# <name> <path>: the call while the path is a symbolic link to what lay there: the seven places of lean_verify.sh.
list_linked() {
  step mv -- "$2" "$2.real"; step ln -s -- "$2.real" "$2"
  call "$1" 2 "$U" "$K"; no_lean "$1"; tells "$1" 'is a symbolic link: refused'
  step mv -T -- "$2" "$W/links-put-aside/$1"; step mv -T -- "$2.real" "$2"
}
list_linked list-lean-is-a-link "$T/lean"
list_linked list-lake-directory-is-a-link "$L"
list_linked list-packages-is-a-link "$L/packages"
list_linked list-build-is-a-link "$L/build"
list_linked list-a-package-is-a-link "$L/packages/pkg"
list_linked list-a-packages-lake-directory-is-a-link "$L/packages/pkg/.lake"
list_linked list-a-packages-build-is-a-link "$L/packages/pkg/.lake/build"
# The six arguments of lean_verify.sh by mistake: no line of a verdict, no code 0.
call list-with-six-arguments 2 "$U" "$K" "${six[@]}"; no_lean list-with-six-arguments
call list-top-directory-ends-in-a-newline 2 "$U" "$W/twin${nl}/tools/lean-verify/check_list.sh"
no_lean list-top-directory-ends-in-a-newline
call list-path-holds-a-newline 2 "$U" "$W/y${nl}VERDICT: ACCEPT (exact match)/tree/tools/lean-verify/check_list.sh"
no_lean list-path-holds-a-newline; tells list-path-holds-a-newline "the repository's path holds a newline"
call list-temporary-directory-with-a-newline 2 "$U" env "TMPDIR=$W/tmp${nl}x" "$K"; no_lean list-temporary-directory-with-a-newline
tells list-temporary-directory-with-a-newline "the work directory's path holds a newline"
call list-no-timeout 2 "$U" env "PATH=$W/tools-alone" "$K"; tells list-no-timeout 'timeout is not on the PATH'
AT=$W/twin call list-path-with-dots-behind-a-link 2 "$U" l/../tools/lean-verify/check_list.sh
no_lean list-path-with-dots-behind-a-link
AT=$W/twin/l call list-standing-inside-a-link 2 "$U" ../tools/lean-verify/check_list.sh; no_lean list-standing-inside-a-link
PLAY_WALK=sleeps call list-signal 124 "$U" timeout --foreground -k 5 5 "$K"; gone list-signal
PLAY_WALK=sleeps call list-time-limit 124 "$U" timeout -k 5 5 "$K"; gone list-time-limit
PLAY_FITS=sleeps call list-signal-in-the-look 124 "$U" timeout --foreground -k 5 5 "$K"; gone list-signal-in-the-look
PLAY_FITS=sleeps call list-time-limit-in-the-look 124 "$U" timeout -k 5 5 "$K"; gone list-time-limit-in-the-look
call list-no-standard-output 2 '' bash -c '"$@" >&-' - "$K"
# A signal behind Lean's end: it is passed over, and one line stands on standard output. A stand-in for
# grep, which both scripts call there, sleeps through the limit of five seconds.
step mkdir -- "$W/bin-slow-end"
PLAY_GREP=$(command -v grep)
export PLAY_GREP
cat > "$W/bin-slow-end/grep" << 'EOF'
#!/bin/bash
# A stand-in for grep: the real one; before the two calls behind Lean's end it sleeps.
case "${1:-} ${2:-}" in '-a -c' | '-v -e') sleep 7 ;; esac
exec "$PLAY_GREP" "$@"
EOF
step chmod +x -- "$W/bin-slow-end/grep"
PLAY=accept call signal-behind-the-end 124 "$A" env "PATH=$W/bin-slow-end:$PATH" timeout --foreground -k 5 5 "$S" "${six[@]}"
# A signal while Lean's lines are being shown is not yet passed over: that one large write is made with the
# trap in place. A stand-in for sed sleeps there through the limit.
step mkdir -- "$W/bin-slow-show"
PLAY_SED=$(command -v sed)
export PLAY_SED
cat > "$W/bin-slow-show/sed" << 'EOF'
#!/bin/bash
# A stand-in for sed: the real one; before it shows Lean's lines it sleeps.
case ${1:-} in 's/^/lean: /') sleep 7 ;; esac
exec "$PLAY_SED" "$@"
EOF
step chmod +x -- "$W/bin-slow-show/sed"
PLAY=accept call signal-while-the-lines-are-shown 124 "$N" env "PATH=$W/bin-slow-show:$PATH" timeout --foreground -k 5 5 "$S" "${six[@]}"
tells signal-while-the-lines-are-shown 'a signal stopped this script'
call list-signal-behind-the-end 124 'native-modules.txt: the same as made anew' env "PATH=$W/bin-slow-end:$PATH" \
  timeout --foreground -k 5 5 "$K"
call list-posixly-correct 0 'native-modules.txt: the same as made anew' env POSIXLY_CORRECT=1 "$K"
got_line list-posixly-correct 'posix unset'
AT=$T call list-cdpath-without-p 0 'native-modules.txt: the same as made anew' env "CDPATH=$W/decoy" \
  bash tools/lean-verify/check_list.sh

# 7. This test's own argument: a name that would lead into another directory than the one just made, and a
# directory that is there already. Each call ends at once, and nothing is laid in the directory beside.
# ("timeout": were a refusal ever lost, the test would start itself anew inside itself, without end.)
step mkdir -- "$W/beside"
call own-argument-ends-in-a-newline 2 '' timeout 20 bash "$here/lean_verify_test.sh" "$W/beside${nl}"
AT=$W/beside call own-argument-is-a-dash 2 '' timeout 20 bash "$here/lean_verify_test.sh" -
call own-argument-exists 2 '' timeout 20 bash "$here/lean_verify_test.sh" "$W/beside"
call own-arguments-are-two 2 '' timeout 20 bash "$here/lean_verify_test.sh" "$W/new-1" "$W/new-2"
# ".." behind a link: mkdir makes elsewhere/beside; "cd" would enter the directory beside, which is there.
step mkdir -p -- "$W/elsewhere/sub"; step ln -s -- "$W/elsewhere/sub" "$W/l"
AT=$W call own-argument-with-dots-behind-a-link 2 '' timeout 20 bash "$here/lean_verify_test.sh" l/../beside
tells own-argument-with-dots-behind-a-link 'leads elsewhere than it reads'
# The test itself is called by a path with ".." behind a link: two copies of it, in two folders.
step mkdir -p -- "$W/own/a/tools/lean-verify" "$W/own/a/lean" "$W/own/b/tools/lean-verify"; step ln -s -- "$W/own/a/lean" "$W/own/b/l"
step cp -- "$here/lean_verify_test.sh" "$W/own/a/tools/lean-verify/"; step cp -- "$here/lean_verify_test.sh" "$W/own/b/tools/lean-verify/"
AT=$W/own/b call own-path-with-dots-behind-a-link 2 '' timeout 20 bash l/../tools/lean-verify/lean_verify_test.sh "$W/new-3"
[ ! -e "$W/new-3" ] || wrong "own-path-with-dots-behind-a-link: a directory was made"
[ -z "$(ls -A -- "$W/beside")" ] || wrong "the test's own refusals laid something in the directory beside"
{ [ ! -e "$W/beside${nl}" ] && [ ! -e "$W/new-1" ]; } || wrong "the test's own refusals made a directory"

# 8. No line of any call's error stream begins like a verdict: Lean's own stand behind "lean: ".
if grep -l -a -E '^(VERDICT: |NO VERDICT)' -- "$W"/calls/*.err; then wrong "an error stream holds a line that begins like a verdict"; fi
tells accept 'lean: VERDICT: ACCEPT'; tells accept-status-11 'lean: VERDICT: ACCEPT'
# The words "exact match" are an accept's line's alone: they stand on no error stream.
if grep -l -a -F 'exact match' -- "$W"/calls/*.err; then wrong "an error stream says \"exact match\""; fi

# By their text: each script signals in one place alone, a trap that lets no second trap start inside
# it, its own jobs, as jobs and never by number; each of its jobs is one process under "timeout 0",
# none a pipeline (from a job that is a pipeline of which a later process has ended and an earlier
# one has not, bash 5.2.21's "kill %job" does not come back); and nothing follows a file.
for f in lean_verify.sh check_list.sh; do
  [ "$(grep -c -w kill -- "$here/$f")" = 1 ] || wrong "$f: the word kill does not stand on exactly one line"
  grep -q -F "do kill -TERM \"%\$j\" 2> /dev/null; done" -- "$here/$f" || wrong "$f: the trap does not signal its jobs as jobs"
  grep -q -x -F "trap 'trap \"\" INT TERM HUP" -- "$here/$f" || wrong "$f: the trap does not begin by passing over further signals"
  if grep -v '^ *#' -- "$here/$f" | grep -q -E ' \| .*[^&]&$'; then wrong "$f starts a pipeline as a job"; fi
  if grep -v '^ *#' -- "$here/$f" | grep -q -E 'tail .*-f'; then wrong "$f follows a file"; fi
  [ "$(grep -v '^ *#' -- "$here/$f" | grep -c -E '[^&]&$')" = "$(grep -v '^ *#' -- "$here/$f" | grep -c -E '^ *timeout 0 ')" ] ||
    wrong "$f starts a job that is not under \"timeout 0\""
done

# 9. The real files, by their text alone. The stand-ins above play the statuses 10, 11 and 12 and the
# fixed words; a later turn of the entry or of Program.lean that parts from them would give no verdict
# at every call, and only a run with Lean would show it.
for l in 'import Johnson.PVR.Program' 'set_option stderrAsMessages false' \
  '  let args := ((← IO.getEnv "LEAN_VERIFY_ARGS").getD "").splitOn "\n"' '  let code ← PV.Program.run args' \
  '  IO.Process.exit (match code with | 0 => 10 | 1 => 11 | _ => 12)'; do
  grep -q -x -F -e "$l" -- "$here/LeanVerify.lean" || wrong "LeanVerify.lean lacks the line: $l"
done
program=$here/../../lean/Johnson/PVR/Program.lean looked='Program.lean was looked at'
if [ -f "$program" ]; then
  for l in '| .accept => 0' '| .reject => 1' '| .noVerdict _ => 2' '| .accept => "VERDICT: ACCEPT"' \
    '| .reject => "VERDICT: REJECT"' '| .noVerdict why => "NO VERDICT: " ++ why' 'say s!"claim: commitment ID: {' \
    'say s!"claim: checked positions: from r = {c.prompt.length - 1} up to, not including,' \
    'say s!"claim: prompt, {c.prompt.length} tokens: {c.prompt}"' \
    'say s!"claim: output, {c.output.length} tokens: {c.output}"' \
    'say s!"not part of the claim: tokens of the request behind position e, which go to the' \
    'verifier with the proof: {c.further.length}"' \
    "say s!\"public files: the acceptance policy, by the parameter file's own lines:"; do
    grep -q -F -e "$l" -- "$program" || wrong "Program.lean lacks: $l"
  done
else
  looked='Program.lean does not lie here and was not looked at'
fi

# 10. The caller's files are as they were, and nothing was written beside them or into the tree
# outside lean/.lake.
(cd -- "$I" && sha256sum --status -c "$W/sums-at-the-start") || wrong "a file of the caller's has changed"
[ -z "$(find "$I/config" "$I/ivk" "$T/tools" "$T/examples" -newer "$W/bin/lake" -type f ! -name native-modules.txt ! -name 'read_claim.py*')" ] ||
  wrong "a new file lies beside the caller's, or in the tree"
cmp -s -- "$here/read_claim.py" "$T/tools/lean-verify/read_claim.py" || wrong "read_claim.py in the stand-in tree has changed"
cmp -s -- "$W/list" "$T/tools/lean-verify/native-modules.txt" || wrong "the list in the stand-in tree has changed"
[ -z "$(find "$T/lean" -mindepth 1 -maxdepth 1 ! -name '.lake*')" ] || wrong "something was written into lean/ outside .lake"

# 11. A number of more than twenty digits in the parameter file is refused, unshown: it would only swell
# what is shown and kept. (The format page allows ten.) Twenty digits pass.
policy policy-number-of-21-digits 'lt 2' 'V 100' 'C 100000000000000000000' "${seven[@]:1}"; tells policy-number-of-21-digits 'at most 20 digits'
policy policy-V-of-21-digits 'lt 2' 'V 100000000000000000000' "${seven[@]}"; tells policy-V-of-21-digits 'at most 20 digits'
policy policy-lt-of-21-digits 'lt 000000000000000000002' 'V 100' "${seven[@]}"; tells policy-lt-of-21-digits 'at most 20 digits'
tolerant policy-number-of-20-digits 18446744073709551615 256 0 8 4 'C 18446744073709551615' 'theta 256' 'cap 0' 'sh 8' 'sT 4'
if grep -a -q -F 100000000000000000000 -- "$W"/calls/policy-*-of-21-digits.err; then wrong "a number of 21 digits is shown"; fi

# The number of calls is fixed, so that none can drop out in silence.
due=344
[ "$calls" = "$due" ] || wrong "$calls calls were made where $due were due"
echo "$cores"
echo "$looked"
if [ "$off" = 0 ]; then
  echo "lean_verify_test.sh: $calls calls: all as expected"
  exit 0
fi
echo "lean_verify_test.sh: $calls calls: NOT AS EXPECTED: $off"
exit 1
