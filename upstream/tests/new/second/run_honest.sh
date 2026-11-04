#!/bin/bash
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.

# run_honest.sh <new work directory ("folder" in messages), outside the tree> [jobs, default 4]
#
# Builds blocks/toy/handover.cpp, blocks/frame/handover_files.cpp and
# blocks/frame/handover_proof.cpp. These tools
# write toy proofs, key files and leaf files (`entry` in names and output): honest ones, and copies
# with altered bytes. A key file commits to a table; a leaf file ties a second table to it. Each
# file is checked by the C++ verifiers and by the second verifiers, in Python, beside this script.
# Both must reject ("refused" in output) the altered copies, and two leaf files that files_check.py
# makes by copying or swapping bytes within an honest one (`bent` in names and output).
# The parameters of the toy key file and leaf file are also compared with the format page's known
# answers (vectors_against_the_toy_files.py).
#
# Synthetic data only. The private randomness comes from small fixed numbers, so nothing is hidden
# in these files, and every run makes the same bytes: their SHA-256 digests are compared with
# expected/sha256.txt.
#
# The script creates the work directory itself, for its owner only, in a directory that exists and
# that open_above() finds safe from other accounts; a name that exists, even as a symbolic link, is
# rejected. It writes only below the work directory, apart from the compiler's temporary files, and
# deletes nothing.
#   <work>/build/                         the tools
#   <work>/toy-files/honest-and-altered/{product,links,key-and-entry,proofs,
#       proofs-of-one-link-with-arrays,proofs-with-further-messages}/
#                                         honest files, their lists, altered copies
#                                         ("altered" in their names)
#   <work>/toy-files/bent/key-and-entry/  the two `bent` leaf files
#   <work>/said/                          each program's output
#
# Exit status:
#   0  the last line begins "ALL AS EXPECTED: "
#   1  something was not as expected
#   2  the script could not start (no argument: bash's own error status)
#   3  as 1, and a verifier accepted a copy made with `--more-altered`, an altered copy in
#      key-and-entry/ or a `bent` leaf file; each such line begins "WAS NOT REFUSED", and the file
#      stays where it is. An altered copy accepted within a tool's own run, or by toy_check.py,
#      links_check.py or proof_check.py, gives 1.
set -u
unset CDPATH   # with it set, cd may go elsewhere and print
umask 077      # what this script makes is for its owner only
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
W="${1:?a new work folder}"; J="${2:-4}"
HERE="$(cd "$(dirname "$0")" && pwd)" || exit 2; TOP="$(cd "$HERE/../../.." && pwd)" || exit 2; PAGE="$TOP/docs/format"
TOP_REAL="$(realpath -- "$TOP")" || exit 2   # the tree's path without links: make builds there, and the work folder's walked path has none
[ -d "$TOP/blocks" ] && [ -f "$PAGE/format_rules.py" ] || { echo "blocks/ and docs/format/ are not where this script expects them: $TOP"; exit 2; }
case "$W$PWD$TOP$TOP_REAL" in *[[:space:]]*) echo "make cannot build under a path with a blank in it: $W (from $PWD), or $TOP (that is, $TOP_REAL)"; exit 2;; esac
[[ $W == /* ]] || W=$PWD/$W
while [[ $W == */ && $W != / ]]; do W=${W%/}; done
case ${W##*/} in '' | . | ..) echo "run_honest.sh: name the work folder by a name of its own"; exit 2 ;; esac
# The tools are built and run below the work directory: the script rejects it if another account may
# be able to change a directory above it, and uses `real`, the checked path without links.
if open_above "${W%/*}/"; then
  if [ "$why" = 'it does not exist' ]; then
    echo "run_honest.sh: $place, above the work folder, does not exist: make it first; this script makes only the work folder. Nothing was created."
    exit 2
  fi
  echo "run_honest.sh: the work folder $(printf %q "$W") lies below $place, which another account may be able to change ($why): it could put another folder in the work folder's place while the tools are built and run, and choose what they run as you. Name a new work folder below folders that only you or root can change (such as one in your home directory, or in /var/tmp, whose sticky bit keeps others from replacing it)$cure; nothing was run"
  exit 2
fi
W=${real%/}/${W##*/}
case "$W" in *[!A-Za-z0-9_./-]*) echo "$W: the work folder's path may hold only letters, digits and _ . / - (make would read other signs, such as \$, as its own)"; exit 2;; esac
case "$W/" in "$TOP"/* | "$TOP_REAL"/*) echo "the work folder lies inside the source tree"; exit 2;; esac
mkdir -m 700 -- "$W" || { echo "$W exists already, or could not be made; nothing was run"; exit 2; }
H="$W/toy-files/honest-and-altered"; BENT="$W/toy-files/bent/key-and-entry"; S="$W/said"; B="$W/build"
mkdir -p "$H/product/tiny-with-a-repeat" "$H/links/tiny-with-a-repeat" "$H/key-and-entry" "$BENT" "$S" "$B" "$H/proofs/on-the-first-table" "$H/proofs/on-the-copy" "$H/proofs-of-one-link-with-arrays/on-the-first-table" "$H/proofs-of-one-link-with-arrays/on-the-copy" "$H/proofs-with-further-messages/on-the-first-table" "$H/proofs-with-further-messages/on-the-copy"
bad=0; accepted=0
say() { echo "$@"; }
fail() { echo "NOT AS EXPECTED: $*"; bad=$((bad + 1)); }
not_refused() { echo "WAS NOT REFUSED: $*"; accepted=$((accepted + 1)); bad=$((bad + 1)); }
# $1: a tool's `judge`, `judge-key` or `judge-entry` status: 1 rejected, 0 accepted, 2 not checked.
judged() { case "$1" in 1) ;; 0) not_refused "$2 accepts $3";; *) fail "$2 did not judge $3 (status $1)";; esac; }
ends_with() { case "$(tail -n 1 "$1")" in *"$2") ;; *) fail "$1 ends: $(tail -n 1 "$1" | cut -c1-160)";; esac; }

# ---- 1. the three tools (`f2h1`: the Makefile's only setting) ----
make -C "$TOP/blocks" -j "$J" OUT="$B/plain" SETTINGS=f2h1 "$B/plain/f2h1/toy/handover.bin" "$B/plain/f2h1/frame/handover_files.bin" "$B/plain/f2h1/frame/handover_proof.bin" > "$S/build.txt" 2>&1 || { echo "the tools do not build: see $S/build.txt"; exit 1; }
T1="$B/plain/f2h1/toy/handover.bin"; T2="$B/plain/f2h1/frame/handover_files.bin"; T3="$B/plain/f2h1/frame/handover_proof.bin"

# ---- 2. honest files and altered copies, checked by the C++ verifiers as they are made ----
"$T1" "$H/product" product > "$S/product.txt" 2>&1 || fail "the tool on «product»"
"$T1" "$H/product" tiny > "$S/tiny.txt" 2>&1 || fail "the tool on «tiny»"
"$T1" "$H/links" links > "$S/links.txt" 2>&1 || fail "the tool on «links»"
"$T1" "$H/links" tiny-links > "$S/tiny-links.txt" 2>&1 || fail "the tool on «tiny-links»"
# The tiny sizes again, under randomness keys (47, 50) at which a position is drawn twice in a list
# whose claims follow the draws (docs/format/FORMAT.md, R6.6).
"$T1" "$H/product/tiny-with-a-repeat" tiny 6 47 > "$S/tiny-repeat.txt" 2>&1 || fail "the tool on «tiny» with a repeat"
"$T1" "$H/links/tiny-with-a-repeat" tiny-links 6 50 > "$S/tiny-links-repeat.txt" 2>&1 || fail "the tool on «tiny-links» with a repeat"
"$T2" "$H/key-and-entry" > "$S/key-and-entry.txt" 2>&1 || fail "the tool on key and entry"
"$T3" "$H/proofs" > "$S/proofs.txt" 2>&1 || fail "the tool on two batches of proofs"
# Again with requests of one segment (`link` in names): each leaves one claim on the committed
# table, so no claims are batched after the last proximity test; and with a layout that has arrays.
"$T3" "$H/proofs-of-one-link-with-arrays" 1 arrays > "$S/proofs-one-link.txt" 2>&1 || fail "the tool on two batches of proofs of one link"
# Again: the second segment has two more oracle messages: one in a leaf class of its own, so in its
# own Merkle tree, and one at a later commit point.
"$T3" "$H/proofs-with-further-messages" 3 class later > "$S/proofs-further.txt" 2>&1 || fail "the tool on two batches of proofs with further messages"
for f in product tiny links tiny-links tiny-repeat tiny-links-repeat; do say "C++: $(tail -n 1 "$S/$f.txt" | cut -c1-170)"; done
tail -n 3 "$S/key-and-entry.txt" | sed 's/^/C++: /' | cut -c1-170
for f in proofs proofs-one-link proofs-further; do say "C++: $f: $(grep -c 'proof: .* YES' "$S/$f.txt") files accepted; $(tail -n 1 "$S/$f.txt" | cut -c1-140)"; done

# ---- 3. the same files by the second verifiers ----
py() { local p="$1"; shift; python3 -B "$HERE/$p" "$PAGE" "$@"; }
py toy_check.py "$H/product" product --write > "$S/second.product.txt" 2>&1; ends_with "$S/second.product.txt" "product: ALL AGREES"
py toy_check.py "$H/product" tiny --write > "$S/second.tiny.txt" 2>&1; ends_with "$S/second.tiny.txt" "tiny: ALL AGREES"
py toy_check.py "$H/product/tiny-with-a-repeat" tiny --write > "$S/second.tiny-repeat.txt" 2>&1; ends_with "$S/second.tiny-repeat.txt" "tiny: ALL AGREES"
py links_check.py "$H/links" links --write > "$S/second.links.txt" 2>&1; ends_with "$S/second.links.txt" "links: ALL AGREES"
py links_check.py "$H/links" tiny-links --write > "$S/second.tiny-links.txt" 2>&1; ends_with "$S/second.tiny-links.txt" "tiny-links: ALL AGREES"
py links_check.py "$H/links/tiny-with-a-repeat" tiny-links --write > "$S/second.tiny-links-repeat.txt" 2>&1; ends_with "$S/second.tiny-links-repeat.txt" "tiny-links: ALL AGREES"
# Three more altered copies of the key file, at offsets that only a parser knows; the C++ verifier
# checks them as files.
py files_check.py "$H/key-and-entry" --more-altered > "$S/more-altered.tsv" 2>&1
TAB="$(printf '\t')"
while IFS="$TAB" read -r which number what at; do
  [ -s "$H/key-and-entry/key.altered-$number.file" ] || { fail "key.altered-$number.file was not made"; continue; }   # the tool's judge would call an empty file «refused»
  "$T2" "$H/key-and-entry" judge-key "$H/key-and-entry/key.altered-$number.file" > /dev/null 2>&1; judged $? "the C++ reader" "key.altered-$number.file"
  printf '%s\t%s\t%s\t%s\t%s\n' "$which" "$number" "$what" "$at" "$(cat "$H/key-and-entry/key.altered-$number.txt")" >> "$H/key-and-entry/altered.tsv"
done < "$S/more-altered.tsv"
for f in key entry; do [ -s "$H/key-and-entry/$f.targets.tsv" ] || fail "$f.targets.tsv was not made: the two verifiers' values on the way would not be compared"; done
py files_check.py "$H/key-and-entry" --write > "$S/second.key-and-entry.txt" 2>&1; ends_with "$S/second.key-and-entry.txt" "ALL AGREES"
py files_check.py "$H/key-and-entry" --altered > "$S/second.key-and-entry-altered.txt" 2>&1; ends_with "$S/second.key-and-entry-altered.txt" "ALL REFUSED"
grep "$TAB""ACCEPTED$TAB" "$S/second.key-and-entry-altered.txt" | while IFS= read -r line; do echo "WAS NOT REFUSED: by the second verifier: $(echo "$line" | cut -f1-4)"; done > "$S/second.accepted.txt"
[ ! -s "$S/second.accepted.txt" ] || { cat "$S/second.accepted.txt"; accepted=$((accepted + $(wc -l < "$S/second.accepted.txt"))); bad=$((bad + 1)); }
py files_check.py "$H/key-and-entry" --spoil-vertex-record > "$S/second.record.txt" 2>&1; ends_with "$S/second.record.txt" "THE RECORD OF VALUES AT VERTICES IS KEPT AND READ"
py vertex_values_test.py "$H" > "$S/second.record-2.txt" 2>&1; ends_with "$S/second.record-2.txt" "THE RECORD OF VALUES AT VERTICES IS KEPT AND READ"
# The format page's known answers on this toy run (docs/format/format-vectors.txt), compared with the
# parameters that the tool has written.
py vectors_against_the_toy_files.py "$H/key-and-entry" > "$S/second.vectors.txt" 2>&1; ends_with "$S/second.vectors.txt" "ALL AGREES"
# Two more altered copies of request 0's proof in each batch, at offsets that only a parser knows;
# check_proof() checks them as files.
for v in "proofs:" "proofs-of-one-link-with-arrays:1 arrays" "proofs-with-further-messages:3 class later"; do d="${v%%:*}"; a="${v#*:}"
  py proof_check.py "$H/$d" --more-altered > "$S/more-altered.$d.tsv" 2>&1
  [ "$(wc -l < "$S/more-altered.$d.tsv")" = 4 ] || fail "four more altered copies were to be made in $d: see $S/more-altered.$d.tsv"
  while IFS="$TAB" read -r where number what at; do
    [ -s "$H/$d/$where/0.altered-$number.proof" ] || { fail "$d/$where/0.altered-$number.proof was not made"; continue; }
    "$T3" "$H/$d" $a judge "$where" 0 "$H/$d/$where/0.altered-$number.proof" > /dev/null 2>&1; judged $? "check_proof()" "$d/$where/0.altered-$number.proof"
    printf '%s\t%s\t%s\t%s\n' "$number" "$what" "$at" "$(cat "$H/$d/$where/0.altered-$number.txt")" >> "$H/$d/$where/altered.tsv"
  done < "$S/more-altered.$d.tsv"
done
# Each directory's own key file and leaf file (under other public files than key-and-entry's), then
# both batches: six honest proofs, every altered copy, and honest proofs under a wrong request,
# reply, table number or table.
py files_check.py "$H/proofs" > "$S/second.proofs-key-and-entry.txt" 2>&1; ends_with "$S/second.proofs-key-and-entry.txt" "ALL AGREES"
py proof_check.py "$H/proofs" --all > "$S/second.proofs.txt" 2>&1; ends_with "$S/second.proofs.txt" "ALL AGREES"
py files_check.py "$H/proofs-of-one-link-with-arrays" > "$S/second.proofs-one-link-key-and-entry.txt" 2>&1; ends_with "$S/second.proofs-one-link-key-and-entry.txt" "ALL AGREES"
py proof_check.py "$H/proofs-of-one-link-with-arrays" --all > "$S/second.proofs-one-link.txt" 2>&1; ends_with "$S/second.proofs-one-link.txt" "ALL AGREES"
py files_check.py "$H/proofs-with-further-messages" > "$S/second.proofs-further-key-and-entry.txt" 2>&1; ends_with "$S/second.proofs-further-key-and-entry.txt" "ALL AGREES"
py proof_check.py "$H/proofs-with-further-messages" --all > "$S/second.proofs-further.txt" 2>&1; ends_with "$S/second.proofs-further.txt" "ALL AGREES"
for f in product tiny tiny-repeat links tiny-links tiny-links-repeat key-and-entry key-and-entry-altered record record-2 proofs-key-and-entry proofs proofs-one-link-key-and-entry proofs-one-link proofs-further-key-and-entry proofs-further; do say "Python: $(tail -n 1 "$S/second.$f.txt" | cut -c1-220)"; done

# ---- 4. the two `bent` leaf files, checked by both verifiers ----
py files_check.py "$H/key-and-entry" --bend "$BENT" > "$S/bend.txt" 2>&1
for b in BENT_COLUMN BENT_SWAP; do [ -s "$BENT/entry.cheat-$b.file" ] || { fail "entry.cheat-$b.file was not made"; continue; }; "$T2" "$H/key-and-entry" judge-entry "$BENT/entry.cheat-$b.file" > /dev/null 2>&1; judged $? "the C++ reader" "the bent entry $b"; done
py files_check.py "$H/key-and-entry" --cheats "$BENT" "$HERE/expected/bent.tsv" > "$BENT/BOTH-VERIFIERS.tsv" 2>&1; ends_with "$BENT/BOTH-VERIFIERS.tsv" "2 false files, ALL REFUSED AS EXPECTED"
! grep -q "the second verifier: ACCEPTED" "$BENT/BOTH-VERIFIERS.tsv" || not_refused "the second verifier accepts a bent entry: see $BENT/BOTH-VERIFIERS.tsv"
say "Python: bent entries: $(tail -n 1 "$BENT/BOTH-VERIFIERS.tsv" | cut -c1-160)"

# ---- 5. the bytes: SHA-256 digests against expected/sha256.txt ----
# sed: that list has the two `bent` leaf files under another directory.
( cd "$W/toy-files" && find . -type f \( -name '*.proof' -o -name '*.file' -o -name '*.parameters' -o -name '*.configuration' -o -name 'public-file-*' -o -name '*.claims' -o -name '*.request' -o -name '*.reply' \) | LC_ALL=C sort | xargs -d '\n' sha256sum ) > "$S/sha256.txt"
grep -v '^#' "$HERE/expected/sha256.txt" | sed -n -e '\|  \./honest-and-altered/|p' -e 's|  \./FALSE-PROOFS-every-verifier-must-refuse/key-and-entry/\(entry\.cheat-BENT_[A-Z]*\.file\)$|  ./bent/key-and-entry/\1|p' | LC_ALL=C sort -k 2 > "$S/sha256.expected.txt"
if [ "$(wc -l < "$S/sha256.expected.txt")" != 409 ]; then fail "expected/sha256.txt names $(wc -l < "$S/sha256.expected.txt") honest, altered and bent files, not 409"
elif cmp -s "$S/sha256.expected.txt" "$S/sha256.txt"; then say "bytes: all $(wc -l < "$S/sha256.txt") proofs, files, parameters and public files have the sha256 that expected/sha256.txt gives them"
else fail "the bytes differ from expected/sha256.txt: compare $S/sha256.expected.txt with $S/sha256.txt"; fi

n_altered=$(find "$H" -type f -name '*altered-*' \( -name '*.proof' -o -name '*.file' \) | wc -l)
n_honest=$(find "$H" -type f \( -name '*.proof' -o -name '*.file' \) ! -name '*altered-*' | wc -l)
[ "$n_honest" = 32 ] || fail "32 honest files were to be made; $n_honest lie there"
if [ "$bad" = 0 ]; then echo "ALL AS EXPECTED: $n_honest honest files accepted; $n_altered altered copies and 2 bent entries refused; each judged by two verifiers"; exit 0
elif [ "$accepted" = 0 ]; then echo "$bad THINGS ARE NOT AS EXPECTED"; exit 1
else echo "$bad THINGS ARE NOT AS EXPECTED, $accepted OF THEM AN ALTERED OR BENT FILE THAT WAS NOT REFUSED"; exit 3; fi
