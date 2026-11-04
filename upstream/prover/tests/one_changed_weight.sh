#!/bin/bash
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.

# Test: the prover cp_prove, given a leaf table's unaltered secret files and model files in
# which one bit of one quantized weight is flipped, releases only an empty example.failed.
# Why: a leaf table is a committed copy of the weights with its own masks (random values encoded
# with the weights), derived from the seed and 32 stored bytes, both in the secret key directory
# (`isk`). Given other weights, cp_prove computes the same masks over other data, and two tables
# with equal masks reveal their difference. This does harm only if part of the result leaves the
# process. What must prevent that is a check within the table's opening, which sends the table's
# columns at drawn positions: the lowest stored Merkle node above every drawn column is recomputed
# and must match before any word of that column is written [hc-own-node]; and a batch that fails
# releases only empty `.failed` files.
# other_weights_nothing_leaves.sh replaces all the weights; this script makes the smallest change.
# Limits: the script looks at files, not at memory or at what unlinked files left on the disk. The
# opening's folded row (a linear combination of the table's rows) is computed over the changed
# weights and the committed masks; its root and answers are compared with nothing, and are withheld
# only because the batch fails as a whole. One weight and one bit are tried. The weights are
# synthetic: nothing real is at stake.
# usage: one_changed_weight.sh <programs> <work> <configuration> <model>, four directories
# (`folder` in messages):
#   <programs>       cp_commit, cp_prove, cp_verify of the production build (prover/build.sh's
#                    default): no other build accepts model files
#   <work>           new; made with mode 700, as is each missing directory above it. Name it below
#                    directories that only you can write to: the script does not check them. Gets a
#                    copy of the model files (530 MB for prover/examples/target_small_file_f2)
#   <configuration>  (`kit` in a message) policy.txt (parameter file), KL.bin (quantization scales)
#   <model>          written by make_model (prover/tests/weights_blocks/make_model.cpp) for the
#                    configuration's model.txt
# GPU_RUN in the environment: a command prefix for each call of cp_commit and cp_prove.
# Needs one H100 that nothing else uses during these calls. About three minutes.
# Last line, except at status 2 or if an argument is missing (bash's own message and status):
#   "one_changed_weight, version <policy.txt's version word>: ...".
# Status 0: all as expected; 1: the line counts the differences; 2: a directory was unusable.
# Started by the runner; by hand: below a directory of your own. (Runner: run_suite.py. The work folder goes there.)
set -u
unset CDPATH
made_here() { # <a folder that this script has just made>: it must be this account's, closed to others (mode 700; 2700 below a set-group-ID directory) and empty
  # One cd, then three tests of ".", so that all three see the same directory. If stat or ls fails,
  # the directory is rejected too. -P: cd takes ".." after a symbolic link as mkdir does, not by the
  # path's text. The slash: cd does not take a directory named "-" for the previous directory.
  (cd -P -- "$1/" && [ -O . ] && [[ $(stat -c %a .) == *700 ]] && list=$(ls -A) && [ -z "$list" ]) && return
  echo "one_changed_weight.sh: $1 is not the folder that was just made: another was put in its place: name a new folder below folders in which only you can write" >&2
  return 1
}
B=${1:?the programs\' folder} W=${2:?a work folder} K=${3:?a kit for the files} M=${4:?a model\'s folder}
here=$(cd -- "$(dirname "$0")" && pwd) || exit 2
case $W in /*) full=$W ;; *) full=$PWD/$W ;; esac; case /$full/ in */-*|*/../*|*[!A-Za-z0-9_./-]*) echo "one_changed_weight: the work folder's path may hold letters, digits and _ . / - only, and no part of it may be \"..\" or begin with \"-\": name another: $full"; exit 2 ;; esac # the path goes into sed commands below
if [ -e "$W" ]; then echo "one_changed_weight: the work folder $W exists: name a new one"; exit 2; fi
[ -f "$M/wsums.txt" ] && [ -f "$M/fold.txt" ] && ls "$M"/*.gguf >/dev/null 2>&1 || { echo "one_changed_weight: $M is no model's folder (wsums.txt, fold.txt, *.gguf)"; exit 2; }
(umask 077 && mkdir -p -- "$(dirname -- "$W")") && mkdir -m 700 -- "$W" && made_here "$W" && mkdir -p -- "$W/config" "$W/bent" "$W/in" || exit 2 # the mkdir without -p fails if anything has appeared at the name since the -e test on it above
full=$(cd -- "$W" && pwd) && [ "$full" -ef "$W" ] && W=$full || exit 2; M=$(cd -- "$M" && pwd) || exit 2 # -ef: stop unless cd reached the directory just made (a lone "-", or ".." after a symbolic link, can lead cd elsewhere)
export HOME=$W/home TMPDIR=$W/tmp; mkdir "$HOME" "$TMPDIR" || exit 2 # THE build's cp_prove wants both to name folders
V=$(awk '$1 == "version" { print $2; exit }' "$K/policy.txt") # the version word: H821 (p = 2^64 - 2^32 + 1)
bad=0
no() { echo "OTHERWISE: $*"; bad=$((bad + 1)); }
end() { if [ $bad = 0 ]; then echo "one_changed_weight, version $V: as foretold"; exit 0; fi; echo "one_changed_weight, version $V: $bad things otherwise"; exit 1; }
cp "$K/policy.txt" "$K/KL.bin" "$W/config/"
"$B/cp_verify" -plan arrays "$W/config" 128 >"$W/arrays.txt" 2>&1 # a planning call: its status is 4
python3 -B "$here/../tools/pack_layout.py" --arrays "$W/arrays.txt" 17 4 >"$W/config/layout.txt" 2>"$W/pack.err" || { no "the layout file cannot be made: $(head -c 300 "$W/pack.err")"; end; }
# Steps:
# 1. cp_commit as in production: a drawn seed, the worst-case check, the target of 128 bits.
#    `-M 1`: one leaf table.
# 2. `bent`: the model directory's copy with one bit flipped, at an offset found from the GGUF
#    file's header; wsums.txt (the weights' checksums) and fold.txt stay unchanged.
# 3. The request: T = 2^lt token ids, then r = T - 1, e = T: the proof is about the next
#    token only.
# 4. `same`, `one-bit`: two copies of `isk` and the request. In `one-bit`, isk.txt's line
#    `weights`, which cp_prove does not authenticate, names `bent`.
# 5. cp_prove on each copy.
#    Expected for `same`: status 0; example.proof and example.y, which cp_verify accepts; of the
#    files present before, only the leaf table's column count, count.bin (columns of the openings
#    begun), has changed.
#    Expected for `one-bit`: a status other than 0; the last line "p258 end 1 hc-own-node" (of this
#    failure the production build prints only the check's name); one new file, example.failed, of
#    0 bytes; no staging directory .tmp; of the files present before, only count.bin (a batch that
#    has begun is counted) and isk.txt (step 4) have changed.
# shellcheck disable=SC2086 # GPU_RUN is a list of words, here and below
${GPU_RUN:-} "$B/cp_commit" -weights "$M" -config "$W/config" -M 1 -ivk "$W/ivk" -isk "$W/isk" >"$W/cp_commit.log" 2>&1 || { no "cp_commit: status $?"; end; }
id=$(sed -n "s/^commitment ID: //p" "$W/cp_commit.log" | tr -d ' ')
[ -n "$id" ] || { no "cp_commit's log names no ID"; end; }
cp "$M"/* "$W/bent/"
python3 -I -B - "$W/bent" <<'END' || { no "no weight was changed: the play would say nothing"; end; }
# Flips the lowest bit of ONE quantized weight, in place. GGUF, version 3: magic, version, number of tensors, number of key-value pairs; the pairs; for each tensor its name, dimensions, type and offset;
# then the data, aligned. Q4_K (type 12): blocks of 144 bytes for 256 weights: d (2), dmin (2), scales (12), 128 bytes of two 4-bit quants each: the byte taken is the first of a block's quants.
import glob, struct, sys
SIZE = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
for path in sorted(glob.glob(sys.argv[1] + "/*.gguf")):
    f = open(path, "r+b")
    u32 = lambda: struct.unpack("<I", f.read(4))[0]
    u64 = lambda: struct.unpack("<Q", f.read(8))[0]
    text = lambda: f.read(u64()).decode()
    def value(kind):
        if kind == 8: return text()
        if kind == 9:
            k, n = u32(), u64()
            return [value(k) for _ in range(n)]
        return f.read(SIZE[kind])
    if f.read(4) != b"GGUF": sys.exit(path + " is no GGUF file")
    _, tensors, pairs = u32(), u64(), u64()
    alignment = 32
    for _ in range(pairs):
        key, kind = text(), u32()
        v = value(kind)
        if key == "general.alignment": alignment = struct.unpack("<I", v)[0]
    infos = []
    for _ in range(tensors):
        name, nd = text(), u32()
        dims = [u64() for _ in range(nd)]
        infos.append((name, dims, u32(), u64()))
    data = (f.tell() + alignment - 1) // alignment * alignment
    for name, dims, kind, offset in infos:
        cells = 1
        for d in dims: cells *= d
        if kind != 12 or cells < 512: continue
        number = cells // 256 // 2
        at = data + offset + number * 144 + 16
        f.seek(at); old = f.read(1)[0]; f.seek(at); f.write(bytes([old ^ 1])); f.close()
        print(f"changed: {name} (Q4_K, {cells} weights), block {number}: byte {at} of {path.rsplit('/', 1)[1]}: {old:#04x} -> {old ^ 1:#04x}")
        sys.exit(0)
sys.exit(1)
END
differ=0; for f in "$M"/*; do n=$(cmp -l "$f" "$W/bent/$(basename "$f")" | wc -l); differ=$((differ + n)); done
[ $differ = 1 ] || no "the bent folder differs from the model's in $differ bytes, not in one"
python3 -I -B - "$W/config/policy.txt" "$W/in/example.req" <<'END'
import struct, sys
word = dict(line.split() for line in open(sys.argv[1]) if len(line.split()) == 2)
T, V = 1 << int(word["lt"]), int(word["V"])
tokens = [(2654435761 * (t + 1) >> 7) % V for t in range(T)]
open(sys.argv[2], "wb").write(struct.pack("<%dI" % (T + 2), *tokens, T - 1, T))
END
for c in same one-bit; do mkdir "$W/$c" && cp -a "$W/isk" "$W/$c/isk" && cp -a "$W/in" "$W/$c/in" || exit 2; done
sed -i "s#^weights .*#weights $W/bent#" "$W/one-bit/isk/isk.txt"
[ "$(diff "$W/same/isk/isk.txt" "$W/one-bit/isk/isk.txt" | grep -c '^[<>]')" = 2 ] && [ "$(diff -rq "$W/same/isk" "$W/one-bit/isk" | wc -l)" = 1 ] || no "the two copies of the secret folder do not differ in that one line alone"
for c in same one-bit; do
  (cd -- "$W/$c" && find . -type f | LC_ALL=C sort) >"$W/$c.before"
  ${GPU_RUN:-} "$B/cp_prove" -isk "$W/$c/isk" -in "$W/$c/in" -out "$W/$c/out" >"$W/$c.log" 2>&1; rc=$?
  (cd -- "$W/$c" && find . -type f | LC_ALL=C sort) >"$W/$c.after"
  new=$(comm -13 "$W/$c.before" "$W/$c.after" | tr '\n' ' '); gone=$(comm -23 "$W/$c.before" "$W/$c.after" | tr '\n' ' ')
  moved=$(for f in $(comm -12 "$W/$c.before" "$W/$c.after"); do cmp -s "$W/$c/$f" "$W/$f" || printf '%s ' "$f"; done)
  echo "$c: status $rc; new files: ${new:-none}; gone: ${gone:-none}; moved against the commitment's: ${moved:-none}"
  if [ $c = same ]; then
    [ $rc = 0 ] || no "same: status $rc"
    [ -s "$W/$c/out/example.proof" ] && [ -s "$W/$c/out/example.y" ] || no "same: no proof or no reply"
    "$B/cp_verify" "$W/config" "$W/ivk" "$W/in/example.req" "$W/$c/out/example.y" "$W/$c/out/example.proof" -id "$id" >"$W/$c.verify.log" 2>&1 || no "same: cp_verify does not accept the response: $(tail -n 1 "$W/$c.verify.log" | cut -c1-160)"
    grep -a -q 'hc-own-node' "$W/$c.log" && no "same: the own test has spoken"
    [ "$moved" = "./isk/leaf/0/count.bin " ] || no "same: moved: $moved"
  else
    [ $rc != 0 ] || no "one bit: status 0"
    [ "$(tail -n 1 "$W/$c.log")" = "p258 end 1 hc-own-node" ] || no "one bit: the last line does not name the opening's own test: the run ended for another reason: $(tail -n 2 "$W/$c.log" | cut -c1-160 | tr '\n' '|')"
    [ "$new" = "./out/example.failed " ] || no "one bit: the new files are not the request's empty mark alone"
    [ -f "$W/$c/out/example.failed" ] && [ ! -s "$W/$c/out/example.failed" ] || no "one bit: example.failed is missing or holds bytes"
    [ -e "$W/$c/out/.tmp" ] && no "one bit: the staging folder is still there"
    [ "$moved" = "./isk/isk.txt ./isk/leaf/0/count.bin " ] || no "one bit: moved: $moved"
  fi
  [ -z "$gone" ] || no "$c: files are gone"
done
end
