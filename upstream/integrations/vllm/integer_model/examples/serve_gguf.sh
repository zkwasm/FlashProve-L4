#!/bin/sh
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Serves a model from GGUF files with vLLM on 127.0.0.1:8000 until Ctrl-C, only if its five small
# files have the SHA-256 digests of your reference copies. The script removes nothing. This is step
# 4 of integrations/vllm/integer_model/OPERATING.md, "Serve from GGUF files". The digests cover no
# GGUF file. The script takes and writes none of them: it gives vLLM the reference directory's path,
# and the package's loader takes the digests there, in memory. The server's new working directory
# (mode 700), in which the script puts nothing, stays after the server has ended. The script needs
# python3 and must stand at its place in the checkout: the package's own routine checks the
# directory for temporary files ("folder" in messages) first.
#
# usage: VLLM_API_KEY=KEY examples/serve_gguf.sh REFERENCE_DIRECTORY MODEL_DIRECTORY NAME [LENGTH]
#          [--middleware MODULE.CLASS]...
#   REFERENCE_DIRECTORY, MODEL_DIRECTORY  the directories K and M of steps 1 to 3
#   NAME, VLLM_API_KEY   the model's name for clients, and vLLM's API key
#   LENGTH               --max-model-len: at most the model's window length; 1024 if omitted
set -eu

misplaced() {
  echo "$0: run this script by its path inside a checkout of FlashProve" >&2
  exit 1
}
tools=/usr/sbin:/usr/bin:/sbin:/bin${PATH:+:$PATH} # for tr and realpath: the system's come first
# A "$0" without a slash is refused. Sourced or read from standard input, it is the shell's name.
# Else the shell may have found the file through PATH, but realpath looks in the working directory.
case $0 in
*/*) ;;
*)
  said='by a path with a slash, such as ./serve_gguf.sh or its full path. It cannot be'
  printf '%s: start serve_gguf.sh %s sourced or read from standard input.\n' \
    "$(printf %s "$0" | LC_ALL=C PATH=$tools tr -c ' -~' '?')" "$said" >&2
  exit 2
  ;;
esac
script=$(PATH=$tools realpath -- "$0") || misplaced
case $script in
*/integrations/vllm/integer_model/examples/serve_gguf.sh) ;;
*) misplaced ;;
esac
# Where vllm is not on PATH, find_or_give_root_up.sh beside this script takes over: it takes vllm
# from .venv/bin beside the package, or, as root, stops and says what works. It is started only
# as a regular file with this script's owner and group and no write permission for group or others
# that this script lacks, in a directory in which others may not write. Otherwise the script stops.
if ! command -v vllm >/dev/null 2>&1; then
  about() { PATH=/usr/sbin:/usr/bin:/sbin:/bin stat -c "$1" -- "$2" 2>/dev/null; }
  helper=${script%/*}/find_or_give_root_up.sh
  if [ -f "$helper" ] && [ ! -L "$helper" ] && whose=$(about %u:%g "$script") &&
    [ "$(about %u:%g "$helper")" = "$whose" ] && mine=$(about %a "$script") &&
    its=$(about %a "$helper") && folder=$(about %a "${script%/*}") &&
    [ "$((0$its & 022 & ~0$mine | 0$folder & 002))" = 0 ]; then
    exec /bin/sh "$helper" vllm "$0" "$script" "$@"
  fi
  printf '%s %s %s %s\n' \
    "$0: vllm was not found on PATH. find_or_give_root_up.sh beside this script, which takes" \
    'over in that case, is missing, is a link, or it or its folder fails the check of owner,' \
    'group and permissions. Nothing was made. See' \
    'integrations/vllm/integer_model/README.md, "Install and quick start".' >&2
  exit 2
fi

usage() {
  echo "usage: VLLM_API_KEY=KEY $0 REFERENCE_DIRECTORY MODEL_DIRECTORY NAME LENGTH" >&2
  echo "LENGTH, the value of --max-model-len, can be omitted; it is then 1024" >&2
  echo "then any number of pairs: --middleware MODULE.CLASS (letters, digits, _ and . only)" >&2
  echo "every other option is rejected on purpose" >&2
  exit 2
}
if [ $# -lt 3 ] || [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]; then usage; fi
reference=$1 model=$2 name=$3 length=1024
shift 3
if [ $# -gt 0 ] && [ "$1" != --middleware ]; then
  length=${1:-1024}
  shift
fi
case $length in -*) usage ;; esac
# Only --middleware pairs may follow: other options, such as --trust-remote-code, could undo the
# script's precautions.
expected=option
for word in "$@"; do
  case $expected:$word in
  option:--middleware) expected=value ;;
  value: | value:*[!A-Za-z0-9_.]* | option:*) usage ;;
  *) expected=option ;;
  esac
done
[ $expected = option ] || usage
# The three vLLM environment variables tested below can undo this script's precautions, so none
# may be set, even to the empty string (integrations/vllm/integer_model/OPERATING.md, "Settings").
refuse_setting() {
  echo "$0: $1 is set, which can undo this script's precautions. Unset it." >&2
  exit 2
}
[ -z "${VLLM_SERVER_DEV_MODE+1}" ] || refuse_setting VLLM_SERVER_DEV_MODE
[ -z "${VLLM_MODEL_REDIRECT_PATH+1}" ] || refuse_setting VLLM_MODEL_REDIRECT_PATH
[ -z "${VLLM_LOGGING_LEVEL+1}" ] || refuse_setting VLLM_LOGGING_LEVEL
# Despite PYTHONSAFEPATH=1, an empty entry in PYTHONPATH puts the working directory on Python's
# import path, and a relative entry adds a directory that this script does not choose. An empty
# PYTHONPATH adds nothing.
if [ -n "${PYTHONPATH-}" ]; then
  case ":$PYTHONPATH:" in
  *::* | *:[!/]*)
    echo "$0: PYTHONPATH has an empty or relative entry; use absolute paths, or unset it." >&2
    exit 2
    ;;
  esac
fi
: "${VLLM_API_KEY:?set it to an API key of your choosing}"
given=$model
unset CDPATH # so that cd takes a relative path from the current directory alone
model=$(cd -- "$given" && pwd) || {
  echo "$0: cannot enter the model directory $given" >&2
  exit 1
}

# The path is made absolute by words alone and not resolved here, so that the look below and the
# loader go by one and the same name. It goes into JSON at the script's end, unescaped.
case $reference in /*) ;; *) reference=$PWD/$reference ;; esac
case $reference in *[\"\\]* | *[[:cntrl:]]*)
  echo "$0: the reference directory's path, or the current directory's before a relative one," \
    "has a quote, a backslash or a control character, which this script cannot hand on to vLLM." \
    "Nothing was made. Give the directory by a path that has none." >&2
  exit 2
  ;;
esac
for file in integer_model.json policy.txt KL.bin wsums.txt fold.txt; do
  if [ ! -r "$reference/$file" ]; then
    echo "$0: cannot read the reference copy $reference/$file" >&2
    exit 1
  fi
done

# The server runs in a new directory, also its TMPDIR and HOME, so that the caller's directory is
# not on Python's path. vLLM is given that directory's path, so no other account may be able to
# change the directory that holds it, or any directory above that one. The package's own routine,
# others_can_change() in integer_model_vllm/example.py, checks this first, from this checkout.
place=${TMPDIR:-/tmp}
case $place in /*) ;; *) place=$PWD/$place ;; esac
# The script does not first test whether the directory exists: the routine refuses a missing one,
# and another account could make it between such a test and mktemp. Without SUDO_UID: under sudo,
# root's server runs there, so the account that ran sudo is not trusted with that directory either.
# python3 -I takes no PYTHON* variable, no user site and not the working directory; -B writes no
# __pycache__ into the checkout.
# It prints the directory's real path, which it has looked at too, and mktemp draws there: the
# script goes on by the path that it has seen, not by the one that it was given.
real=$(env -u SUDO_UID python3 -I -B -c '
import os, sys, unicodedata
sys.path.insert(0, sys.argv[1])
from integer_model_vllm.example import others_can_change, refusal
place = sys.argv[2]
real = os.path.realpath(place)
for folder in (place, real):
    found = others_can_change(folder)
    if found and found[1] == "does not exist":
        sys.exit(f"{sys.argv[3]}: TMPDIR, {place}, does not exist. Make it below folders that only "
                 "you or root can change, or unset TMPDIR.")
    if found:
        words = refusal("folder for temporary files", folder, *found)
        sys.exit(f"{sys.argv[3]}: {words}. Set TMPDIR to such a folder.")
if os.path.realpath(real) != real:  # checked from the top: no part can turn into a link now
    sys.exit(f"{sys.argv[3]}: TMPDIR changed while it was looked at. Nothing was made.")
# The path is printed below as it is, and vLLM and the programs that it starts get it as HOME and
# TMPDIR: none of them was tried with one of these characters in it.
if any(c in real for c in "\"\\") or any(unicodedata.category(c) == "Cc" for c in real):
    sys.exit(f"{sys.argv[3]}: the folder for temporary files, {real!r}, has a quote, a backslash "
             "or a control character in its path; set TMPDIR to an absolute path without one.")
# The bytes as they are (also where they are no UTF-8), and a / after them: a newline at the end of
# the path would be cut by the command substitution.
sys.stdout.buffer.write(os.fsencode(real) + b"/")
' "${script%/examples/serve_gguf.sh}" "$place" "$0") || exit 2
real=${real%/}
work=$(TMPDIR=$real mktemp -d)
case $work in /*) ;; *) work=$PWD/$work ;; esac # mktemp prints a relative path if TMPDIR is one
cd -P -- "$work" # -P: as the system resolves the path, so a link followed by .. cannot mislead it
echo "The server's working directory, TMPDIR and HOME: $work"
export VLLM_API_KEY VLLM_USE_V2_MODEL_RUNNER=1 VLLM_NO_USAGE_STATS=1 GLOO_SOCKET_IFNAME=lo
export PYTHONSAFEPATH=1 TMPDIR="$work" HOME="$work"
# vLLM gets the reference directory's path and no digest: whoever has the digest of wsums.txt can
# test guesses of the weights against it, every account on the machine can read a command line, and
# a file of digests would outlast the server. The loader takes the digests itself, in memory.
# "reference_files": 5 demands all five copies, whatever lies in the directory at the load.
exec vllm serve "$model" "$@" --host 127.0.0.1 --served-model-name "$name" \
  --load-format integer_model --max-num-seqs 1 --max-model-len "$length" \
  --no-enable-prefix-caching --enforce-eager --generation-config vllm \
  --fingerprint-mode none --max-logprobs 0 \
  --model-loader-extra-config "{\"reference_directory\": \"$reference\", \"reference_files\": 5}"
