#!/bin/sh
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Creates an example in a new temporary directory; vLLM serves it on 127.0.0.1:8000 until Ctrl-C.
# The directory stays. vLLM's options: integrations/vllm/integer_model/OPERATING.md, "Settings".
set -eu

refuse() {
  echo "$0: run this script by its path inside a checkout of FlashProve, as in" >&2
  echo "  /path/to/flashprove/integrations/vllm/integer_model/examples/serve.sh small_synthetic" >&2
  exit 1
}
tools=/usr/sbin:/usr/bin:/sbin:/bin${PATH:+:$PATH} # for tr and realpath: the system's come first
# A "$0" without a slash is refused. Sourced or read from standard input, it is the shell's name.
# Else the shell may have found the file through PATH, but realpath looks in the working directory.
case $0 in
*/*) ;;
*)
  said='by a path with a slash, such as ./serve.sh or its full path. It cannot be'
  printf '%s: start serve.sh %s sourced or read from standard input.\n' \
    "$(printf %s "$0" | LC_ALL=C PATH=$tools tr -c ' -~' '?')" "$said" >&2
  exit 2
  ;;
esac
script=$(PATH=$tools realpath -- "$0") || refuse
case $script in
*/integrations/vllm/integer_model/examples/serve.sh) package=${script%/examples/serve.sh} ;;
*) refuse ;;
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
# Keeps the caller's directory off Python's import path (Python 3.11 and later), also in vLLM's
# processes. It has no effect on PYTHONPATH's entries (checked below). PYTHONNOUSERSITE keeps the
# user site under HOME out of them too: under sudo with HOME kept, it would be another account's.
export PYTHONSAFEPATH=1 PYTHONNOUSERSITE=1
case ${1-} in
small_synthetic) length=64 ;;
full_size) length=1024 ;;
small_gguf) length=64 ;;
small_float_pair) length=64 ;;
*)
  echo "usage: $0 NAME" >&2
  echo "NAME is small_synthetic, full_size, small_gguf or small_float_pair" >&2
  exit 2
  ;;
esac

# The three vLLM variables below must be unset, not merely empty. What each can undo:
# integrations/vllm/integer_model/OPERATING.md, "Settings".
refuse_setting() {
  echo "$0: $1 is set, which can undo this script's precautions. Unset it." >&2
  exit 2
}
[ -z "${VLLM_SERVER_DEV_MODE+1}" ] || refuse_setting VLLM_SERVER_DEV_MODE
[ -z "${VLLM_MODEL_REDIRECT_PATH+1}" ] || refuse_setting VLLM_MODEL_REDIRECT_PATH
[ -z "${VLLM_LOGGING_LEVEL+1}" ] || refuse_setting VLLM_LOGGING_LEVEL
# Python resolves an empty or relative entry of PYTHONPATH against the working directory, despite
# PYTHONSAFEPATH=1, so a file there or, through .., above it could be imported. An empty PYTHONPATH
# counts as unset.
if [ -n "${PYTHONPATH-}" ]; then
  case ":$PYTHONPATH:" in
  *::* | *:[!/]*)
    echo "$0: PYTHONPATH has an empty or relative entry; use absolute paths, or unset it." >&2
    exit 2
    ;;
  esac
fi

# A relative TMPDIR means the caller's directory: made absolute here, before the cd below.
case ${TMPDIR-} in '' | /*) ;; *) TMPDIR=$PWD/$TMPDIR && export TMPDIR ;; esac
# The imported package must be this script's, since small_gguf builds a program from the checkout
# around it. So only `$package` is on PYTHONPATH, and python3 -m starts there, where Python before
# 3.11 looks first. The variable keeps example.py from starting again under another Python.
directory=$(
  cd -- "$package" && INTEGER_MODEL_ENVIRONMENT_TRIED=1 PYTHONPATH=$package \
    python3 -B -s -m integer_model_vllm.example "$1" --quiet
)
echo "The example's files are in $directory"
# vLLM starts in the example's new directory, which is also its HOME (for small_gguf, in a folder
# that serve_gguf.sh makes): under sudo with HOME kept, the caller's would be another account's.
cd -- "$directory"
export HOME="$directory"

if [ "$1" = small_gguf ]; then
  export VLLM_API_KEY=example
  exec "$package/examples/serve_gguf.sh" "$directory/reference" "$directory/model" "$1" "$length"
fi

if [ "$1" = small_float_pair ]; then
  export VLLM_PLUGINS='' VLLM_NO_USAGE_STATS=1 GLOO_SOCKET_IFNAME=lo
  exec vllm serve "$directory/float_model" --host 127.0.0.1 --served-model-name "$1" \
    --dtype bfloat16 --max-model-len "$length" --moe-backend triton --enforce-eager \
    --generation-config vllm
fi

export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_NO_USAGE_STATS=1 GLOO_SOCKET_IFNAME=lo
exec vllm serve "$directory/model" --host 127.0.0.1 --served-model-name "$1" \
  --load-format integer_model --max-num-seqs 1 --max-model-len "$length" \
  --no-enable-prefix-caching --enforce-eager --generation-config vllm
