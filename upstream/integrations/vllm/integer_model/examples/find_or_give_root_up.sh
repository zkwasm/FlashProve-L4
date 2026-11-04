#!/bin/sh
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# The package's scripts start this file when they do not find vllm, or a Python package that they
# need. As an ordinary account it takes .venv/bin beside the package and starts the caller again. As
# root it stops and says what works. It writes no file.
#
# usage: find_or_give_root_up.sh vllm SCRIPT PLACE [ARGUMENT]...
#        find_or_give_root_up.sh python3 PACKAGE PYTHON SCRIPT [WORD]...
#   SCRIPT            the caller, as it was called
#   PLACE             its physical path: a /bin/sh script, which is started again with the ARGUMENTs
#   PACKAGE, PYTHON   the missing Python package, and the interpreter that did not find it
#   WORD              what followed the interpreter's name on its command line
set -eu

# This file's own tools come from system directories, and it never changes the working directory.
system=/usr/sbin:/usr/bin:/sbin:/bin
callers_path=${PATH:-$system}
export PATH=$system
level() { printf %s "$1" | LC_ALL=C tr -c ' -~' '?'; } # only printable ASCII reaches a terminal
# The scripts start this file by its full path. Read from standard input, "$0" would be the name of
# the shell, which says nothing of this file's place.
by_hand() {
  printf '%s: the scripts of the package start this file; it is not meant to be run by hand\n' \
    "$(level "$0")" >&2
  exit 2
}
case $0 in
*/*) ;;
*) by_hand ;;
esac
if [ "${1-}" = vllm ] && [ $# -ge 3 ]; then
  missing='vllm was not found on PATH' caller=$2
elif [ "${1-}" = python3 ] && [ $# -ge 4 ]; then
  missing="the Python package $(level "$2") was not found by $(level "$3")" caller=$4
else
  by_hand
fi
say() { printf '%s: %s\n' "$(level "$caller")" "$*" >&2; }
stop() {
  say "$@"
  exit 2
}
uid=$(id -u)
[ "$uid" != 0 ] || stop "started as root, and $missing (sudo may drop an activated environment's" \
  "PATH), so it stops here. Run the same command as the checkout's owner, without sudo. If it" \
  "must be root, activate an environment of root's own, in a checkout of root's own; if you" \
  "activate another account's environment in a root shell instead, root runs whatever that" \
  'environment holds. See integrations/vllm/integer_model/OPERATING.md, "As root or under sudo".'
make='Nothing was made. Make the virtual environment as integrations/vllm/integer_model/README.md'
make="$make says, or activate yours."
this=$(realpath -- "$0" 2>/dev/null) || stop "$(level "$0") could not be examined. $make"
bin=${this%/*/*}/.venv/bin

# The caller is started again only once: if it comes back here, .venv/bin has not helped.
if [ -n "${INTEGER_MODEL_ENVIRONMENT_TRIED-}" ]; then
  [ "$1" != vllm ] || stop "vllm was still not found with $(level "$bin") first on PATH. $make"
  stop "$missing either. $make"
fi
there() { [ -f "$bin/$1" ] && [ -x "$bin/$1" ]; }
export INTEGER_MODEL_ENVIRONMENT_TRIED=1
if [ "$1" = vllm ]; then
  there vllm || stop "$missing, nor in $(level "$bin"). $make"
  say "$missing: $(level "$bin") is now first on PATH."
  PATH=$bin:$callers_path
  shift 2
  exec /bin/sh "$@"
fi
there python3 || stop "$missing, and $(level "$bin") has no python3. $make"
say "$missing: starting again with $(level "$bin")/python3."
PATH=$callers_path
shift 4
exec "$bin/python3" "$@"
