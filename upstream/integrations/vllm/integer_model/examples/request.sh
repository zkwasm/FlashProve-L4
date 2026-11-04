#!/bin/sh
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Sends one request of token ids to the server that serve.sh or serve_gguf.sh started.
#
# usage: examples/request.sh NAME TOKEN_IDS
#   NAME       the served model's name; small_synthetic if omitted
#   TOKEN_IDS  the prompt, separated by commas; 17,400,3 if omitted
#
# The reply is one line of JSON, with the served token ids in choices[0].token_ids. With the
# examples' synthetic weights and the default prompt they are 365, 287, 374, 290
# (small_synthetic) and 93, 45, 59, 289 (small_gguf); for full_size they should be 42470, 62623,
# 33076, 45768, which no run at full size has checked. A non-empty VLLM_API_KEY is sent:
# small_gguf expects "example".
set -eu

usage() {
  echo "usage: $0 NAME TOKEN_IDS (both can be omitted)" >&2
  echo "TOKEN_IDS as in 17,400,3; NAME of letters, digits and _ . : / -" >&2
  exit 2
}
[ $# -le 2 ] || usage
name=${1:-small_synthetic} ids=${2:-17,400,3}
case $name in *[!A-Za-z0-9_.:/-]*) usage ;; esac
case $ids in *[!0-9,]* | ,* | *, | *,,*) usage ;; esac

body="{\"model\": \"$name\", \"prompt\": [$ids], \"max_tokens\": 4, \"temperature\": 0"
body="$body, \"return_token_ids\": true}"
set -- -sS --noproxy 127.0.0.1 127.0.0.1:8000/v1/completions -d "$body" \
  -H 'Content-Type: application/json'
if [ -n "${VLLM_API_KEY-}" ]; then
  # Standard input keeps the key out of the process list.
  printf 'Authorization: Bearer %s\n' "$VLLM_API_KEY" | curl "$@" -H @-
else
  curl "$@"
fi
