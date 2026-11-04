# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""A fake cp_verify for the tests. It checks nothing.

So that its output cannot be mistaken for a verifier's, every line on standard output ends with
" (fake)", which the tests strip, and it exits with code 2 if FAKE_CP_VERIFY is unset or empty.

It answers integrations/check_proof.py's calls ("-plan id" and the verification call) with the lines
that the script reads, as cp_verify prints them when every check passes; the other lines are filler.
A missing or unneeded "-policy" gives "NO: policy"; "-bits" below 128 gives "NOT-A-VERIFICATION".
It shares no code with integrations/check_proof.py.

The environment variable FAKE_CP_VERIFY is a JSON object with optional keys:
  log                       file for one JSON line per call: arguments, directory, files, names of
                            environment variables
  plan_id, plan_status      the commitment ID that "-plan id" prints; its exit code instead of 4
  plan_rejects              the only line of "-plan id", which then exits with 1
  plan_second_id            true: "-plan id" prints a second commitment ID, 64 zeros
The other keys change the verification call:
  status, last_line         the exit code; the last line
  output                    file whose text replaces the usual output, before "replace"
  test_build                true: the output starts with a "TEST BUILD:" line, like a test build's
  work                      true: the output reports 32 bits of proof of work
  assumed                   number of commitment rounds in the line "commitment:" and the field
                            commitment-assumed-rounds=, instead of ASSUMED
  result_v1                 {field: value} for the "result-v1" line
  copy_result_v1            true: the request stage also prints the "result-v1" line
  replace                   [[old, new], ...]: old must occur exactly once
  cut_at                    the output ends before this text; applied after "replace"
  create                    {path: text}: files written before any output
  stderr                    a line for standard error
"""

import hashlib
import json
import os
import re
import struct
import sys

STATUS_WITHIN_LIMITS = 0  # as in check_proof.py
BUILD = "b1" * 32
WORK = (
    "proof of work: up to 32 bits counted in 14 rounds; with it the least round has 128.02 bits, "
    "without it 96.02 bits"
)
ASSUMED = 49
SOURCE_WORDS = " on another source than the bound within a third of the distance"
RECORD_CLAUSE = (
    "; the table has a record: the binding of the weights rests on the theorem on codes "
    "(ePrint 2025/2055, Theorem 4.6), taken as given"
)
MARK = " (fake)"
CHANGES = json.loads(os.environ.get("FAKE_CP_VERIFY") or "{}")

OUTPUT = """\
---- stage key ---- 1 thread(s)
computed commitment ID: {id}
result: asked={bits} by=least-row-and-sum files=1 members=1 build={build} dw=8 (fake)
STAGE PASSED: key
---- stage leaf ---- 1 thread(s)
computed commitment ID: {id}
leaf_root={build}
result: asked={bits} by=least-row-and-sum build={build} dw=8 (fake)
STAGE PASSED: leaf
---- stage request ---- 4 thread(s)
leaf number {leaf_number}
weight hiding: not checked in this run (fake)
result: word=1 hb=139 t=335 asked={bits} by=least-row-and-sum T={T} lim=- C={C} theta={theta} \
cap={cap} s={s} sh={sh} sT={sT} F={F} leaves=4 place=1 card=0 judged=1 of 1 dj={build} \
build={build} dw=8 pow=0 pows=0 powc=0 (fake)
STAGE PASSED: request
---- the call ----
accounting: the largest round error (fake)
commitment ID: {id}
id: named by the caller and equal
weight hiding: not checked in this call (fake)
covered: r={r} e={e} T={T} C={C} theta={theta} cap={cap} s={s} sh={sh} positions={r}..{end} \
tokens={tokens} y={y} next-token={next_token} policy={policy} above-clip={above_clip} (fake)
meaning: every check of this program passed.
{on_commitment}
{on_work}
result-v1 status={status} claim={claim} id={id} bits={bits} work={work} work-rounds={rounds} \
least-round=128.02 least-round-without-work={without} commitment-assumed-rounds={assumed} record=1 \
r={r} e={e} T={T} tokens={tokens} y={y} \
policy={policy} C={C} theta={theta} cap={cap} s={s} sh={sh} sT={sT} F={F} \
above-clip={above_clip} leaf={leaf_number} stages=key,leaf,request dw=8 build={build}
{last_line}
"""


def write(text):
    """Print text, ending every line with MARK."""
    lines = text.split("\n")
    rest = lines.pop()
    sys.stdout.write("".join(line + MARK + "\n" for line in lines) + (rest and rest + MARK))


def read(*path):
    with open(os.path.join(*path), "rb") as f:
        return f.read()


def on_commitment(rounds):
    """Return cp_verify's line on the commitment's one-time files, up to its clause."""
    if rounds == 1:
        return "commitment: 1 closeness round of the one-time files rests" + SOURCE_WORDS
    return "commitment: %d closeness rounds of the one-time files rest" % rounds + SOURCE_WORDS


def commitment_id(parameter_file, key):
    """Return the fake's commitment ID for a parameter file's and key.bin's bytes."""
    return hashlib.sha256(parameter_file + key).hexdigest()


def log_call():
    names = [os.path.join(where, name) for where, _, names in os.walk(".") for name in names]
    files = {
        os.path.relpath(name): "symlink" if os.path.islink(name) else read(name).hex()
        for name in names
    }
    with open(CHANGES["log"], "a") as f:
        call = {"arguments": sys.argv[1:], "directory": os.getcwd(), "files": files}
        json.dump({**call, "environment": sorted(os.environ)}, f)
        f.write("\n")


def plan(what, *paths):
    if what == "id" and "plan_rejects" in CHANGES:
        write(CHANGES["plan_rejects"] + "\n")
        sys.exit(1)
    assert what == "id"
    files_id = commitment_id(read(paths[0], "policy.txt"), read(paths[1]))
    write("commitment ID: " + CHANGES.get("plan_id", files_id) + "\n")
    if CHANGES.get("plan_second_id"):
        write("commitment ID: " + "0" * 64 + "\n")
    write("PLAN ONLY: no proof was checked\n")
    sys.exit(CHANGES.get("plan_status", 4))


def ending(v, option):
    """Return a verification call's last line and exit code."""
    limits = "C={C} theta={theta} cap={cap} s={s} sh={sh}".format(**v)
    needs_policy = v["tokens"] != "none" and v["policy"] != "exact-match"
    if needs_policy != ("-policy" in option):
        return "NO: policy", 1
    if int(v["bits"]) < 128:
        return "NOT-A-VERIFICATION", 14
    if needs_policy:
        assert option["-policy"] == "clipped-gap,{C},{theta},{cap},{s},{sh}".format(**v)
        return "WITHIN LIMITS: clipped-gap " + limits, STATUS_WITHIN_LIMITS
    return "YES", 0


def set_fields(text, line_start, fields):
    """Return text with new "name=value" fields in the line that starts with line_start."""
    for name, value in fields.items():
        pattern = "(?m)^(%s.* %s)=\\S+" % (re.escape(line_start), re.escape(name))
        text, count = re.subn(pattern, lambda found: found[1] + "=" + value, text)
        assert count == 1, name
    return text


def changed(text):
    """Return a verification call's output with FAKE_CP_VERIFY's changes."""
    text = set_fields(text, "result-v1", CHANGES.get("result_v1", {}))
    if CHANGES.get("copy_result_v1"):
        line = re.search("(?m)^result-v1 .*\n", text)[0]
        text = text.replace("STAGE PASSED: request\n", line + "STAGE PASSED: request\n")
    if CHANGES.get("test_build"):
        text = "TEST BUILD: no run of this program is a verification\n" + text
    if "output" in CHANGES:
        text = read(CHANGES["output"]).decode()
    for old, new in CHANGES.get("replace", []):
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    if "cut_at" in CHANGES:
        text = text[: text.index(CHANGES["cut_at"])]
    return text


def verify(config, public_key_dir, request_file, _token_file, proof, *options):
    option = dict(zip(options, options[1:] + ("",)))
    for path, text in CHANGES.get("create", {}).items():
        with open(path, "w") as f:
            f.write(text)
    words = [line.split() for line in read(config, "policy.txt").decode().splitlines()]
    v = {pair[0]: int(pair[1]) for pair in words if len(pair) == 2 and pair[1].isdigit()}
    request = read(request_file)
    length = len(request) // 4 - 2
    r, e = struct.unpack_from("<2I", request, 4 * length)
    checked = min(e, length - 1) - r  # tokens under the policy
    exact = (v["C"], v["theta"], v["cap"], v["sT"]) == (256, 256, 0, 4)
    above_clip = min(v["cap"], checked)
    if v["C"] > 256:
        above_clip = min(above_clip, checked * max(v["theta"] - 256, 0) // (v["C"] - 256))
    leaf_number = struct.unpack("<I", read(proof)[8:12])[0]
    assumed = CHANGES.get("assumed", ASSUMED)
    files_id = commitment_id(read(config, "policy.txt"), read(public_key_dir, "key.bin"))
    v.update(
        {"T": length, "r": r, "e": e, "end": e - 1},
        id=files_id,
        bits=option["-bits"],
        build=BUILD,
        leaf_number=leaf_number,
        work=32 if CHANGES.get("work") else 0,
        rounds=14 if CHANGES.get("work") else 0,
        without="96.02" if CHANGES.get("work") else "128.02",
        on_work=WORK if CHANGES.get("work") else "proof of work: none counted",
        assumed=assumed,
        on_commitment=on_commitment(assumed) + RECORD_CLAUSE,
        claim="next-token" if not checked else ("exact-match" if exact else "policy"),
        tokens="%d..%d" % (r + 1, r + checked) if checked else "none",
        y="yes" if e == length else "no",
        next_token="yes" if r + 1 == length else "no",
        policy="exact-match" if exact else "clipped-gap",
        above_clip=above_clip,
    )
    last_line, status = ending(v, option)
    text_of_last_line = CHANGES.get("last_line", last_line)
    text = OUTPUT.format(**v, status=status, last_line=text_of_last_line)
    if CHANGES.get("status", status) == 1:  # a rejection closes with the stage's name alone
        text = text[: text.index("---- the call ----\n")]
        text += "---- the call ----\nrefused by the stage request\n%s\n" % text_of_last_line
    write(changed(text))
    if "stderr" in CHANGES:
        print(CHANGES["stderr"], file=sys.stderr)
    sys.exit(CHANGES.get("status", status))


if __name__ == "__main__":
    if not os.environ.get("FAKE_CP_VERIFY"):
        print("fake_cp_verify.py is a stand-in for tests: it checks nothing", file=sys.stderr)
        sys.exit(2)
    if "log" in CHANGES:
        log_call()
    if sys.argv[1] == "-plan":
        plan(*sys.argv[2:])
    verify(*sys.argv[1:])
