#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.

"""Stand-ins for cp_prove, cp_verify and nvidia-smi, for driver/tests/test_measure_rate.py only.
They prove, check and measure nothing. A test that uses them shows how driver/measure_rate.py
behaves, not how the three programs do. Started by driver/tests/test_measure_rate.py, which the
runner starts; by hand: below a directory of your own.

usage: stand_in_programs.py <cp_prove | cp_verify | nvidia-smi> <scene file> <the program's own arguments>

The test writes the scene file, JSON, into its own temporary directory. Each stand-in takes only
the options that the pages document for its program and that the script may pass, and ends with a
refusal at any other. It appends what it did to the file "record" beside the scene file, one JSON
object a line. The stand-in for cp_prove works only on a spool directory below the scene file's
directory, so it moves and removes nothing outside the test's temporary directory.

Keys of the scene, all optional but "tokens" and "id":
  tokens, id             T, and the commitment ID that cp_verify expects
  plan_ids, plan_status  what cp_verify -plan id prints and its exit code, instead of id and 4
  start_up_s, end_s      sleeps before the first batch is taken and after the last release
  work_s                 a list: the sleep in each batch. A batch is released after the next is taken
  stop_goes_after_s      the start removes a stop file only after this sleep, and does not obey it
  first_batch_lacks      the first batch takes this many requests fewer than -batch
  in_is_away_s           once the first batch is taken, in/ has another name for so long
  fail, refuse           SHA-256 sums of requests: cp_prove answers "failed"; cp_verify says NO
  wrong_tag              the same: the response's tag is not that of its bytes
  within_limits          the same: cp_verify ends with exit code 0 and a last line that is not YES
  check_s                {SHA-256 sum of a request: cp_verify sleeps so long before it answers}
  stamp_off_ms           added to the modification time that a batch's files get
  exit_code, last_line   how cp_prove ends after its batches (default 0, "p258 end 0")
  in_use                 true: cp_prove refuses to start, as on a secret key directory in use
  gpu_mib                a list: the memory in use that successive samples report, then its last entry
  describe_s             nvidia-smi sleeps so long before it describes the GPU
  late_sample_s          nvidia-smi sleeps so long before every sample but the first"""

import hashlib
import json
import os
import re
import resource
import signal
import struct
import sys
import time

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import spool  # noqa: E402

ROLE, SCENE_FILE, WORDS = sys.argv[1], os.path.realpath(sys.argv[2]), sys.argv[3:]
HOME = os.path.dirname(SCENE_FILE)
ABOUT_HOME = os.lstat(HOME)
if ABOUT_HOME.st_uid != os.geteuid() or ABOUT_HOME.st_mode & 0o077:
    sys.exit("stand_in_programs.py: the scene file's directory, %s, must be yours alone (mode 0700), as a test's temporary directory is" % HOME)
with open(SCENE_FILE, encoding="utf-8") as scene_file:
    SCENE = json.load(scene_file)
# A test may have set a low limit on the size of the files that the script writes. It is not meant for the stand-ins.
resource.setrlimit(resource.RLIMIT_FSIZE, (resource.getrlimit(resource.RLIMIT_FSIZE)[1],) * 2)


def note(**what):
    with open(os.path.join(HOME, "record"), "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(what, role=ROLE, ns=time.time_ns())) + "\n")


def say(line):
    print(line, flush=True)


def proof_of(request):
    """Returns the stand-in's proof of a request (given without its nonce); its length varies with the request."""
    digest = hashlib.sha256(request).digest()
    return b"stand-in proof" + digest + bytes(digest[0])


def options_of(words, with_value, alone):
    """Returns {option: value} ({option: True} for one that stands alone), or the first word that is neither."""
    found = {}
    while words:
        word = words.pop(0)
        if word in alone:
            found[word] = True
        elif word in with_value and words:
            found[word] = words.pop(0)
        else:
            return word
    return found


def cp_prove():
    o = options_of(list(WORDS), ("-isk", "-spool", "-batch", "-in-step", "-most-batches", "-memory", "-public-weights"), ("-huge-pages",))
    if not isinstance(o, dict):
        say("cp_prove: unknown option " + o)
        say("p258 end 1 usage")
        return 1
    folders = {name: os.environ.get(name, "") for name in ("HOME", "TMPDIR")}
    note(event="call", options=o, folders=folders, leads_a_session=os.getsid(0) == os.getpid(), process_id=os.getpid())
    for number in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(number, lambda received, frame: note(event="signal", number=received))
    if not all(re.fullmatch(r"/[A-Za-z0-9/._-]{0,399}", folder) and os.path.isdir(folder) for folder in folders.values()):
        say("refused (cp-prove-environment): HOME and TMPDIR must each name a folder that exists")
        say("p258 end 1 cp-prove-environment")
        return 1
    if SCENE.get("in_use"):
        say("refused (cp-isk): the folder named with -isk is in use: another cp_prove runs on it")
        say("p258 end 1 cp-isk")
        return 1
    top = os.path.realpath(o["-spool"])
    if not top.startswith(HOME + os.sep):
        say("refused (stand-in): a spool directory outside the test's directory")
        say("p258 end 1")
        return 1
    here = {name: os.path.join(top, *name.split("/")) for name in ("in", "in/.tmp", "work", "out", "out/.tmp", "stop")}
    for name in ("in", "in/.tmp", "work", "out", "out/.tmp"):
        os.makedirs(here[name], mode=0o700, exist_ok=True)
    time.sleep(SCENE.get("stop_goes_after_s", 0))
    if os.path.lexists(here["stop"]):
        os.remove(here["stop"])
        note(event="the start removed a stop file")
    say("p258 closed")
    time.sleep(SCENE.get("start_up_s", 0))

    def take(most):
        """Moves the oldest requests to work/, by modification time and then by name, and returns {id: the file's bytes}."""
        waiting = sorted((os.stat(os.path.join(here["in"], x)).st_mtime_ns, x) for x in os.listdir(here["in"]) if x.endswith(".req"))
        taken = {}
        for _, file_name in waiting[:most]:
            os.rename(os.path.join(here["in"], file_name), os.path.join(here["work"], file_name))
            with open(os.path.join(here["work"], file_name), "rb") as f:
                taken[file_name[:-4]] = f.read()
        note(event="take", ids=sorted(taken), secrets=[part.hex() for data in taken.values() for part in (data[-spool.NONCE_BYTES :], spool.tag_key(data))])
        return taken

    def release(taken):
        """Gives a batch's responses their names in out/, .y, .tag, then .proof or .failed, all with one modification time."""
        stamp = (time.time_ns() // 10**6 + SCENE.get("stamp_off_ms", 0)) * 10**6
        for name, data in sorted(taken.items()):
            request, key = data[: -spool.NONCE_BYTES], spool.tag_key(data)
            if hashlib.sha256(request).hexdigest() in SCENE.get("fail", ()):
                files, tag = {"failed": b""}, spool.tag_of_failed(key, name)
            else:
                files = {"y": struct.pack("<I", 7), "proof": proof_of(request)}
                tag = spool.tag_of(key, name, files["y"], files["proof"])
            if hashlib.sha256(request).hexdigest() in SCENE.get("wrong_tag", ()):
                tag = bytes(spool.TAG_BYTES)
            for ending, content in files.items():
                with open(os.path.join(here["out/.tmp"], name + "." + ending), "xb") as f:
                    f.write(content)
                os.utime(os.path.join(here["out/.tmp"], name + "." + ending), ns=(stamp, stamp))
            for ending in ("y", "tag", "failed", "proof"):
                if ending == "tag":
                    spool.lay_tag(top, name, tag, stamp)
                elif ending in files:
                    os.rename(os.path.join(here["out/.tmp"], name + "." + ending), os.path.join(here["out"], name + "." + ending))
        note(event="release", ids=sorted(taken), at_ns=time.time_ns())
        for name in taken:
            os.remove(os.path.join(here["work"], name + ".req"))

    batch, most, done = int(o["-batch"]), int(o["-most-batches"]), 0
    stopped = os.path.lexists(here["stop"])
    taken = {} if stopped else take(batch - SCENE.get("first_batch_lacks", 0))
    if "in_is_away_s" in SCENE:
        os.rename(here["in"], here["in"] + ".away")
        time.sleep(SCENE["in_is_away_s"])
        os.rename(here["in"] + ".away", here["in"])
        note(event="in/ is back")
    while taken:
        time.sleep(SCENE.get("work_s", [0])[min(done, len(SCENE.get("work_s", [0])) - 1)])
        done += 1
        stopped = os.path.lexists(here["stop"])
        following = take(batch) if done < most and not stopped else {}
        release(taken)
        say("batch %d leaf 0 n %d" % (done - 1, len(taken)))
        taken = following
    if stopped:
        os.remove(here["stop"])
        note(event="stopped by the stop file")
    time.sleep(SCENE.get("end_s", 0))
    say(SCENE.get("last_line", "p258 end 0"))
    note(event="end")
    return SCENE.get("exit_code", 0)


def cp_verify():
    if WORDS[:2] == ["-plan", "id"]:
        if len(WORDS) != 4 or not os.path.isfile(os.path.join(WORDS[2], "policy.txt")) or os.path.basename(WORDS[3]) != "key.bin" or not os.path.isfile(WORDS[3]):
            say("usage: cp_verify -plan id CONFIG_DIR KEY_FILE")
            return 2
        for one in SCENE.get("plan_ids", [SCENE["id"]]):
            say("commitment ID: " + one)
        return SCENE.get("plan_status", 4)
    o = options_of(list(WORDS[5:]), ("-id", "-j"), ("-next-token", "-q"))
    if len(WORDS) < 5 or not isinstance(o, dict):
        say("usage: cp_verify CONFIG_DIR PUBLIC_KEY_DIR REQUEST_FILE TOKEN_FILE PROOF_FILE -id COMMITMENT_ID")
        return 2
    config, public, *files = WORDS[:5]
    request, y, proof = (open(path, "rb").read() for path in files)
    note(event="check", options=o, request=hashlib.sha256(request).hexdigest(), working_directory=os.getcwd(), folders=[os.environ.get(name) for name in ("HOME", "TMPDIR")])
    time.sleep(SCENE.get("check_s", {}).get(hashlib.sha256(request).hexdigest(), 0))
    say("a stand-in: nothing is checked")
    if o.get("-id") != SCENE["id"]:
        say("NO: model-statement")
        return 1
    asks_for_y_alone = struct.unpack("<2I", request[-8:]) == (SCENE["tokens"] - 1, SCENE["tokens"])
    if not (os.path.isfile(os.path.join(config, "policy.txt")) and os.path.isfile(os.path.join(public, "key.bin")) and asks_for_y_alone):
        say("NOT-A-VERIFICATION")
        return 14
    if hashlib.sha256(request).hexdigest() in SCENE.get("refuse", ()) or proof != proof_of(request) or y != struct.pack("<I", 7):
        say("NO: stand-in")
        return 1
    if hashlib.sha256(request).hexdigest() in SCENE.get("within_limits", ()):
        say("WITHIN LIMITS: clipped-gap C=512 theta=300 cap=1 s=4 sh=8")
        return 0
    say("YES")
    return 0


def nvidia_smi():
    o = options_of([part for word in WORDS for part in word.split("=", 1)], ("-i", "--query-gpu", "--format"), ())
    if not isinstance(o, dict) or o.get("--format") != "csv,noheader,nounits" or "-i" not in o:
        return 2
    with open(os.path.join(HOME, "record"), encoding="utf-8") as f:
        earlier = sum('"event": "sample"' in line for line in f)
    fields = o["--query-gpu"].split(",")
    note(event="sample" if fields == ["memory.used"] else "describe", gpu=o["-i"], process_id=os.getpid())
    time.sleep(SCENE.get("late_sample_s", 0) * bool(earlier) if fields == ["memory.used"] else SCENE.get("describe_s", 0))
    used = SCENE.get("gpu_mib", [0])
    known = {"name": "STAND-IN GPU", "driver_version": "0.0", "memory.total": "81559", "memory.used": str(used[min(earlier, len(used) - 1)]),
             "pci.bus_id": "00000000:00:00.0", "uuid": "GPU-stand-in"}  # fmt: skip
    say(", ".join(known[field] for field in fields))
    return 0


if __name__ == "__main__":
    sys.exit({"cp_prove": cp_prove, "cp_verify": cp_verify, "nvidia-smi": nvidia_smi}[ROLE]())
