#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.
# Started by the runner; by hand: below a directory of your own. (Runner: run_suite.py. The work folder goes there.)
"""Checks that cp_verify prints what its YES is about, ends with NOT-A-VERIFICATION if -id is
missing, -next-token does not fit the request or -no-proof-of-work is given where a round needs
its work, and ends with NO if -policy names another acceptance policy than the parameter file
(policy.txt) sets.

usage: what_a_yes_covers_test.py <cp_verify> <world> <new work folder>
The second argument is a test fixture: honest files on synthetic weights, as
verifier/tests/other_parameters_worlds.sh writes to worlds/true: config/, ivk/, id.txt (the
commitment ID), in/example.req with out/example.y and out/example.proof.
A request file holds T token ids, numbered from 0, then r and e: tokens r + 1 to e are checked
("covered" in the output); token T is y, the next token. In example.req only y is checked
(r = T - 1, e = T).
Optional: two requests in which token T - 1 is checked, under the exact match (every checked token
is the integer model's top choice): in-two/two.req (r = T - 2, e = T) with out-two/two.y,
two.proof, and in-one/one.req (the same tokens, r = T - 2, e = T - 1) with out-one/one.y, one.proof.
No GPU: no proof is made. The test creates the work folder, with the directories above it if they
are missing, and does not remove it.

Except where marked, the expected results follow the documentation, not cp_verify's source:
verifier/README.md ("The call") and verifier/docs/output.md. The line "covered:" is
shown in docs/verifying.md ("When to accept"), and the policy names that -policy accepts are in
verifier/docs/parameter-file.md ("The standard parameter form").
  - The honest call ends with YES (0).
  - A verification call ("whole call" in verifier/docs/output.md) that ends with YES or
    NOT-A-VERIFICATION (14) prints exactly one line "covered:", equal to what covered_line() of
    verifier/tests/call_accounting_test.py forms from the caller's request file and policy.txt,
    whatever the options say. It ends with four lines. The first is "commitment: ..", the only
    line that begins so; it agrees with commitment-assumed-rounds= in the third. The second is
    "proof of work: none counted" if policy.txt declares no work; otherwise that line or
    "proof of work: up to <n> bits counted in ..", where n is the largest number declared. The
    third is the line for programs, "result-v1 ..", the only line with that word. It matches what
    result_pattern() of the same file forms from the caller's request file, policy.txt, proof
    file and -id (without -id: id=not-shown-without-id); its fields on work repeat the second
    line; least-round=<b> repeats the accounting's largest round error, 2^-<b>; build= is that of
    the line "result:" of the stage "request", where the output has that line (a stage is one of
    three child processes of cp_verify). The fourth is the verdict.
  - A call that ends with NOT-A-VERIFICATION prints no stage's output, so that no stage's
    "STAGE PASSED" can be taken for the answer; with -show-stages it prints all three.
  - Where this list says only "NO": no line "covered:", nowhere the word result-v1.
  - -key and -leaf (part-calls) and -plan accounting (a planning call) on the honest files print
    neither the word result-v1 nor a line "commitment: ..". Nor do the other calls by plain().
  - Without -id: NOT-A-VERIFICATION (unmet condition: id-named). With another ID: NO. Neither
    call's standard output has the files' ID: a caller who pasted it after -id would accept these
    files.
  - Not in those pages: -no-proof-of-work. policy.txt, not cp_verify's output, says which result is
    expected, so a fixture that declares work must need it in the key, the entry and the proof. If
    policy.txt declares none: the line on work is "proof of work: none counted", and the option
    changes nothing. If it declares some: that line tells of counted work, and with the option a
    verification call and both part-calls end with NOT-A-VERIFICATION (unmet condition:
    no-proof-of-work). A proof with one bit turned still ends with NO. -plan accounting with it: 2.
  - -policy with exact-match and policy.txt's numbers C, theta, cap, s, sh: as without the option.
    With clipped-gap, or with one number greater by one: two lines only, "policy: <why>" and
    "NO: policy" (1): the call is rejected before any stage runs.
    With log-probability, which the option does not accept: status 2, nothing on standard output.
  - policy.txt with sT other than sh - 4: the same two lines, even if only y is checked.
  - The same two lines under copies of config/ with other C, theta, cap: if the copy sets another
    policy than the exact match, a token of the request is checked and the call has no -policy;
    and if -policy names a policy that excludes no token (theta = C, and cap is not below the
    number of the request's checked tokens). With the first copy's policy named correctly, or with
    only y checked, the call passes that check and ends with another NO: the copy is not the
    commitment's configuration.
  - An input file that is missing or not a regular file: one line "input-unreadable-<role>: <the
    file's name or role> cannot be read: <why>", without the work folder's path, status 2: the
    caller's mistake is no verdict on a proof.
  - Not in those pages, which only list a wrong length among the causes of the two names below: a
    request file that is not 4 (T + 2) bytes long, or a y file that is not 4: the one line
    "NO: statement-range", or "NO: statement-required" if the file is empty or its length is no
    multiple of 4; status 1. The lengths are tested before a byte of either file is read: a shorter
    request gets the same line under another ID, with another policy named, and with no proof file.
  - -next-token: as without it if r = T - 1 and e = T; otherwise NOT-A-VERIFICATION (unmet
    condition: next-token).
  - A proof fits one request. Another r or e, a changed token, checked or not, another y where
    e = T, the other request's proof: NO.
Not tested: a call that passes under any policy but the exact match, which the fixture sets, so the
last line WITHIN LIMITS has no case here; the conditions exact-match and policy-named of the stage
"request", and the parent process's guards internal-policy-not-given and internal-policy-kind (the
parent rejects first, so no call here fails them); more than one checked token of a request; what
a YES says about the model; the line for programs with claim=policy or with bits= other than 128;
-no-proof-of-work under a parameter file that declares work that no round needs, where it is to
change nothing; what cp_verify writes to standard error; a part-call's two figures on proof of work.
Last line: "what_a_yes_covers_test.py: <n> calls, <m> not as expected", status 0 or 1; if an
optional file is missing and m = 0: "in part: ..", status 77.
Status 3: cp_verify ended with status 0 or 14 in a call where neither was expected. 2: wrong
arguments, or the work folder's name is taken, even by a dangling link. 1 also, without that last
line: the fixture's files are not what this test needs, or the first honest call does not end with
YES."""
import os
import re
import shutil
import struct
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from call_accounting_test import commitment_agrees, covered_line, figures_agree, result_pattern, values, work_agrees  # noqa: E402

if len(sys.argv) != 4 or os.path.lexists(os.path.abspath(sys.argv[3])):
    print(__doc__.split("\n\n")[1].splitlines()[0], file=sys.stderr)
    sys.exit(2)
verifier, world, work = (os.path.abspath(x) for x in sys.argv[1:])
os.makedirs(os.path.dirname(work), exist_ok=True)
os.mkdir(work, 0o700)  # fails if the name is taken; closed to other users
with open(world + "/id.txt") as f:
    ID = f.read().strip()
with open(world + "/config/policy.txt", encoding="ascii") as f:
    POLICY = f.read()
V = values(POLICY)
DECLARED = max([int(x) for l in POLICY.splitlines() if l.startswith("work ") for x in l.split()[2:]], default=0)  # the most bits of work that the file declares
WORK = re.compile("proof of work: none counted" if not DECLARED else r"proof of work: (none counted|up to %d bits counted in [1-9]\d* rounds?; with it the least round has \d+\.\d\d bits, without it \d+\.\d\d bits)" % DECLARED)
calls = bad = passed_wrongly = 0
YES = None  # (status, last line) of the first honest call
ON_WORK = None  # that call's line "proof of work: .."


def words_of(path):
    with open(path, "rb") as f:
        b = f.read()
    return list(struct.unpack("<%dI" % (len(b) // 4), b))


def laid(name, words):
    with open("%s/%s" % (work, name), "wb") as f:
        f.write(struct.pack("<%dI" % len(words), *words))
    return "%s/%s" % (work, name)


def stages(lines):
    """Counts the stage headings in `lines`, one above each stage's output."""
    return sum(l.startswith("---- stage ") for l in lines)


def call(what, want, request, y, proof, *options, cause=None):
    """want: "yes", "14" (NOT-A-VERIFICATION), "NO" or "policy" (NO: policy, before any stage).
    cause, with "14": the condition, or the list of conditions, that lines "not the standard
    parameter form: .." name."""
    global calls, bad, passed_wrongly, YES, ON_WORK
    calls += 1
    r = subprocess.run([verifier, world + "/config", world + "/ivk", request, y, proof, *options], capture_output=True, text=True, errors="replace", check=False)
    lines = r.stdout.splitlines()
    last = lines[-1] if lines else ""
    said = [l for l in lines if l.startswith("covered:")]
    with open(request, "rb") as f, open(proof, "rb") as g:
        request_bytes, proof_bytes = f.read(), g.read()
    mine = covered_line(request_bytes, POLICY)
    for_programs = [l for l in lines if "result-v1" in l]

    def closes(status):
        """Checks the three lines before the verdict (the module's docstring). status: 0 or 14."""
        fields = re.fullmatch(result_pattern(request_bytes, POLICY, proof_bytes, status, ID if ID in options else None), lines[-2]) if len(lines) > 3 else None
        if fields is None or for_programs != [lines[-2]] or [l for l in lines if l.startswith("commitment: ")] != [lines[-4]]:
            return False
        return commitment_agrees(lines[-4], fields) and work_agrees(lines[-3], fields) and figures_agree(lines, fields, status == 0)

    named = sorted({l.split("not the standard parameter form: ")[1].split(" ")[0] for l in lines if l.startswith(("the call: ", "stage ")) and "not the standard parameter form: " in l})
    if want == "yes":
        if YES is None and (r.returncode, last) == (0, "YES"):
            YES, ON_WORK = (r.returncode, last), lines[-3]
        ok = (r.returncode, last) == YES and said == [mine] and not named and closes(0) and WORK.fullmatch(lines[-3]) is not None and sum("YES" in l for l in lines) == 1 and stages(lines) == 3
    elif want == "14":
        ok = (r.returncode, last) == (14, "NOT-A-VERIFICATION") and said == [mine] and named == ([cause] if isinstance(cause, str) else sorted(cause)) and closes(14) and WORK.fullmatch(lines[-3]) is not None
        ok = ok and stages(lines) == sum(l.startswith("STAGE PASSED") for l in lines) == (3 if "-show-stages" in options else 0)
    elif want == "policy":
        ok = r.returncode == 1 and len(lines) == 2 and lines[0].startswith("policy: ") and last == "NO: policy"
        passed_wrongly += r.returncode in (0, 14)
    else:
        ok = r.returncode == 1 and last.startswith("NO: ") and not said and not for_programs and len(lines) == 3  # three lines: a rejection is to print nothing that was read from the files; the test counts the lines and, below, looks for the ID
        passed_wrongly += r.returncode in (0, 14)
    if ID not in options:
        ok = ok and ID not in r.stdout
    bad += not ok
    print("%s %s: status %d, %s%s%s" % ("as expected    " if ok else "NOT AS EXPECTED", what, r.returncode, last[:60], "; for " + ", ".join(named) if named else "",
                                        "; " + said[0].split(" (")[0] if len(said) == 1 else "; %d lines covered:" % len(said)), flush=True)  # fmt: skip


def plain(what, words, status, fits):
    """Runs cp_verify with the arguments in words. fits: a test on its standard output's lines."""
    global calls, bad, passed_wrongly
    calls += 1
    r = subprocess.run([verifier, *words], capture_output=True, text=True, errors="replace", check=False)
    lines = r.stdout.splitlines()
    ok = r.returncode == status and fits(lines) and "result-v1" not in r.stdout and not any(l.startswith("commitment: ") for l in lines)
    bad += not ok
    passed_wrongly += r.returncode in (0, 14)
    print("%s %s: status %d, %s" % ("as expected    " if ok else "NOT AS EXPECTED", what, r.returncode, lines[-1][:100] if lines else "(no line)"), flush=True)


def unreadable(what, role, *words):
    named = {"policy": "policy.txt", "KL": "KL.bin", "layout": "layout.txt", "key": "key.bin"}.get(role, "the %s file" % role)
    plain(what, words, 2, lambda lines: len(lines) == 1 and lines[0].startswith("input-unreadable-%s: %s cannot be read: " % (role, named)) and work not in lines[0])


def config_with(name, without=None, **lines):
    """A copy of the config folder: without one file, or with other numbers in lines of policy.txt."""
    folder = "%s/%s" % (work, name)
    shutil.copytree(world + "/config", folder)
    if without:
        os.remove("%s/%s" % (folder, without))
    text = POLICY
    for line, number in lines.items():
        text, times = re.subn(r"(?m)^%s \d+$" % line, "%s %d" % (line, number), text)
        if times != 1:
            sys.exit("what_a_yes_covers_test.py: policy.txt does not have exactly one line " + line)
    if lines:
        with open(folder + "/policy.txt", "w", encoding="ascii") as f:
            f.write(text)
    return folder


def own(kind=None, **off):
    """The -policy argument: kind (default exact-match), then policy.txt's numbers plus off[n]."""
    return ",".join([kind or "exact-match"] + [str(int(V[n]) + off.get(n, 0)) for n in ("C", "theta", "cap", "s", "sh")])


named = ("-id", ID)
R, Y, P = world + "/in/example.req", world + "/out/example.y", world + "/out/example.proof"
w = words_of(R)
T = len(w) - 2
if (w[T], w[T + 1]) != (T - 1, T):
    sys.exit("what_a_yes_covers_test.py: in/example.req does not ask for the next token")
call("the next token: the honest call", "yes", R, Y, P, *named)
if YES is None:
    print("what_a_yes_covers_test.py: the honest call does not end with a yes: nothing can be asked")
    sys.exit(1)
call("the same with -q", "yes", R, Y, P, *named, "-q")
call("without -id", "14", R, Y, P, cause="id-named")
call("without -id, with -show-stages", "14", R, Y, P, "-show-stages", cause="id-named")
call("under another ID", "NO", R, Y, P, "-id", ID.translate(str.maketrans("0123456789abcdef", "123456789abcdef0")))
call("-policy with the file's own kind and numbers", "yes", R, Y, P, *named, "-policy", own())
call("-policy names the kind clipped-gap", "policy", R, Y, P, *named, "-policy", own("clipped-gap"))
plain("-policy names the kind log-probability", [world + "/config", world + "/ivk", R, Y, P, *named, "-policy", own("log-probability")], 2, lambda lines: not lines)
for n in ("C", "theta", "cap", "s", "sh"):
    call("-policy names %s greater by one" % n, "policy", R, Y, P, *named, "-policy", own(**{n: 1}))
call("-next-token", "yes", R, Y, P, *named, "-next-token")
call("-next-token and no -id", "14", R, Y, P, "-next-token", cause="id-named")
# -no-proof-of-work. policy.txt says which result is expected; cp_verify's own line must agree.
counted = DECLARED > 0
agrees = (ON_WORK != "proof of work: none counted") == counted
calls += 1
bad += not agrees
print("%s policy.txt declares %s work, and the honest call says: %s%s" % ("as expected    " if agrees else "NOT AS EXPECTED", "some" if counted else "no", ON_WORK.split(";")[0],
                                                                          "" if agrees or not counted else " (this test needs a fixture whose declared work is needed)"), flush=True)  # fmt: skip
for what, more in (("", ()), (", with -show-stages", ("-show-stages",)), (", twice", ("-no-proof-of-work",)), (", with -next-token", ("-next-token",))):
    call("-no-proof-of-work" + what, "14" if counted else "yes", R, Y, P, *named, "-no-proof-of-work", *more, cause="no-proof-of-work")
call("-no-proof-of-work and no -id", "14", R, Y, P, "-no-proof-of-work", cause=["id-named", "no-proof-of-work"] if counted else "id-named")
call("-no-proof-of-work under another ID", "NO", R, Y, P, "-no-proof-of-work", "-id", ID.translate(str.maketrans("0123456789abcdef", "123456789abcdef0")))
call("-no-proof-of-work and another y", "NO", R, laid("other-no-proof-of-work.y", [words_of(Y)[0] ^ 1]), P, *named, "-no-proof-of-work")
turned = words_of(P)
turned[len(turned) // 2] ^= 1
call("-no-proof-of-work and a proof with one bit turned", "NO", R, Y, laid("turned-no-proof-of-work.proof", turned), *named, "-no-proof-of-work")
call("a wider stretch is asked of this proof (r = T - 2)", "NO", laid("wider.req", w[:T] + [T - 2, T]), Y, P, *named)
call("a prompt token turned", "NO", laid("prompt.req", [w[0] ^ 1] + w[1:]), Y, P, *named)
call("another y", "NO", R, laid("other.y", [words_of(Y)[0] ^ 1]), P, *named)
# Parameter files that set another policy than the exact match. The fixture has no proof under one,
# and needs none: "NO: policy" comes before the proof file is opened.
least = 1 << int(V["sh"])
gap, loose = config_with("config-gap", C=least + 44, theta=least + 43), config_with("config-loose", cap=1)
token = laid("token.req", w[:T] + [T - 2, T])  # covers one token of the request, and y
of = lambda C, theta, cap: "clipped-gap,%d,%d,%d,%s,%s" % (C, theta, cap, V["s"], V["sh"])  # noqa: E731
for_policy = lambda begins: lambda lines: len(lines) == 2 and lines[0].startswith("policy: " + begins) and lines[1] == "NO: policy"  # noqa: E731
otherwise = lambda lines: bool(lines) and lines[-1].startswith("NO: ") and lines[-1] != "NO: policy" and not any(l.startswith("policy: ") for l in lines)  # noqa: E731
plain("another policy in the file, a token is covered, no -policy", [gap, world + "/ivk", token, Y, P, *named], 1, for_policy("the request covers more than the output token, "))
plain("the same, the file's policy rightly named", [gap, world + "/ivk", token, Y, P, *named, "-policy", of(least + 44, least + 43, 0)], 1, otherwise)
plain("another policy in the file, y alone is covered, no -policy", [gap, world + "/ivk", R, Y, P, *named], 1, otherwise)
plain("a policy that bars no token, rightly named, a token is covered", [loose, world + "/ivk", token, Y, P, *named, "-policy", of(least, least, 1)], 1, for_policy("a result under the named policy would mean nothing: theta = C and cap "))
plain("the same policy, y alone is covered", [loose, world + "/ivk", R, Y, P, *named, "-policy", of(least, least, 1)], 1, otherwise)
for kind, sT in (("log-probability", int(V["s"])), ("another-table", int(V["sh"]) - 3)):
    plain("the file sets the kind " + kind, [config_with("config-" + kind, sT=sT), world + "/ivk", R, Y, P, *named], 1, for_policy("the parameter file sets the kind of policy %s; " % kind))
# A request or a y of another length.
def of_bytes(name, data):
    with open("%s/%s" % (work, name), "wb") as f:
        f.write(data)
    return "%s/%s" % (work, name)


alone = lambda name: lambda lines: lines == ["NO: " + name]  # noqa: E731
whole = lambda request, y, proof, *more: [world + "/config", world + "/ivk", request, y, proof, *more]  # noqa: E731
shorter = laid("shorter.req", w[:-1])
with open(R, "rb") as f:
    request_bytes = f.read()
plain("the request one word shorter", whole(shorter, Y, P, *named), 1, alone("statement-range"))
plain("the request one word longer", whole(laid("longer.req", w + [0]), Y, P, *named), 1, alone("statement-range"))
plain("the request one byte longer", whole(of_bytes("odd.req", request_bytes + bytes(1)), Y, P, *named), 1, alone("statement-required"))
plain("an empty request", whole(of_bytes("empty.req", b""), Y, P, *named), 1, alone("statement-required"))
plain("a y of two words", whole(R, laid("two-words.y", words_of(Y) + [0]), P, *named), 1, alone("statement-range"))
plain("an empty y", whole(R, of_bytes("empty.y", b""), P, *named), 1, alone("statement-required"))
plain("the shorter request under another ID", whole(shorter, Y, P, "-id", ID.translate(str.maketrans("0123456789abcdef", "123456789abcdef0"))), 1, alone("statement-range"))
plain("the shorter request, another policy named", whole(shorter, Y, P, *named, "-policy", own("clipped-gap")), 1, alone("statement-range"))
plain("the shorter request, no proof file", whole(shorter, Y, work + "/none.proof", *named), 1, alone("statement-range"))
# Files that cannot be read.
os.makedirs(work + "/no-key/leaf")
os.makedirs(work + "/a-folder")
folder_for_layout = config_with("config-layout-is-a-folder", without="layout.txt")
os.makedirs(folder_for_layout + "/layout.txt")
unreadable("no policy.txt", "policy", config_with("config-no-policy", without="policy.txt"), world + "/ivk", R, Y, P, *named)
unreadable("no KL.bin", "KL", config_with("config-no-KL", without="KL.bin"), world + "/ivk", R, Y, P, *named)
unreadable("layout.txt is a folder", "layout", folder_for_layout, world + "/ivk", R, Y, P, *named)
unreadable("no key.bin", "key", world + "/config", work + "/no-key", R, Y, P, *named)
unreadable("no request file", "request", world + "/config", world + "/ivk", work + "/none.req", Y, P, *named)
unreadable("no y file", "y", world + "/config", world + "/ivk", R, work + "/none.y", P, *named)
unreadable("no proof file", "proof", world + "/config", world + "/ivk", R, Y, work + "/none.proof", *named)
unreadable("the proof is a folder", "proof", world + "/config", world + "/ivk", R, Y, work + "/a-folder", *named)
unreadable("-key: no key.bin", "key", "-key", world + "/config", work + "/no-key/key.bin")
unreadable("-leaf: no entry file", "entry", "-leaf", world + "/config", world + "/ivk/key.bin", work + "/no-key/leaf/0.bin")
unreadable("-plan id: no key.bin", "key", "-plan", "id", world + "/config", work + "/no-key/key.bin")
# Part-calls and a planning call that find no fault.
# A proof's third 32-bit word is k: its leaf file ("entry" in the output) is ivk/leaf/<k>.bin.
with open(P, "rb") as f:
    leaf = struct.unpack("<3I", f.read(12))[2]
part = lambda name: lambda lines: lines[-1] == "PART PASSED: " + name and WORK.fullmatch(lines[-2]) is not None  # noqa: E731
plain("-key on the honest key", ["-key", world + "/config", world + "/ivk/key.bin", *named], 3, part("key"))
plain("-leaf on the honest entry", ["-leaf", world + "/config", world + "/ivk/key.bin", "%s/ivk/leaf/%d.bin" % (world, leaf), *named], 3, part("leaf"))
plain("-plan accounting", ["-plan", "accounting", world + "/config", "128"], 4, lambda lines: lines[-1].startswith("PLAN ONLY"))
plain("-plan accounting with -no-proof-of-work", ["-plan", "accounting", world + "/config", "128", "-no-proof-of-work"], 2, lambda lines: not lines)


def refusing_work(what, words):
    """A part-call with -no-proof-of-work, compared with the same call without it. If policy.txt
    declares no work: that call's line says "none counted", and the option changes neither status
    nor the last two lines. Else that line tells of counted work, and the call with the option ends
    with NOT-A-VERIFICATION (14) for no-proof-of-work alone and prints no stage's output."""
    global calls, bad, passed_wrongly
    calls += 1
    before = subprocess.run([verifier, *words], capture_output=True, text=True, errors="replace", check=False)
    r = subprocess.run([verifier, *words, "-no-proof-of-work"], capture_output=True, text=True, errors="replace", check=False)
    old, lines = before.stdout.splitlines(), r.stdout.splitlines()
    if before.returncode != 3 or len(old) < 2 or WORK.fullmatch(old[-2]) is None or (old[-2] != "proof of work: none counted") != counted:
        ok = False
    elif not counted:
        ok = r.returncode == 3 and lines[-2:] == old[-2:]
    else:
        causes = [l for l in lines if "not the standard parameter form: " in l]
        ok = r.returncode == 14 and lines[-2:] == [old[-2], "NOT-A-VERIFICATION"] and len(causes) == 1 and causes[0].startswith("the call: not the standard parameter form: no-proof-of-work (")
        ok = ok and stages(lines) == 0 and not any(l.startswith("STAGE PASSED") for l in lines)
    ok = ok and "result-v1" not in r.stdout and not any(l.startswith("commitment: ") for l in lines)
    bad += not ok
    passed_wrongly += before.returncode in (0, 14) or r.returncode == 0 or (r.returncode == 14 and not counted)
    print("%s %s: status %d, %s (without the option: %s)" % ("as expected    " if ok else "NOT AS EXPECTED", what, r.returncode, lines[-1][:60] if lines else "(no line)", old[-2].split(";")[0] if len(old) > 1 else "(no line)"), flush=True)


refusing_work("-key with -no-proof-of-work", ["-key", world + "/config", world + "/ivk/key.bin", *named])
refusing_work("-leaf with -no-proof-of-work", ["-leaf", world + "/config", world + "/ivk/key.bin", "%s/ivk/leaf/%d.bin" % (world, leaf), *named])
there = all(os.path.isfile(world + p) for p in ("/in-two/two.req", "/out-two/two.y", "/out-two/two.proof", "/in-one/one.req", "/out-one/one.y", "/out-one/one.proof"))
if there:
    R2, Y2, P2 = world + "/in-two/two.req", world + "/out-two/two.y", world + "/out-two/two.proof"
    R1, Y1, P1 = world + "/in-one/one.req", world + "/out-one/one.y", world + "/out-one/one.proof"
    w2, w1 = words_of(R2), words_of(R1)
    if w2[T:] != [T - 2, T] or w1[T:] != [T - 2, T - 1] or w1[:T] != w2[:T]:
        sys.exit("what_a_yes_covers_test.py: in-two/two.req and in-one/one.req are not (T - 2, T) and (T - 2, T - 1) on the same tokens")
    call("a token and y are covered: the honest call", "yes", R2, Y2, P2, *named)
    call("the same with -next-token", "14", R2, Y2, P2, *named, "-next-token", cause="next-token")
    call("the same with -policy, the file's own", "yes", R2, Y2, P2, *named, "-policy", own())
    call("the covered token turned", "NO", laid("two-token.req", w2[: T - 1] + [w2[T - 1] ^ 1] + w2[T:]), Y2, P2, *named)
    call("a narrower stretch is asked of this proof (r = T - 1)", "NO", laid("two-narrower.req", w2[:T] + [T - 1, T]), Y2, P2, *named)
    call("y left out of what is asked (e = T - 1)", "NO", laid("two-no-y.req", w2[:T] + [T - 2, T - 1]), Y2, P2, *named)
    call("another y", "NO", R2, laid("two-other.y", [words_of(Y2)[0] ^ 1]), P2, *named)
    call("a token alone is covered: the honest call", "yes", R1, Y1, P1, *named)
    call("the same with -next-token", "14", R1, Y1, P1, *named, "-next-token", cause="next-token")
    call("the covered token turned", "NO", laid("one-token.req", w1[: T - 1] + [w1[T - 1] ^ 1] + w1[T:]), Y1, P1, *named)
    call("y is asked of this proof too (e = T)", "NO", laid("one-with-y.req", w1[:T] + [T - 2, T]), Y1, P1, *named)
    call("the other request's proof", "NO", R1, Y1, P2, *named)
tail = "%d calls, %d not as expected" % (calls, bad)
print(("what_a_yes_covers_test.py: " + tail) if there or bad else ("in part: %s; not played: a request that covers a token (the world has no in-two/ and in-one/)" % tail))
sys.exit(3 if passed_wrongly else 1 if bad else 0 if there else 77)
