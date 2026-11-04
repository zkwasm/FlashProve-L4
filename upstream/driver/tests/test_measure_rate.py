#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.

"""Tests of driver/measure_rate.py: its reckoning by itself, on times written here by hand, and
whole runs in which driver/tests/stand_in_programs.py takes the places of cp_prove, cp_verify and
nvidia-smi. The stand-ins prove, check and measure nothing, and no GPU is used.

A pass shows that the script reckons rightly from given times, gives no rate where its head says so,
writes only below its work directory, removes no file, signals nothing, and waits for cp_prove and
says so if it fails itself. It does not show that the
script works with the real programs, or that it reaches any rate that this repository reports: only
a run on a GPU shows that.

A test cannot set a file's change time. So the rule of the steady rate is held twice: on given
times, by steady_rate() alone, and in whole runs, where the test reads the change times of the files
that the run left in out/ and reckons the rate from them in its own way.

Each test works in a new temporary directory and removes it. Some tests send SIGINT, SIGTERM or
SIGHUP to the process group of the script that they started, as a terminal does. A call that must be
refused is killed, by its handle, if it has not ended after 60 seconds. Linux only.

usage: python3 driver/tests/test_measure_rate.py

Exit status 0: all tests passed. 1: not all. 77: the tests of the reckoning passed, but the whole
runs were skipped, because another account may be able to change a directory above the temporary
directories, where the script refuses to work; the last line then begins "stood aside"."""

import hashlib
import json
import os
import re
import resource
import shlex
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import time
import unittest

sys.dont_write_bytecode = True
TOP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def beside_the_script():
    """Returns every path in the script's directory and below it."""
    return sorted(os.path.join(folder, name) for folder, folders, files in os.walk(TOP) for name in folders + files)


BESIDE_THE_SCRIPT = beside_the_script()  # taken before the script is imported, which must not write there either
sys.path.insert(0, TOP)
import measure_rate  # noqa: E402
import spool  # noqa: E402

SCRIPT = os.path.join(TOP, "measure_rate.py")
STAND_INS = os.path.join(TOP, "tests", "stand_in_programs.py")
ID = "ab" * 32
TOKENS, VOCABULARY = 4, 1000  # as the parameter file that setUp() writes has them
S = 10**9
OPEN_GROUND = []  # gets one entry, "<place> (<why>)", if the whole runs are skipped


class Reckoning(unittest.TestCase):
    def test_five_batches_by_hand(self):
        # Intervals 10, 11, 12 and 8 s: the mean is 10.25 s. 2 x 4 tokens / 10.25 s = 0.7804.., cut to 0.78.
        # 10.25 s / 2 requests = 5.125 s.
        got = measure_rate.steady_rate([100 * S, 110 * S, 121 * S, 133 * S, 141 * S], 2, 4)
        self.assertEqual(got["intervals_ns"], [10 * S, 11 * S, 12 * S, 8 * S])
        self.assertEqual((got["counted"], got["sum_ns"]), (4, 41 * S))
        self.assertEqual((got["tokens_a_second"], got["seconds_a_request"]), ("0.78", "5.125"))

    def test_five_batches_of_the_full_size_by_hand(self):
        # Intervals 223.9, 224.0, 224.2 and 224.03 s: their sum is 896.13 s. 24 x 1,024 x 4 = 98,304 tokens;
        # 98,304 / 896.13 = 109.698.., cut to 109.69 (rounded 109.70). 896.13 / 4 / 24 = 9.3346.., cut to 9.334 (rounded 9.335).
        releases = [0, 223900 * 10**6, 447900 * 10**6, 672100 * 10**6, 896130 * 10**6]
        got = measure_rate.steady_rate([1790000000 * S + r for r in releases], 24, 1024)
        self.assertEqual((got["counted"], got["tokens_a_second"], got["seconds_a_request"]), (4, "109.69", "9.334"))

    def test_six_batches_by_hand(self):
        # Four intervals of 10 s and one of 1 s: 2 x 4 x 5 tokens / 41 s = 0.9756.., cut to 0.97 (rounded it would be 0.98).
        got = measure_rate.steady_rate([10 * S, 20 * S, 30 * S, 40 * S, 50 * S, 51 * S], 2, 4)
        self.assertEqual((got["counted"], got["tokens_a_second"], got["seconds_a_request"]), (5, "0.97", "4.100"))

    def test_three_releases_are_enough(self):
        self.assertEqual(measure_rate.steady_rate([10 * S, 20 * S, 30 * S], 2, 4)["tokens_a_second"], "0.80")

    def test_too_few_releases_or_releases_out_of_order_give_no_rate(self):
        for releases in ([1, 2], [1, 2, 2, 3], [1, 3, 2, 4]):
            with self.assertRaises(ValueError):
                measure_rate.steady_rate(releases, 2, 4)

    def test_a_quotient_is_cut_and_never_rounded_up(self):
        self.assertEqual([measure_rate.cut(*x) for x in ((2, 3, 2), (1999, 1000, 2), (5, 1, 3), (-19, 10, 1), (-1, 100, 1))], ["0.66", "1.99", "5.000", "-1.9", "-0.0"])

    def test_clocks_count_as_apart_only_beyond_two_seconds(self):
        self.assertEqual(measure_rate.clocks_apart({"a": 0, "b": 2 * S, "c": 2 * S + 1}), [("a", "c")])

    def test_a_window_follows_the_written_rule(self):
        # Token 9 of window 2 under seed 1 is word 1 of the hash with block number 1.
        digest = hashlib.sha256(b"measure_rate window" + (1).to_bytes(8, "little") + (2).to_bytes(4, "little") + (1).to_bytes(4, "little")).digest()
        made = measure_rate.window(1, 2, 16, 202048)
        self.assertEqual((len(made), made[9]), (16, int.from_bytes(digest[4:8], "little") % 202048))
        self.assertEqual(made, measure_rate.window(1, 2, 16, 202048))
        self.assertNotEqual(made, measure_rate.window(2, 2, 16, 202048))
        self.assertNotEqual(made, measure_rate.window(1, 3, 16, 202048))

    def test_a_window_of_text_follows_the_written_rule(self):
        # Two rows of four ids. 100 places modulo the three ids that turn is 1 place; 200 is 2; 300 is none.
        rows = [(5, 1, 2, 3), (6, 7, 8, 9)]
        self.assertEqual([measure_rate.text_window(rows, w) for w in range(8)],
                         [[5, 1, 2, 3], [6, 7, 8, 9], [5, 2, 3, 1], [6, 8, 9, 7], [5, 3, 1, 2], [6, 9, 7, 8], [5, 1, 2, 3], [6, 7, 8, 9]])  # fmt: skip
        # At the full size the second turn begins, behind the first id, with the id that stood at place 101.
        self.assertEqual(measure_rate.text_window([tuple(range(1024))], 1), [0] + list(range(101, 1024)) + list(range(1, 101)))

    def answers(self, *batches):
        """{id: Answer} and {id: when it left in/} for batches given as (start, stamp, [change times])."""
        answers, left = {}, {}
        for start, stamp, changed in batches:
            for at in changed:
                name = "%064x" % len(answers)
                answers[name], left[name] = measure_rate.Answer(False, at, stamp, at + 1, 100), start
        return answers, left

    def test_responses_are_grouped_by_their_modification_time_and_a_release_is_the_latest_change_time(self):
        batches = measure_rate.batches_of(*self.answers((50, 7, [90, 95]), (10, 3, [41, 40])))
        self.assertEqual([(b["started_ns"], b["release_ns"], b["modified_ns"], b["seen_ns"], len(b["ids"])) for b in batches], [(10, 41, 3, 42, 2), (50, 95, 7, 96, 2)])
        self.assertEqual(measure_rate.batch_faults(batches, 2, 2), [])

    def test_batches_that_are_not_full_too_few_or_not_apart_are_named(self):
        batches = measure_rate.batches_of(*self.answers((10, 3, [40, 60]), (20, 7, [50])))
        self.assertEqual(
            measure_rate.batch_faults(batches, 2, 3),
            ["the number of batches seen is 2, not 3", "batch 1 did not have exactly 2 requests: it had 1", "batches 1 and 2 cannot be told apart: their starts or their releases overlap"],
        )

    def test_lists_of_cpus(self):
        self.assertEqual(measure_rate.ranges({0, 1, 2, 5, 7, 8}), "0-2,5,7-8")
        self.assertEqual(measure_rate.numbers_in("0-2,5,7-8"), {0, 1, 2, 5, 7, 8})


class Runs(unittest.TestCase):
    def setUp(self):
        found = spool.others_can_change(tempfile.gettempdir())
        if found:
            OPEN_GROUND[:] = ["%s (%s)" % found]
            self.skipTest("the temporary directories lie below an open place")
        self.top = os.path.realpath(tempfile.mkdtemp(prefix="measure-rate-"))
        self.addCleanup(shutil.rmtree, self.top, ignore_errors=True)
        for folder in ("programs", "config", "ivk", "isk"):
            os.mkdir(self.path(folder), 0o700)
        for role in ("cp_prove", "cp_verify", "nvidia-smi"):
            line = 'exec %s -B %s %s %s "$@"' % (shlex.quote(sys.executable), shlex.quote(STAND_INS), role, shlex.quote(self.path("scene.json")))
            self.put(("programs", role), "#!/bin/sh\n%s\n" % line, 0o700)
        self.put(("config", "policy.txt"), "lt 2\nV %d\nslots 8\n" % VOCABULARY)
        for name in (("config", "KL.bin"), ("config", "layout.txt"), ("ivk", "key.bin"), ("record",)):
            self.put(name, "")
        self.environment = dict(os.environ, PATH=self.path("programs") + os.pathsep + os.environ.get("PATH", ""))
        self.environment.pop("CUDA_VISIBLE_DEVICES", None)

    def path(self, *parts):
        return os.path.join(self.top, *parts)

    def put(self, parts, text, mode=0o600, hole=0):
        """Creates a new file below the test's directory; fails if the name is taken. hole: see numpy_file()."""
        descriptor = os.open(self.path(*parts), os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
        os.write(descriptor, text if isinstance(text, bytes) else text.encode())
        if hole:
            os.pwrite(descriptor, b"\0", os.fstat(descriptor).st_size + hole - 1)
        os.close(descriptor)

    def command(self, *more, work="work", script=(SCRIPT,)):
        return [sys.executable, *script, "--programs", self.path("programs"), "--isk", self.path("isk"), "--ivk", self.path("ivk"),
                "--config", self.path("config"), "--id", ID, "--work", self.path(work), "--batch", "2", *more]  # fmt: skip

    def set_the_scene(self, **scene):
        self.put(("scene.json",), json.dumps(dict({"tokens": TOKENS, "id": ID, "start_up_s": 0.1, "work_s": [0.15]}, **scene)))

    def measure(self, *more, largest_file=None, **scene):
        """One whole run. Returns its exit code, standard output and standard error. largest_file: no file that the script
        itself writes can grow beyond so many bytes, as on a full disk; the stand-ins lift that limit for themselves."""
        self.set_the_scene(**scene)
        limit = largest_file and (lambda: resource.setrlimit(resource.RLIMIT_FSIZE, (largest_file, resource.getrlimit(resource.RLIMIT_FSIZE)[1])))
        done = subprocess.run(self.command(*more), stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, env=self.environment, preexec_fn=limit)
        return done.returncode, done.stdout, done.stderr

    def record(self, event):
        with open(self.path("record"), encoding="utf-8") as f:
            return [line for line in map(json.loads, f) if line["event"] == event]

    def figures(self):
        with open(self.path("work", "figures.json"), encoding="utf-8") as f:
            return json.load(f)

    def sum_of_request(self, number):
        return hashlib.sha256(spool.request(measure_rate.window(1, number, TOKENS, VOCABULARY), TOKENS - 1, TOKENS)).hexdigest()

    def releases(self):
        """Each batch's release by the test's own look: the batches as the stand-in recorded them, the change times as they are now."""
        return [max(os.lstat(self.path("work", "spool", "out", name + ".proof")).st_ctime_ns for name in event["ids"]) for event in self.record("release")]

    def no_rate(self, code, out, *reasons):
        self.assertEqual(code, 1, out)
        self.assertNotIn("tokens a second", out)
        told = out[out.index("NO RATE IS GIVEN") :]
        for reason in reasons:
            self.assertRegex(told, reason)
        self.assertLessEqual(len(self.record("call")), 1)
        self.assertEqual(self.record("signal"), [])

    def test_a_whole_run_of_five_batches(self):
        before = sorted(os.listdir(self.top) + ["scene.json", "work"])
        code, out, err = self.measure(start_up_s=0.3, work_s=[0.4], end_s=0.2, gpu_mib=[100, 300, 200])
        self.assertEqual((code, err), (0, ""), out)
        r = self.releases()
        self.assertEqual(len(r), 5)
        # The sum of the intervals is r[4] - r[0]; 2 requests x 4 tokens x 4 intervals, in hundredths of a token a second.
        hundredths = 2 * 4 * 4 * S * 100 // (r[4] - r[0])
        self.assertIn("STEADY RATE: %d.%02d tokens a second, that is " % divmod(hundredths, 100), out)
        for number, release in enumerate(r, 1):
            self.assertRegex(out, r"(?m)^ +%d +2  \d+\.\d{9}  %d\.%09d  \d+\.\d{9}  \d+\.\d{9}$" % (number, release // S, release % S))
        # Within a tolerance. The stand-in slept 0.4 s before each release, so none of these intervals is
        # shorter, but for the step of the clock from which the kernel takes a change time. On a busy machine they are
        # longer, so from above the releases are compared with the stand-in's own clock, read right after each release.
        for interval in (r[1] - r[0], r[2] - r[1], r[3] - r[2], r[4] - r[3]):
            self.assertGreaterEqual(interval, 0.38 * S)
        self.assertLessEqual(hundredths, 8 * 100 / 0.38)
        for release, event in zip(r, self.record("release")):
            self.assertLess(abs(release - event["at_ns"]), 0.25 * S)
        self.assertNotIn("NOTE", out)
        figures = self.figures()
        for request in figures["requests"]:
            about = os.lstat(self.path("work", "spool", "out", request["id"] + ".proof"))
            self.assertEqual((request["changed_ns"], request["modified_ns"], request["size"]), (about.st_ctime_ns, about.st_mtime_ns, about.st_size))
        for batch, event in zip(figures["batches"], self.record("take")):
            self.assertEqual(batch["ids"], event["ids"])
            self.assertLess(abs(batch["started_ns"] - event["ns"]), 2 * S)
        # The call of cp_prove: once, by the documented options alone, in a session of its own.
        (call,) = self.record("call")
        self.assertEqual(call["options"], {"-isk": self.path("isk"), "-spool": self.path("work", "spool"), "-batch": "2", "-in-step": "2", "-most-batches": "5",
                                           "-memory": "lean,tensor-rooms=8"})  # fmt: skip
        self.assertEqual(call["folders"], {"HOME": self.path("work", "home"), "TMPDIR": self.path("work", "tmp")})
        self.assertTrue(call["leads_a_session"])
        self.assertEqual(self.record("signal"), [])
        # Every proof is checked by a call of its own, on the script's own copy of the request.
        self.assertEqual(sorted(check["request"] for check in self.record("check")), sorted(self.sum_of_request(n) for n in range(10)))
        self.assertEqual({json.dumps(check["options"], sort_keys=True) for check in self.record("check")}, {json.dumps({"-id": ID, "-next-token": True, "-j": "4"}, sort_keys=True)})
        self.assertEqual({(check["working_directory"], *check["folders"]) for check in self.record("check")}, {(self.path("work"), self.path("work", "home"), self.path("work", "tmp"))})
        self.assertIn("cp_prove is process %d.\n" % call["process_id"], out)
        self.assertEqual(figures["process_id"], call["process_id"])
        self.assertIn("requests: 10 served with a proof, 0 failed, 0 without a response", out)
        self.assertIn("cp_verify: 10 accepted, 0 not accepted, 0 not checked", out)
        self.assertRegex(out, r"the GPU's memory in use, sampled once a second: the largest of \d+ samples is 300 MiB\n")
        self.assertIn("GPU: STAND-IN GPU; driver 0.0; 81559 MiB of memory", out)
        self.assertEqual({event["gpu"] for event in self.record("sample") + self.record("describe")}, {"0"})
        for name in ("cp_prove", "cp_verify"):
            self.assertIn("sha256 %s  %s" % (spool.sha256(self.path("programs", name)), name), out)
        # Nothing is written outside the work directory, and no file is removed in it.
        self.assertEqual(sorted(os.listdir(self.top)), before)
        self.assertEqual(os.listdir(self.path("isk")), [])
        self.assertEqual(beside_the_script(), BESIDE_THE_SCRIPT)
        self.assertEqual(sorted(os.listdir(self.path("work"))), ["checks", "cp_prove.txt", "figures.json", "home", "report.txt", "requests", "spool", "tmp"])
        self.assertEqual((len(os.listdir(self.path("work", "requests"))), len(os.listdir(self.path("work", "checks")))), (10, 20))
        self.assertEqual(len([x for x in os.listdir(self.path("work", "spool", "out")) if x != ".tmp"]), 30)
        with open(self.path("work", "report.txt"), encoding="utf-8") as f:
            self.assertEqual(f.read(), out)
        # Everything is for its owner alone, and no nonce and no key of a tag is printed or written down.
        for folder, folders, files in os.walk(self.path("work")):
            self.assertEqual({os.lstat(os.path.join(folder, name)).st_mode & 0o777 for name in folders} | {0o700}, {0o700}, folder)
            self.assertEqual({os.lstat(os.path.join(folder, name)).st_mode & 0o777 for name in files} | {0o600}, {0o600}, folder)
        secrets = [secret for event in self.record("take") for secret in event["secrets"]]
        self.assertEqual(len(set(secrets)), 20)
        with open(self.path("work", "figures.json"), encoding="utf-8") as f:
            written = out + f.read()
        self.assertEqual([secret for secret in secrets if secret in written], [])

    def test_six_batches_in_an_empty_directory_of_ones_own(self):
        os.mkdir(self.path("work"), 0o700)
        code, out, _ = self.measure("--batches", "6", "--in-step", "1", "--memory", "", "--least-memory", "0", "--huge-pages", "--public-weights", self.path("seed"), work_s=[0.2, 0.2, 0.2, 0.2, 0.2, 0.05])
        self.assertEqual(code, 0, out)
        r = self.releases()
        self.assertEqual(len(r), 6)
        self.assertIn("STEADY RATE: %d.%02d tokens a second, that is " % divmod(2 * 4 * 5 * S * 100 // (r[5] - r[0]), 100), out)
        self.assertIn("-public-weights given: nothing is hidden", out)
        options = self.record("call")[0]["options"]
        self.assertEqual((options["-most-batches"], options["-in-step"], options["-huge-pages"], options["-public-weights"], "-memory" in options), ("6", "1", True, self.path("seed"), False))

    def test_a_commitment_id_that_the_files_do_not_clearly_give_is_refused(self):
        for scene in ({"plan_status": 0}, {"plan_status": 1}, {"plan_ids": []}, {"plan_ids": [ID, ID]}, {"plan_ids": ["cd" * 32]}):
            self.put(("scene.json",), json.dumps({"tokens": TOKENS, "id": ID, **scene}))
            self.refused(words="cp_verify gives these files another commitment ID than --id, or none")
            os.remove(self.path("scene.json"))

    def test_three_batches_give_a_rate(self):
        code, out, _ = self.measure("--batches", "3", work_s=[0.1])
        self.assertEqual(code, 0, out)
        self.assertIn("STEADY RATE: ", out)

    def test_a_batch_that_is_not_full(self):
        code, out, _ = self.measure(first_batch_lacks=1)
        self.no_rate(code, out, r"- batch 1 did not have exactly 2 requests: it had 1\n", r"- 1 of the requests got no response: numbered \d\n")

    def test_a_failed_request(self):
        code, out, _ = self.measure(fail=[self.sum_of_request(3)])
        self.no_rate(code, out, r"- 1 of the requests failed \(an empty file <id>\.failed in out/\): numbered 3\n")
        self.assertIn("requests: 9 served with a proof, 1 failed, 0 without a response", out)
        self.assertIn("cp_verify: 9 accepted, 0 not accepted, 0 not checked", out)

    def test_a_proof_that_cp_verify_refuses(self):
        code, out, _ = self.measure(refuse=[self.sum_of_request(4)])
        self.no_rate(code, out, r"- cp_verify did not accept the proof of request 4: exit code 1, verdict NO: stand-in; its output is in checks/0004\.txt\. Keep that response as secret")
        self.assertIn("cp_verify: 9 accepted, 1 not accepted, 0 not checked", out)

    def test_exit_code_0_without_the_last_line_yes_is_no_acceptance(self):
        code, out, _ = self.measure(within_limits=[self.sum_of_request(0)])
        self.no_rate(code, out, r"- cp_verify did not accept the proof of request 0: exit code 0, verdict WITHIN LIMITS; its output is in checks/0000\.txt\. ")
        self.assertNotIn("theta", out + json.dumps(self.figures()))

    def test_a_response_with_a_wrong_tag(self):
        code, out, _ = self.measure(wrong_tag=[self.sum_of_request(5)])
        self.no_rate(code, out, r"- the response to request 5 is not as the service promises: the tag is not that of this id and these bytes\n")
        self.assertIn("cp_verify: 9 accepted, 0 not accepted, 0 not checked", out)

    def test_another_exit_code_than_0(self):
        code, out, _ = self.measure(exit_code=9, last_line="p258 end 9 own-test-slot-pads")
        self.no_rate(code, out, r"- cp_prove ended with exit code 9, not with exit code 0\n")
        self.assertIn("  | p258 end 9 own-test-slot-pads\n", out)
        self.assertIn('docs/faults.md, "Always first"', out[out.index("cp_prove ended with exit code 9") :])
        self.assertIn("cp_verify: 10 accepted", out)

    def test_a_secret_key_directory_in_use(self):
        code, out, _ = self.measure(in_use=True)
        self.no_rate(code, out, r"- cp_prove ended with exit code 1, not with exit code 0\n", r"- 10 of the requests got no response: numbered 0-9\n")
        self.assertIn("  | refused (cp-isk): the folder named with -isk is in use: another cp_prove runs on it\n", out)
        self.assertIn("The secret key directory is in use: cp_prove did not start, so nothing was measured.", out)
        self.assertEqual((self.record("check"), len(self.record("call"))), ([], 1))
        before_the_start = out[out.index("BEFORE THE START") : out.index("THE SERVICE")]
        for words in ("on a copy", "on another machine that shares it", "by another route on the same machine", "made for the measurement and serves nobody"):
            self.assertIn(words, re.sub(r"\s+", " ", before_the_start))

    def refused(self, *more, work="work", words):
        """A call that must end with exit code 2 and these words, start nothing and leave the test's directory as it is.
        The script gets at most 2 GiB of memory, so that one that read a large file whole would fail and fill nothing."""
        laid = not os.path.exists(self.path("scene.json"))  # cp_verify -plan id reads it
        if laid:
            self.put(("scene.json",), json.dumps({"tokens": TOKENS, "id": ID}))
        before = {folder: sorted(names + files) for folder, names, files in os.walk(self.top)}
        done = subprocess.run(self.command(*more, work=work), stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, env=self.environment, timeout=60,
                              preexec_fn=lambda: resource.setrlimit(resource.RLIMIT_AS, (1 << 31, resource.getrlimit(resource.RLIMIT_AS)[1])))  # fmt: skip
        self.assertEqual((done.returncode, done.stdout), (2, ""), done.stderr)
        self.assertRegex(done.stderr, words)
        self.assertEqual({folder: sorted(names + files) for folder, names, files in os.walk(self.top)}, before)
        self.assertEqual(self.record("call") + self.record("describe"), [])
        if laid:
            os.remove(self.path("scene.json"))

    def test_a_work_directory_that_is_not_new_or_empty(self):
        os.mkdir(self.path("work"), 0o700)
        self.put(("work", "mine"), "kept")
        self.refused(words=r"^measure_rate\.py: the work directory \S+/work exists and is not an empty directory of yours with mode 0700\.")
        os.mkdir(self.path("open"), 0o700)
        os.chmod(self.path("open"), 0o755)
        self.refused(work="open", words="exists and is not an empty directory of yours with mode 0700")
        self.put(("file",), "")
        self.refused(work="file", words="exists and is not an empty directory of yours with mode 0700")
        self.refused(work=os.path.join("absent", "work"), words=r"the directory above the work directory, \S+/absent, does not exist")
        self.refused(work="a blank", words="must have at most 390 characters, all of them letters, digits or")
        os.mkdir(self.path("no-entry"), 0o700)
        os.chmod(self.path("no-entry"), 0o500)
        self.refused(work="no-entry", words="exists and is not an empty directory of yours with mode 0700")
        os.symlink(self.path("absent"), self.path("link"))
        self.refused(work="link", words="exists and is not an empty directory of yours with mode 0700")
        if os.geteuid():  # root can write everywhere
            self.refused(work=os.path.join("no-entry", "work"), words=r"the work directory \S+/no-entry/work cannot be made \(Permission denied\)\. Nothing was written")

    def test_a_work_directory_in_a_directory_that_the_call_names(self):
        os.mkdir(self.path("mine"), 0o700)
        for folder, option in (("isk", "--isk"), ("ivk", "--ivk"), ("config", "--config"), ("programs", "--programs"), ("mine", "--requests")):
            words = r"^measure_rate\.py: the work directory \S+ is, or lies in, the directory given with %s\. .* Nothing was written$" % option
            for work in (os.path.join(folder, "work"), os.path.join(folder, "deeper", "work"), os.path.join("config", "..", folder, "work")):
                self.refused("--requests", self.path("mine"), work=work, words=words)
        self.refused(work="isk", words="is, or lies in, the directory given with --isk")
        os.symlink(self.path("ivk"), self.path("link"))
        self.refused("--ivk", self.path("link"), work=os.path.join("ivk", "work"), words="is, or lies in, the directory given with --ivk")
        self.refused(work=os.path.join("link", "work"), words="is, or lies in, the directory given with --ivk")
        self.assertEqual(os.listdir(self.path("isk")), [])

    def test_a_work_directory_below_a_place_that_others_can_change(self):
        os.mkdir(self.path("shared"), 0o700)
        os.chmod(self.path("shared"), 0o777)
        self.refused(work=os.path.join("shared", "work"), words=r"lies below \S+/shared, which another account may be able to change")

    def test_calls_that_the_programs_would_refuse(self):
        for more, words in (
            (("--batches", "2"), "--batches takes at least 3"),
            (("--batches", "50001"), "--batches times --batch may be at most 100000 requests"),
            (("--least-memory", "100000000"), r"^measure_rate\.py: only \d+\.\d GB of host memory are available, on the machine or under the container's limit, and --least-memory asks for 100000000 GB\. "),
            (("--checks-at-once", "0"), "--checks-at-once takes 1 to %d, the number of CPUs" % len(os.sched_getaffinity(0))),
            (("--checks-at-once", "100000"), "--checks-at-once takes 1 to %d, the number of CPUs" % len(os.sched_getaffinity(0))),
            (("--in-step", "3"), "--in-step takes 1 to 16, and at most --batch"),
            (("--batch", "9"), "--batch takes 1 to 8"),
            (("--id", "ab"), "--id takes the commitment ID: 64 hex digits"),
            (("--id", "cd" * 32), "cp_verify gives these files another commitment ID than --id, or none"),
            (("--isk", "isk"), "--isk takes an absolute path"),
            (("--programs", self.path("config")), "cp_prove is not a program that you can run"),
            (("--config", self.path("ivk")), "has no file policy.txt"),
            (("--ivk", self.path("config")), "has no file key.bin"),
        ):
            self.refused(*more, words=words)

    def test_parameter_files_that_the_script_cannot_use(self):
        for name, text in (("no-lt", "V 1000\n"), ("no-number", "lt two\nV 1000\n"), ("long", "lt 40\nV 1000\n"), ("no-words", "lt 2\nV 0\n")):
            os.mkdir(self.path(name), 0o700)
            for file_name in ("KL.bin", "layout.txt"):
                self.put((name, file_name), "")
            self.put((name, "policy.txt"), text)
            self.refused("--config", self.path(name), words=r"policy\.txt is not a parameter file that this script can use: it needs a line lt with a whole number up to 16 and a line V")

    def test_requests_of_the_readers_own(self):
        os.mkdir(self.path("mine"), 0o700)
        mine = [spool.request([n, 1, 2, 3], TOKENS - 1, TOKENS) for n in range(11)]
        for n in range(9):
            self.put(("mine", "%02d" % n), mine[n])
        self.refused("--requests", self.path("mine"), words="holds 9 files, and 5 batches of 2 need 10. No window is sent twice")
        self.put(("mine", "09"), mine[0])
        self.refused("--requests", self.path("mine"), words="two of the first 10 files in \\S+ hold the same request")
        os.mkdir(self.path("other"), 0o700)
        for n in range(10):
            self.put(("other", "%02d" % n), mine[n] if n else spool.request([0, 1, 2, 3], 0, TOKENS))
        self.refused("--requests", self.path("other"), words="00 in \\S+ is not a request that this script takes: 4 token ids below 1000, then r = 3 and e = 4")
        os.mkdir(self.path("pipe"), 0o700)
        for n in range(1, 10):
            self.put(("pipe", "%02d" % n), mine[n])
        os.mkfifo(self.path("pipe", "00"), 0o600)
        self.refused("--requests", self.path("pipe"), words=r"00 in \S+ cannot be read \(it is not a regular file\)")
        os.mkdir(self.path("good"), 0o700)
        for n in range(11):
            self.put(("good", "%02d" % n), mine[n])
        code, out, _ = self.measure("--requests", self.path("good"))
        self.assertEqual(code, 0, out)
        self.assertIn("the first 10 of the 11 files in %s, in the order of their names." % self.path("good"), out)
        for n in range(10):
            with open(self.path("work", "requests", "%04d.req" % n), "rb") as f:
                self.assertEqual(f.read(), mine[n])

    def numpy_file(self, name, rows, descr="<i4", fortran_order=False, shape=None, version=b"\x01\x00", cut=0, hole=0):
        """Writes a NumPy file by hand, in the layout of the format's version 1.0, and returns its path. cut: so many bytes
        fewer. hole: so many bytes more, which take no room on the disk."""
        head = "{'descr': %r, 'fortran_order': %r, 'shape': %r, }" % (descr, fortran_order, shape or (len(rows), len(rows[0])))
        head = head.ljust(117 - 64 * (len(head) < 54)) + "\n"  # so that the data begins at a multiple of 64 bytes
        data = struct.pack("<%di" % sum(map(len, rows)), *(n for row in rows for n in row))
        self.put((name,), b"\x93NUMPY" + version + struct.pack("<H", len(head)) + head.encode() + data[: len(data) - cut], hole=hole)
        return self.path(name)

    def test_windows_of_text(self):
        rows = [(5, 1, 2, 3), (6, 7, 8, 9), (999, 0, 0, 1), (0, 4, 4, 5)]
        text = self.numpy_file("text.npy", rows)
        code, out, _ = self.measure("--text", text)
        self.assertEqual(code, 0, out)
        self.assertIn("10 windows from the 4 rows of %s (sha256 %s): window w is row w mod 4; from window 4 on, its first id stays and its other 3 ids are "
                      "rotated to the left by 100 x (w div 4) places." % (text, spool.sha256(text)), out)  # fmt: skip
        by_hand = [(5, 1, 2, 3), (6, 7, 8, 9), (999, 0, 0, 1), (0, 4, 4, 5), (5, 2, 3, 1), (6, 8, 9, 7), (999, 0, 1, 0), (0, 4, 5, 4), (5, 3, 1, 2), (6, 9, 7, 8)]
        self.assertEqual(len(set(by_hand)), 10)
        for n, window in enumerate(by_hand):
            with open(self.path("work", "requests", "%04d.req" % n), "rb") as f:
                self.assertEqual(f.read(), spool.request(window, TOKENS - 1, TOKENS))

    def test_files_of_text_that_are_refused(self):
        good = [(5, 1, 2, 3), (6, 7, 8, 9), (7, 1, 2, 3), (8, 1, 2, 3)]
        no_such_file = "is not a NumPy file of format 1.0 that holds little-endian int32 \\('<i4'\\) in C order, in rows of 4\\. Nothing was written"
        for name, words, how in (
            ("wide", no_such_file, {"descr": "<i8"}),
            ("big-endian", no_such_file, {"descr": ">i4"}),
            ("columns-first", no_such_file, {"fortran_order": True}),
            ("flat", no_such_file, {"shape": (16,)}),
            ("other-shape-than-its-data", no_such_file, {"shape": (4, 8)}),
            ("later-format", no_such_file, {"version": b"\x02\x00"}),
            ("short", no_such_file, {"cut": 1}),
            ("long", no_such_file, {"hole": 1}),
            ("far-longer-than-its-header-says", no_such_file, {"hole": 1 << 32}),
        ):
            self.refused("--text", self.numpy_file(name, good, **how), words=words)
        self.refused("--text", self.numpy_file("long-rows", [row + row for row in good]), words=no_such_file)
        self.put(("words",), "no NumPy file")
        self.refused("--text", self.path("words"), words=no_such_file)
        self.put(("large",), "no NumPy file", hole=1 << 33)
        self.refused("--text", self.path("large"), words=no_such_file)
        os.mkfifo(self.path("pipe"), 0o600)
        for name in ("pipe", "config"):
            self.refused("--text", self.path(name), words=r"%s, given with --text, cannot be read \(it is not a regular file\)" % name)
        self.refused("--text", self.path("absent"), words=r"absent, given with --text, cannot be read \(No such file or directory\)")
        for name, bad in (("at-V", VOCABULARY), ("below-0", -1)):
            self.refused("--text", self.numpy_file(name, good[:3] + [(8, 1, bad, 3)]), words="holds a token id that is not from 0 to 999, as the parameter file's line V asks")
        self.refused("--text", self.numpy_file("same", [(5, 1, 1, 1), (6, 7, 8, 9)]), words="the 2 rows of \\S+ do not give 10 different windows by this script's rule. No window is sent twice")
        self.refused("--text", self.numpy_file("good", good), "--requests", self.path("config"), words="--text and --requests cannot be given together")

    def test_clocks_that_lie_over_two_seconds_apart(self):
        code, out, _ = self.measure(stamp_off_ms=-5000)
        self.assertEqual(code, 0, out)
        self.assertIn("STEADY RATE: ", out)
        for pair in ("RELEASE and stamp", "seen and stamp"):
            self.assertEqual(re.findall(r"(?m)^NOTE: batch (\d): %s lie [45]\.\d{3} s apart, more than 2 s\.$" % pair, out), list("12345"))
        self.assertEqual(out.count("NOTE"), 10)
        self.assertIn("These should be readings of one clock: find out why they differ before you rely on a figure of this run.", out)

    def interrupted(self, event, *more, number=signal.SIGINT, times=1, script=(SCRIPT,), **scene):
        """Starts a run, sends a signal to its process group once the record has the event `times` times, and returns as measure() does."""
        self.set_the_scene(**scene)
        child = subprocess.Popen(
            self.command(*more, script=script),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            env=self.environment,
            start_new_session=True,
        )

        def end():
            if child.poll() is None:
                child.kill()
            child.wait()

        self.addCleanup(end)
        limit = time.monotonic() + 30
        while len(self.record(event)) < times:
            self.assertLess(time.monotonic(), limit, "the record never had %s %d times" % (event, times))
            time.sleep(0.02)
        os.killpg(child.pid, number)
        out, err = child.communicate(timeout=60)
        return child.returncode, out, err

    def test_ctrl_c_creates_the_stop_file_and_nothing_is_signalled(self):
        code, out, _ = self.interrupted("take", work_s=[1.5])
        self.no_rate(code, out, r"- Ctrl-C \(SIGINT\) asked this script to stop\n", r"- 8 of the requests got no response: numbered ")
        self.assertIn("Ctrl-C (SIGINT): this script was asked to stop. It has created the stop file, %s." % self.path("work", "spool", "stop"), out)
        self.assertEqual((len(self.record("stopped by the stop file")), len(self.record("release")), self.record("signal")), (1, 1, []))
        self.assertEqual(self.figures()["exit_code"], 0)
        self.assertIn("cp_verify: 2 accepted, 0 not accepted, 0 not checked", out)

    def test_ctrl_c_before_the_start_ends_the_script_and_starts_nothing(self):
        code, out, err = self.interrupted("describe", describe_s=3)
        self.assertEqual((code, err), (1, "measure_rate.py: interrupted: cp_prove was not started. The work directory is not removed: if it exists by now, name another for the next run\n"), out)
        self.assertEqual((self.record("call"), self.record("signal")), ([], []))
        self.assertIn("report.txt", os.listdir(self.path("work")))

    def test_a_stop_file_that_the_start_removes_is_created_again(self):
        code, out, _ = self.interrupted("call", stop_goes_after_s=1.5, start_up_s=1.5)
        self.no_rate(code, out, r"- Ctrl-C \(SIGINT\) asked this script to stop\n", r"- 10 of the requests got no response: numbered 0-9\n")
        self.assertEqual([len(self.record(event)) for event in ("the start removed a stop file", "stopped by the stop file", "take", "signal")], [1, 1, 0, 0])
        self.assertEqual(self.figures()["exit_code"], 0)

    def stopped_as_by_ctrl_c(self, number, name):
        code, out, _ = self.interrupted("take", number=number, work_s=[1.5])
        self.no_rate(code, out, r"- %s asked this script to stop\n" % name, r"- 8 of the requests got no response: numbered ")
        self.assertIn("%s: this script was asked to stop. It has created the stop file, " % name, out)
        self.assertEqual((len(self.record("stopped by the stop file")), len(self.record("release")), self.record("signal")), (1, 1, []))

    def test_sigterm_is_taken_as_ctrl_c(self):
        self.stopped_as_by_ctrl_c(signal.SIGTERM, "SIGTERM")

    def test_a_hang_up_is_taken_as_ctrl_c(self):
        self.stopped_as_by_ctrl_c(signal.SIGHUP, "SIGHUP")

    def test_ctrl_c_during_the_checks_starts_no_further_check(self):
        # Two calls run when the signal comes, and one of them ends well before the other: its place stays empty.
        code, out, _ = self.interrupted("check", "--checks-at-once", "2", check_s={self.sum_of_request(0): 1, self.sum_of_request(1): 2.5})
        self.no_rate(code, out, r"- 8 proofs were not checked, because a signal asked this script to stop\n")
        self.assertIn("cp_verify: 2 accepted, 0 not accepted, 8 not checked", out)
        self.assertEqual((len(self.record("check")), len(self.record("release")), self.figures()["exit_code"]), (2, 5, 0))

    def main_with(self, change):
        """Returns the words for command(script=...) that import the script as a module, run the statements `change` and call its main().
        -I: Python then takes no module from the working directory or from a directory that the environment names."""
        return ("-I", "-B", "-c", "import sys\nsys.path.insert(0, %r)\nimport measure_rate\n%s\nsys.exit(measure_rate.main())" % (TOP, change))

    def test_main_as_the_tests_call_it_takes_no_module_from_the_working_directory(self):
        self.put(("argparse.py",), "raise SystemExit('argparse.py was taken from the working directory')\n")
        done = subprocess.run([sys.executable, *self.main_with("pass"), "--help"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, env=self.environment, cwd=self.top)
        self.assertEqual((done.returncode, done.stderr), (0, ""), done.stdout)

    def test_ctrl_c_between_the_end_of_cp_prove_and_the_checks_starts_no_check(self):
        # No program runs in that span, so the script marks it itself: before it groups the responses into batches, it adds an
        # event to the stand-ins' record and waits for the signal.
        mark = ("import time\n"
                "group = measure_rate.batches_of\n"
                "def marked(*given):\n"
                "    with open(%r, 'a') as f:\n"
                "        f.write('{\"event\": \"grouping\"}\\n')\n"
                "    while not measure_rate.asked:\n"
                "        time.sleep(0.02)\n"
                "    return group(*given)\n"
                "measure_rate.batches_of = marked" % self.path("record"))  # fmt: skip
        code, out, _ = self.interrupted("grouping", script=self.main_with(mark))
        self.no_rate(code, out, r"- Ctrl-C \(SIGINT\) asked this script to stop\n", r"- 10 proofs were not checked, because a signal asked this script to stop\n")
        self.assertEqual((self.record("check"), len(self.record("release")), self.figures()["exit_code"]), ([], 5, 0))

    def test_ctrl_c_once_the_last_check_is_started_still_withholds_the_rate(self):
        # The tenth call is recorded, so none is waiting; the call on request 9 still runs when the signal comes.
        code, out, _ = self.interrupted("check", times=10, check_s={self.sum_of_request(9): 1.5})
        self.no_rate(code, out, r"- Ctrl-C \(SIGINT\) asked this script to stop\n")
        self.assertIn("cp_verify: 10 accepted, 0 not accepted, 0 not checked", out)

    def test_the_caution_on_secret_weights_stands_in_its_three_places(self):
        reach = "a further commitment to the same weights, which our judgement on weight hiding covers only on thin evidence and on conditions of its own"
        done = subprocess.run([sys.executable, SCRIPT, "--help"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, env=self.environment)
        self.assertEqual((done.returncode, done.stderr), (0, ""), done.stdout)
        self.assertIn("Measure on synthetic weights or a public model's files: with secret weights a commitment made for the measurement is %s." % reach, re.sub(r"\s+", " ", done.stdout))
        _, out, _ = self.measure(in_use=True)
        before_the_start, after_the_refusal = (re.sub(r"\s+", " ", part) for part in out[out.index("BEFORE THE START") :].split("The secret key directory is in use"))
        self.assertIn("on synthetic weights or a public model's files: with secret weights it is %s (driver/README.md, \"The rate\")." % reach, before_the_start)
        self.assertIn("Measure on synthetic weights or a public model's files: with secret weights such a commitment is %s (driver/README.md, \"The rate\")." % reach, after_the_refusal)

    def failed_beside_cp_prove(self, code, out, err, end):
        """What must hold after a fault of the script's own while cp_prove ran: told, waited for, told again, then the fault."""
        (call,) = self.record("call")
        self.assertEqual(code, 1, err)
        self.assertNotIn("tokens a second", out)
        told = re.sub(r"\s+", " ", err)
        running = "measure_rate.py: this script has failed (FileNotFoundError: [Errno 2] No such file or directory: '%s'), and cp_prove is still running, as process %d. " % (self.path("work", "spool", "in"), call["process_id"])
        for words in (running, "Do not kill it, and start no other cp_prove on this commitment", "Its output goes to %s. " % self.path("work", "cp_prove.txt"),
                      "press Ctrl-C once or create the empty file %s. " % self.path("work", "spool", "stop"), "It only waits for cp_prove's end, sends no signal"):  # fmt: skip
            self.assertIn(words, told)
        ended = "measure_rate.py: cp_prove (process %d) has ended with %s. " % (call["process_id"], end)
        self.assertLess(told.index(running), told.index(ended))
        self.assertLess(told.index(ended), told.index("Traceback (most recent call last)"))
        self.assertRegex(err, r"\nFileNotFoundError: .*\n$")
        self.assertEqual((len(self.record("end")), self.record("signal")), (1, []))

    def test_a_fault_of_the_script_while_cp_prove_runs(self):
        code, out, err = self.measure(in_is_away_s=1.5)
        self.failed_beside_cp_prove(code, out, err, "exit code 0")
        self.assertNotIn("docs/faults.md, \"Always first\" and", err)
        self.assertEqual((len(self.record("release")), self.record("stopped by the stop file")), (5, []))

    def test_such_a_fault_and_a_bad_end_of_cp_prove(self):
        code, out, err = self.measure(in_is_away_s=1.5, exit_code=9, last_line="p258 end 9 own-test-slot-pads")
        self.failed_beside_cp_prove(code, out, err, "exit code 9")
        self.assertIn("Do not start cp_prove on this commitment again, and remove nothing in the work directory, before you have read\ndocs/faults.md, \"Always first\" and", err)

    def test_ctrl_c_after_such_a_fault_still_creates_the_stop_file(self):
        code, out, err = self.interrupted("in/ is back", in_is_away_s=1.5, work_s=[1.5])
        self.failed_beside_cp_prove(code, out, err, "exit code 0")
        self.assertEqual((len(self.record("stopped by the stop file")), len(self.record("release"))), (1, 1))

    def test_files_of_the_script_that_cannot_grow(self):
        code, out, err = self.measure(largest_file=1500)
        self.assertEqual((code, err), (0, ""), out)
        self.assertIn("STEADY RATE: ", out)
        self.assertTrue(out.endswith("NOTE: report.txt is not whole: a write to it failed (File too large).\nNOTE: figures.json is not whole: a write to it failed (File too large).\n"), out)
        self.assertEqual([os.lstat(self.path("work", name)).st_size for name in ("report.txt", "figures.json")], [1500, 1500])
        self.assertEqual((len(self.record("release")), len(self.record("check")), self.record("signal")), (5, 10, []))

    def test_a_reader_of_the_output_that_has_gone_away(self):
        self.set_the_scene()
        read_end, write_end = os.pipe()
        os.close(read_end)
        done = subprocess.run(self.command(), stdout=write_end, stderr=subprocess.PIPE, universal_newlines=True, env=self.environment)
        os.close(write_end)
        self.assertEqual((done.returncode, done.stderr), (0, ""))
        with open(self.path("work", "report.txt"), encoding="utf-8") as f:
            self.assertIn("STEADY RATE: ", f.read())

    def test_an_nvidia_smi_that_does_not_answer_is_left_alone(self):
        # The script's own wait of 10 s would make the test long, so main() is called with a shorter one.
        self.set_the_scene(late_sample_s=6, gpu_mib=[100, 300], work_s=[0.7])
        done = subprocess.run(self.command(script=self.main_with("measure_rate.GPU_WAIT_S = 1")), stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, env=self.environment)
        self.assertEqual((done.returncode, done.stderr), (0, ""), done.stdout)
        first, late = self.record("sample")
        self.assertRegex(done.stdout, r"sampled once a second: the largest of 1 samples is 100 MiB; 1 got no answer; \d+\.\d s after cp_prove's start a call of nvidia-smi \(process %d\) "
                         r"had not answered after 1 s: the script left it alone and calls nvidia-smi no more, so a later peak is not seen\n" % late["process_id"])  # fmt: skip
        self.assertEqual(self.figures()["gpu_sampling_given_up"][1], late["process_id"])

    def test_without_nvidia_smi(self):
        self.environment["PATH"] = self.path("config")
        code, out, _ = self.measure()
        self.assertEqual(code, 0, out)
        self.assertIn("GPU: not known: nvidia-smi was not found on PATH, so the GPU is not described and its memory is not sampled\n", out)
        self.assertIn("the GPU's memory in use, sampled once a second: no sample\n", out)
        self.assertEqual(self.record("sample") + self.record("describe"), [])

    def test_an_nvidia_smi_that_does_not_answer_before_the_start_is_not_called_again(self):
        self.set_the_scene(describe_s=6)
        done = subprocess.run(self.command(script=self.main_with("measure_rate.GPU_WAIT_S = 1")), stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, env=self.environment)
        self.assertEqual((done.returncode, done.stderr), (0, ""), done.stdout)
        (late,) = self.record("describe")
        words = "before cp_prove's start a call of nvidia-smi (process %d) had not answered after 1 s: the script left it alone and calls nvidia-smi no more, so the GPU's memory is not sampled\n" % late["process_id"]
        self.assertIn("GPU: not known: %s gave no clear answer about GPU 0; %s" % (self.path("programs", "nvidia-smi"), words), done.stdout)
        self.assertIn("the GPU's memory in use, sampled once a second: no sample; " + words, done.stdout)
        self.assertNotIn("is asked for its memory in use once a second", done.stdout)
        self.assertEqual(self.record("sample"), [])


if __name__ == "__main__":
    run = unittest.main(verbosity=2, exit=False)
    if run.result.wasSuccessful() and OPEN_GROUND:
        sys.stderr.flush()
        print("stood aside: below an open place: %s: the temporary folders lie there, and measure_rate.py refuses every work directory, so the whole "
              "runs were not made. Run the tests below folders that only you or root can change (by hand: set TMPDIR to such a folder)" % OPEN_GROUND[0])
        sys.exit(77)
    sys.exit(not run.result.wasSuccessful())
