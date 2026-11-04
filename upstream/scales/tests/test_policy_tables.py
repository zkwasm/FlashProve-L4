#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""Tests scales/policy_tables.py: the hash, the four tables, the three modes of the command line and the widths.

usage: test_policy_tables.py

Except in test_values_away_from_the_examples, the expected digests and table entries come from
outside the module under test: Python's hashlib, verifier/tests/hash_vectors.txt, known answers
written here, parameter files of the tree (the list EXAMPLES), and the real functions evaluated at 60
decimal digits. verifier/tests/known_answers/format_hash.py prints hash_vectors.txt, computed with
f() in verifier/tests/known_answers/blake2b_f.py; apart from names and arguments, f() is the same
Python code for BLAKE2b's compression function as the module's _compress(), translated from
blake2b-ref.c of the BLAKE2 reference implementation. In this function the vectors expose a slip in
copying, not a fault that both share.
Exit status 0 and a last line "OK": every test passed.

A digest has 8 words of 32 bits, as in the parameter files of the list EXAMPLES.
The file SMALL belongs to the tests of integrations/vllm/integer_model. Do not copy or edit it. A case that needs another
value sets it here.
"""

import decimal
import glob
import hashlib
import os
import struct
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.join(HERE, "..", "..")
TOOL = os.path.join(HERE, "..", "policy_tables.py")
sys.dont_write_bytecode = True  # the import below leaves no file in the tree
sys.path.insert(0, os.path.dirname(TOOL))
import policy_tables  # noqa: E402

DW = policy_tables.DW  # the words of a digest
LINES = policy_tables.TBLW  # the digest lines of a parameter file
# The smallest parameter file of the tree with all four tables, and the only one without the line gcl. cp_commit and
# cp_prove refuse it, so no proof is made at it.
SMALL = os.path.join(TREE, "integrations", "vllm", "integer_model", "tests", "configurations", "tiny_without_clamps", "policy.txt")
EXAMPLES = sorted(glob.glob(os.path.join(TREE, "prover", "examples", "*", "policy*.txt"))) + [os.path.join(TREE, x, "policy.txt") for x in ("prover/tests/configurations/pinned", "prover/tests/configurations/pinned_no_proof_of_work", "prover/tests/configurations/pinned_no_proof_of_work_roomy")] + [SMALL]  # the examples; the tests' own files, by name; the small file  # fmt: skip


def read_lines(path):
    with open(path, encoding="ascii", newline="") as f:
        return f.read()[:-1].split("\n")


def sequence_bytes(count):
    """The first count bytes of the test list: word k is ((k + 1) * 0x89ABCDEF) mod 2^32, lowest byte first."""
    n = -(-count // 4)
    return struct.pack("<%dI" % n, *((k + 1) * 0x89ABCDEF % 2**32 for k in range(n)))[:count]


def printed(digest):
    return "".join("%08x" % word for word in digest)


class Hash(unittest.TestCase):
    def test_the_compression_function_chained_as_in_the_rfc_is_blake2b(self):
        """With the counter's high half at 0, the chain of RFC 7693, section 3.3, must give hashlib's BLAKE2b-512."""
        for count in (0, 1, 3, 127, 128, 129, 256, 300):
            data = sequence_bytes(count)
            blocks = max(1, -(-count // 128))
            padded = data.ljust(128 * blocks, b"\0")
            h = (policy_tables._IV[0] ^ 0x01010040,) + policy_tables._IV[1:]
            for k in range(blocks):
                last = k == blocks - 1
                m = struct.unpack_from("<16Q", padded, 128 * k)
                h = policy_tables._compress(h, m, count if last else 128 * (k + 1), 0, last)
            self.assertEqual(struct.pack("<8Q", *h), hashlib.blake2b(data).digest(), "%d bytes" % count)
        # hash_words() has the counter's high half at 0 only for num = 0 and the empty list.
        self.assertEqual(printed(policy_tables.hash_words(0, [])),
                         printed(struct.unpack("<%dI" % DW, hashlib.blake2b(b"").digest()[: 4 * DW])))  # fmt: skip

    def test_the_vectors_of_the_verifier(self):
        """A line "H <num> <L> <state>" gives the hash value of the test list of L words (sequence_bytes)."""
        with open(os.path.join(TREE, "verifier", "tests", "hash_vectors.txt"), encoding="ascii") as f:
            vectors = [line.split() for line in f if line.startswith("H ")]
        self.assertGreaterEqual(len(vectors), 44)
        for _, num, count, state in vectors:
            words = list(struct.unpack("<%dI" % int(count), sequence_bytes(4 * int(count))))
            digest = policy_tables.hash_words(int(num), words)
            self.assertEqual(len(digest), DW)
            self.assertEqual(printed(digest), state[: 8 * DW], "H %s %s" % (num, count))

    def test_the_known_answers_of_the_protocol_document(self):
        """Known answers. The second and third differ only in the list's length.

        verifier/HASHES.md, "Worked values", has the first three and the fifth.
        The answers are written here with 10 words. A digest of 8 words is their first 8.
        """
        for num, words, answer in (
            (10, [], "642033eebd8188a6b2b6dba3ad4e81de70b03c7d9bc86eec189b6da891098fc0c55bfef48b29f4ad"),
            (10, [7], "9199d8e76712b9d5a45ca6ffa7333389d9332b0a830f4274cc71888ff4dd062995295ae7d456aacb"),
            (10, [7, 0], "5270ece847c0c6e6215c895e003b513dc694c460d5091150f1a79692658fcb047fca24176960d0ba"),
            (109, range(1, 13), "5c82c4495384aa74947f336ddd7f0cf56358b10f67b08f5eb23cd2d11daeceb013950b708dbca5f6"),
            (3, range(1, 33), "3d11700a6c1b3f588c605790c666049f2a3b0711fb1cdb03fc8cba0a088f932b0a1ae7d316a75cc2"),
            (10, range(1, 34), "f1f668f5decb4f06babdebc806bf666f12603a7c63785f62d5659df3522f4f99c4184d5ea65602c0"),
        ):
            self.assertEqual(printed(policy_tables.hash_words(num, list(words))), answer[: 8 * DW])

    def test_arguments_for_which_the_hash_is_not_defined(self):
        words = list(range(40))
        for num, faulty in ((-1, words), (2**32, words), (72, [2**32]), (72, [-1]), (72, words + [2**32])):
            self.assertRaises(ValueError, policy_tables.hash_words, num, faulty)
        self.assertEqual(policy_tables.hash_words(72, tuple(words)), policy_tables.hash_words(72, words))


class Tables(unittest.TestCase):
    """The SiLU, sigmoid and softmax tables, entry by entry, against the real functions they round.

    Of the rotary tables only the sizes, each pair's radius and a few entries are tested.
    The tables are defined in docs/model.md, "4 Public tables".
    """

    @classmethod
    def setUpClass(cls):
        decimal.getcontext().prec = 60

    @staticmethod
    def nearest(x, ties=decimal.ROUND_HALF_UP):
        """x rounded to an integer. decimal's ROUND_HALF_UP rounds ties away from zero."""
        return int(x.quantize(decimal.Decimal(1), rounding=ties))

    def test_the_silu_table(self):
        table = dict(zip(policy_tables.KEYS, policy_tables.silu_table()))
        above = []
        for c, entry in table.items():
            z = decimal.Decimal(min(max(c, -16384), 16383)) / 256
            exact = min(max(self.nearest(64 * z / (1 + (-z).exp())), -4095), 4095)
            if entry != exact:
                self.assertEqual(entry, exact + 1, "key %d" % c)
                above.append(c)
        # For z above 53 ln 2 the binary64 sum 1 + exp(-z) is 1, so 64 z is a tie at these keys and is rounded up.
        self.assertEqual(above, list(range(9406, 16382, 4)))
        self.assertEqual(len(above), 1744)
        # At eg = 7 a key step is 2^-7: key c has the entry of key 2 c at eg = 8, clamped by smax.
        coarse = dict(zip(policy_tables.KEYS, policy_tables.silu_table(7, 255)))
        self.assertTrue(all(coarse[c] == min(table[2 * c], 255) for c in range(-8192, 8192)))
        self.assertEqual((coarse[-32768], coarse[32767]), (coarse[-8192], coarse[8191]))
        # SiLU's minimum is -0.278.., which is -18 in steps of 2^-6; smax also clamps from below.
        self.assertEqual((min(table.values()), min(policy_tables.silu_table(8, 3))), (-18, -3))
        # Keys are clamped to -64 * 2^eg .. 64 * 2^eg - 1. At the upper end 1 + exp(-z) is 1, so the entry is 64 z.
        for eg, last in ((4, 4092), (5, 4094)):
            clamped = policy_tables.silu_table(eg)
            self.assertEqual((clamped[0], clamped[-1]), (0, last))

    def test_the_sigmoid_table(self):
        table = policy_tables.sigmoid_table()
        for c, entry in zip(policy_tables.KEYS, table):
            self.assertEqual(entry, 16 * self.nearest(4096 / (1 + (decimal.Decimal(-c) / 256).exp())), "key %d" % c)
        self.assertEqual((min(table), max(table)), (0, 65536))

    def test_the_softmax_table(self):
        for exp_s, rx, rows in ((12, 8, 2307), (12, 5, 289), (11, 8, None), (13, 7, None)):
            table = policy_tables.softmax_table(exp_s, rx)
            for u, entry in enumerate(table + [0]):  # no entry is a tie, so the rule for ties decides nothing here
                exact = 2**exp_s * (decimal.Decimal(-u) / 2**rx).exp()
                self.assertEqual(entry, self.nearest(exact, decimal.ROUND_HALF_EVEN), "S %d, rx %d, u %d" % (exp_s, rx, u))
            self.assertNotIn(0, table)
            if rows:
                self.assertEqual(len(table), rows)

    def test_the_rotary_tables(self):
        cosines, sines = policy_tables.rotary_tables(1024, 8, 12, 500000)
        self.assertEqual((len(cosines), len(sines)), (1024 * 4, 1024 * 4))
        self.assertEqual((cosines[:4], sines[:4]), ([4096] * 4, [0] * 4))  # position 0
        # Each pair is a point of the circle of radius 4,096, rounded to the grid.
        self.assertTrue(all(abs(c * c + s * s - 4096**2) <= 2 * 4096 for c, s in zip(cosines, sines)))
        # Pair 0 turns by one radian per position: cos 1 = 0.5403.., sin 1 = 0.8414.., cos 3 = -0.9899.., sin 3 = 0.1411..
        self.assertEqual((cosines[4], sines[4], cosines[12], sines[12]), (2213, 3447, -4055, 578))
        # The scale is 2^cs, and an odd dh has floor(dh / 2) pairs.
        self.assertEqual(policy_tables.rotary_tables(1, 2, 10, 10000), ([1024], [0]))
        self.assertEqual(policy_tables.rotary_tables(1, 1, 12, 10000), ([], []))
        self.assertEqual(len(policy_tables.rotary_tables(1, 127, 12, 10000)[0]), 63)
        # At the base 2^40 and dh = 8, pair 1 turns by exactly 2^-10 radians per position: 1,024 times slower than pair 0.
        cosines, sines = policy_tables.rotary_tables(3073, 8, 12, 2**40)
        for t in range(4):
            self.assertEqual((cosines[4096 * t + 1], sines[4096 * t + 1]), (cosines[4 * t], sines[4 * t]))
        # An entry next to a tie, which depends on the platform's library: either value passes.
        self.assertIn(policy_tables.rotary_tables(690, 128, 12, 3675548936)[0][689 * 64 + 1], (-548, -547))

    def test_ties_are_rounded_away_from_zero(self):
        self.assertEqual([policy_tables._round_half_away(v) for v in (0.5, 1.5, 2.5, -0.5, -1.5, -2.5)], [1, 2, 3, -1, -2, -3])

    def test_the_digest_lines_of_the_examples(self):
        """The files' digest lines. `cp_verify` also computes and compares them, and `cp_prove` does at the files at which a proof is made."""
        for path in EXAMPLES:
            lines = read_lines(path)
            stated = [line for line in lines if policy_tables.is_table_line(line)]
            self.assertEqual(len(stated), LINES, path)
            self.assertEqual(policy_tables.table_lines(policy_tables.digests(lines)), stated, path)
        clamped = sum("gcl 1" in read_lines(path) for path in EXAMPLES)
        self.assertTrue(0 < clamped < len(EXAMPLES), "the files must cover files with and without the line gcl")
        self.assertEqual(len(EXAMPLES), 10, "the number of parameter files under prover/examples/ is not 10: one has come or gone")


class Program(unittest.TestCase):
    """The command line, on variants of the small model's parameter file.

    One test also uses variants of an example's file.
    """

    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        cls.good = read_lines(SMALL)
        cls.first = cls.good.index(next(line for line in cls.good if line.startswith("tsilu0 ")))
        cls.tables = cls.good[cls.first : cls.first + LINES]
        cls.plain = cls.good[: cls.first] + cls.good[cls.first + LINES :]

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def run_tool(self, *arguments):
        """Runs the program.

        An argument of lines or bytes is written to a file, and the file's path is passed instead.
        """
        words = []
        for argument in arguments:
            if not isinstance(argument, str):
                content = argument if isinstance(argument, bytes) else ("\n".join(argument) + "\n").encode("ascii")
                argument = os.path.join(self.folder.name, "policy.txt")
                with open(argument, "wb") as f:
                    f.write(content)
            words.append(argument)
        return subprocess.run([sys.executable, "-B", TOOL, *words], capture_output=True, check=False)

    def with_tables(self, tables):
        return self.good[: self.first] + tables + self.good[self.first + LINES :]

    def changed(self, *numbers):
        """The digest lines, with the value's lowest bit flipped in the lines at these indices."""
        lines = list(self.tables)
        for i in numbers:
            name, value = lines[i].split(" ")
            lines[i] = "%s %d" % (name, int(value) ^ 1)
        return lines

    def replaced(self, name, *lines, base=None):
        """The good file, or base, with other lines in the place of the line of this name."""
        base = base or self.good
        at = base.index(next(line for line in base if line.startswith(name + " ")))
        return base[:at] + list(lines) + base[at + 1 :]

    def optional(self, *lines, base=None):
        """The good file, or base, with these lines (eg, smax or gcl) directly after the line F, where the format has them."""
        return self.replaced("F", next(line for line in self.good if line.startswith("F ")), *lines, base=base)

    def assert_prints(self, run, lines):
        self.assertEqual((run.returncode, run.stdout, run.stderr), (0, ("\n".join(lines) + "\n").encode("ascii"), b""))

    def assert_rejected(self, run, what):
        """Status 2, one line on standard error and nothing on standard output."""
        self.assertEqual((run.returncode, run.stdout, run.stderr.count(b"\n")), (2, b"", 1), what)

    def test_the_default_mode_makes_the_lines(self):
        self.assertTrue(self.good[self.first + LINES].startswith("call "))
        tables = self.tables
        for what, lines in (
            ("right lines", self.good),
            ("no lines", self.plain),
            ("a wrong value", self.with_tables(self.changed(2 * DW + 5))),
            ("one line fewer", self.with_tables(tables[:-1])),
            ("a line more in a group, and a line twice", self.with_tables(tables[:DW] + ["tsilu%d 5" % DW] + tables[DW - 1 :])),
            ("the lines after the first call line", self.plain[: self.first + 1] + tables + self.plain[self.first + 1 :]),
            ("the lines scattered", tables[:7] + self.plain[:30] + tables[7:] + self.plain[30:]),
        ):
            with self.subTest(what):
                self.assert_prints(self.run_tool(lines), self.good)
        # A file with the line gcl, whose sigmoid table differs, and with its digest lines at the end.
        good = read_lines(os.path.join(TREE, "prover", "examples", "target_small_f2", "policy.txt"))
        plain = [line for line in good if not policy_tables.is_table_line(line)]
        tables = [line for line in good if policy_tables.is_table_line(line)]
        self.assertIn("gcl 1", good)
        self.assert_prints(self.run_tool(plain + tables), good)
        # A line that the program does not read is copied byte for byte, whatever its form.
        odd = ["odd  line\t\r", "", "tsilu 5", "tsilux 5", "callx 1"]
        self.assert_prints(self.run_tool(self.plain[:3] + odd + self.plain[3:]), self.good[:3] + odd + self.good[3:])

    def test_each_table_follows_its_own_lines(self):
        """A change of a line changes the digests of the tables that depend on it, and no others."""
        for lines, tables in (
            (self.optional("eg 7"), {"tsilu"}),
            (self.optional("smax 255"), {"tsilu"}),
            (self.optional("gcl 1"), {"tsig"}),
            (self.replaced("theta0_hi", "theta0_hi 1"), {"trope"}),
            (self.replaced("lt", "lt 5"), {"trope"}),
            (self.replaced("dh", "dh 14"), {"trope"}),
            (self.replaced("cs", "cs 11"), {"trope"}),
            (self.replaced("S", "S 13"), {"texp"}),
            (self.replaced("rx", "rx 7"), {"texp"}),
            (self.replaced("d", "d 128"), set()),
        ):
            run = self.run_tool(lines)
            self.assertEqual((run.returncode, run.stderr), (0, b""))
            made = run.stdout.decode("ascii").split("\n")
            moved = {line.rstrip("0123456789 ") for line in made if policy_tables.is_table_line(line) and line not in self.good}
            self.assertEqual(moved, tables)

    def test_which_lines_give_a_value(self):
        """A line counts if it is a name, one space and digits; the first such line of a name counts; others are copied."""
        for name, *lines in (
            ("S", "S 12", "S 11"),
            ("S", "S x", "S 12"),
            ("S", "S 012"),
            ("S", "S " + 5000 * "0" + "12"),
            ("S", "S -11", "S +11", "S 11.0", "S 0x0b", "S  11", "S\t11", "S 11 ", "S 11 # note", "S 11\r", "S 12"),
            ("S", "S 12", "S " + 5000 * "1"),
        ):
            with self.subTest(lines):
                self.assert_prints(self.run_tool(self.replaced(name, *lines, base=self.plain)), self.replaced(name, *lines))
        unusable = ("eg 7 ", "smax  255", "gcl")
        self.assert_prints(self.run_tool(self.optional(*unusable, base=self.plain)), self.optional(*unusable))
        # Only the presence of the line gcl counts, not its value.
        clamped = self.run_tool(self.optional("gcl 1")).stdout
        self.assertEqual(self.run_tool(self.optional("gcl 0")).stdout, clamped.replace(b"gcl 1", b"gcl 0"))

    def test_values_away_from_the_examples(self):
        """Each line reaches its table, at values that no example has.

        With each row's values `cp_verify` accepts no proof. The expected words come from the
        module's own table functions and hash_words(), which the classes Tables and Hash test.
        """
        for changes, silu, rotary, softmax in (
            ({"cs": 10}, (8, 4095), (16, 16, 10, 10000), (12, 5)),  # the rotary offset is 4,096 at every cs
            ({"theta0_lo": 0, "theta0_hi": 256}, (8, 4095), (16, 16, 12, 2**40), (12, 5)),
            ({"dh": 258, "S": 31, "eg": 40, "smax": 5000}, (40, 5000), (16, 258, 12, 10000), (31, 5)),
            ({"lt": 11, "dh": 127, "rx": 12, "eg": 0, "smax": 0}, (0, 0), (2048, 127, 12, 10000), (12, 12)),
            ({"lt": 0, "dh": 1, "S": 0, "rx": 0}, (8, 4095), (1, 1, 12, 10000), (0, 0)),  # no pair: an empty rotary list
        ):
            lines = self.optional(*("%s %d" % (name, changes[name]) for name in ("eg", "smax") if name in changes))
            for name in changes.keys() - {"eg", "smax"}:
                lines = self.replaced(name, "%s %d" % (name, changes[name]), base=lines)
            hashed = (
                [entry + 4096 for entry in policy_tables.silu_table(*silu)],
                policy_tables.sigmoid_table(),
                [entry + 4096 for entry in sum(policy_tables.rotary_tables(*rotary), [])],
                policy_tables.softmax_table(*softmax),
            )
            expected = ["%s%d %d" % (name, i, word) for name, words in zip(("tsilu", "tsig", "trope", "texp"), hashed)
                        for i, word in enumerate(policy_tables.hash_words(72, words))]  # fmt: skip
            with self.subTest(changes):
                run = self.run_tool(lines)
                self.assertEqual((run.returncode, run.stderr), (0, b""))
                made = run.stdout.decode("ascii").split("\n")
                self.assertEqual([line for line in made if policy_tables.is_table_line(line)], expected)

    def test_strip_takes_the_lines_out(self):
        self.assert_prints(self.run_tool("--strip", self.good), self.plain)
        self.assert_prints(self.run_tool("--strip", self.plain), self.plain)
        self.assertEqual(self.run_tool("--strip", self.tables).stdout, b"")
        mixed = self.tables[-10:] + self.plain + self.tables[:9] + ["texp10 1"]
        self.assert_prints(self.run_tool("--strip", mixed), self.plain)
        # --strip looks only for digest lines, so it also accepts files that another mode rejects.
        for name, *lines in (("prof", "prof 19"), ("NE", "NE 0"), ("dh",)):
            stripped = self.replaced(name, *lines, base=self.plain)
            self.assert_prints(self.run_tool("--strip", self.replaced(name, *lines)), stripped)
        no_call = [x for x in self.good if not x.startswith("call ")]
        self.assert_prints(self.run_tool("--strip", no_call), [x for x in self.plain if not x.startswith("call ")])
        self.assert_prints(self.run_tool("--strip", [line + "\r" for line in self.good]), [line + "\r" for line in self.plain])

    def test_check_compares_the_lines_with_the_tables(self):
        """`--check` prints to standard output one line for each group of digest lines that differs.

        All right, no digest lines, or only the groups in another order: one line.
        """
        tables = self.tables
        for what, lines, groups in (
            ("right lines", self.good, None),
            ("right lines, scattered", tables[:7] + self.plain[:30] + tables[7:] + self.plain[30:], None),
            ("right lines, no call line", [x for x in self.good if not x.startswith("call ")], None),
            ("the first value", self.with_tables(self.changed(0)), ["tsilu0"]),
            ("the last value", self.with_tables(self.changed(LINES - 1)), ["texp0"]),
            ("values in all groups", self.with_tables(self.changed(*(DW * g + 5 for g in range(4)))),
             ["tsilu0", "tsig0", "trope0", "texp0"]),  # fmt: skip
            ("a line missing", self.with_tables(tables[:-1]), ["texp0"]),
            ("one line more", self.with_tables(tables + ["texp%d 1" % DW]), ["texp0"]),
            ("two lines swapped", self.with_tables(tables[:4] + [tables[5], tables[4]] + tables[6:]), ["tsilu0"]),
            ("a leading zero", self.with_tables([x.replace(" ", " 0") if x.startswith("trope1 ") else x for x in tables]),
             ["trope0"]),  # fmt: skip
            ("a line gcl for which the lines were not made", self.optional("gcl 1"), ["tsig0"]),
            ("the groups in another order", self.with_tables(tables[DW : 2 * DW] + tables[:DW] + tables[2 * DW :]), [""]),
            ("no lines", self.plain, [""]),
        ):
            with self.subTest(what):
                run = self.run_tool("--check", lines)
                printed_lines = run.stdout.decode("ascii").split("\n")[:-1]
                self.assertEqual((run.returncode, run.stderr), (1 if groups else 0, b""))
                self.assertEqual(len(printed_lines), len(groups or [""]))
                for group, line in zip(groups or [], printed_lines):
                    self.assertIn(group, line)

    def test_rejections(self):
        text = ("\n".join(self.good) + "\n").encode("ascii")
        every, computing = ((), ("--check",), ("--strip",)), ((), ("--check",))
        for what, content, modes in (
            ("a non-ASCII byte", text.replace(b"version", b"versi\xc3\xb3n"), every),
            ("no line feed at the end", text[:-1], every),
            ("an empty file", b"", every),
            ("prof 19", self.replaced("prof", "prof 19"), computing),
            ("prof 21", self.replaced("prof", "prof 21"), computing),
            ("NE 0", self.replaced("NE", "NE 0"), computing),
            ("NE 0, no digest lines", self.replaced("NE", "NE 0", base=self.plain), computing),
            ("theta0 is 0", self.replaced("theta0_lo", "theta0_lo 0"), computing),
            ("CR LF line ends", [line + "\r" for line in self.good], computing),
            ("cs above the limit", self.replaced("cs", "cs 13"), computing),
            ("S above the limit", self.replaced("S", "S 32"), computing),
            ("a value of 5,000 digits", self.optional("smax " + 5000 * "1"), computing),
            ("no call line", [x for x in self.good if not x.startswith("call ")], ((),)),
        ):
            for mode in modes:
                self.assert_rejected(self.run_tool(*mode, content), (what, mode))
        for name in ("prof", "NE", "lt", "dh", "cs", "theta0_lo", "theta0_hi", "S", "rx"):
            for mode in computing:
                run = self.run_tool(*mode, self.replaced(name, name + " x"))
                self.assert_rejected(run, (name, mode))
                self.assertIn(b'"%s ' % name.encode("ascii"), run.stderr)
        # Of several missing lines the message names the first in the order above.
        self.assertIn(b'"lt ', self.run_tool(self.replaced("S", base=self.replaced("lt"))).stderr)
        for arguments in (
            [os.path.join(self.folder.name, "absent.txt")],
            ["--check", os.path.join(self.folder.name, "absent.txt")],
            ["--strip", self.folder.name],
            ["--help"],
            ["-"],
            ["--check", "-h"],
            [],
            ["--check"],
            ["--check", "--strip"],
            [self.good, self.good],
            [self.good, "--check"],
            ["--other", self.good],
            ["--dw", "8", self.good],
            ["--check", "--strip", self.good],
        ):
            self.assert_rejected(self.run_tool(*arguments), arguments[:1])
        # The usage line does not depend on the docstring, which python3 -OO drops.
        self.assertIn(policy_tables.USAGE, policy_tables.__doc__)
        run = subprocess.run([sys.executable, "-B", "-OO", TOOL], capture_output=True, check=False)
        self.assertEqual((run.returncode, run.stderr.decode("ascii")), (2, policy_tables.USAGE + "\n"))


class Module(unittest.TestCase):
    def test_the_widths(self):
        """The values that the format fixes.

        A static_assert in prover/common/setting.h has the numbers of THPW, HEXW and TBLW.
        There LAMW is DW, and BAR is HF_BAR of prover/hash_commitment/hf_table_format.h.
        """
        shared = {
            "SW": 16, "BLKW": 32, "SALTW": 10, "SALT_CELLS": 5, "SESSW": 10, "NONCEW": 4, "RETRYQ": 8, "FILLC": 8,
            "SLABR": 16, "SH2_DIGB": 32,
        }  # fmt: skip
        by_width = {"DW": 8, "THPW": 128, "HEXW": 64, "LAMW": 8, "BAR": 128, "TBLW": 32,
                    "HC_REC0W": 33, "HC_RECW": 41, "HC_TWW": 8, "HC_CARDW": 49}  # fmt: skip
        widths = {**shared, **by_width}
        for name, value in widths.items():
            self.assertIs(type(getattr(policy_tables, name)), int, name)
            self.assertEqual(getattr(policy_tables, name), value, name)

    def test_an_import_prints_nothing(self):
        run = subprocess.run([sys.executable, "-B", "-c", "import policy_tables"], cwd=os.path.dirname(TOOL),
                             capture_output=True, check=False)  # fmt: skip
        self.assertEqual((run.returncode, run.stdout, run.stderr), (0, b"", b""))

    def test_an_import_leaves_the_command_line_alone(self):
        """The module reads sys.argv only when it is run as a program: a script that imports it keeps all its arguments."""
        for words in (["--check", "--dw", "8", "policy.txt"], ["--dw", "10"], ["--help"]):
            run = subprocess.run([sys.executable, "-B", "-c", "import sys, policy_tables; print(sys.argv[1:])", *words],
                                 cwd=os.path.dirname(TOOL), capture_output=True, text=True, check=False)  # fmt: skip
            self.assertEqual((run.returncode, run.stdout, run.stderr), (0, repr(words) + "\n", ""), words)

    def test_digests_raises_a_value_error(self):
        """digest_lines() in scales/words_to_policy.py catches this exception."""
        lines = read_lines(SMALL)
        without_dh = [line for line in lines if not line.startswith("dh ")]
        other_profile = ["prof 19" if line == "prof 20" else line for line in lines]
        for faulty in ([], without_dh, other_profile):
            self.assertRaises(ValueError, policy_tables.digests, faulty)

    def test_which_lines_are_digest_lines(self):
        for line in ("tsilu0 1", "tsig9 4294967295", "trope5 0", "texp9 7", "texp10 1", "tsilu007 x y"):
            self.assertTrue(policy_tables.is_table_line(line), line)
        for line in ("tsilu 1", "tsilux 5", "tsilu0x 5", "texp 9", "S 12", "call FS FS MR", "xtsilu0 1", " tsilu0 1", ""):
            self.assertFalse(policy_tables.is_table_line(line), line)


if __name__ == "__main__":
    unittest.main()
