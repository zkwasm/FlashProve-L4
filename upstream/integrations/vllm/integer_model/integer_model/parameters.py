# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""The public parameters of the integer model.

Two files hold them. The parameter file (`policy.txt`) is a text file: lines on the proof system,
then "name value" lines on the model and its proof, then call lines (the proof's lookup calls). The
file of quantization scales (`KL.bin`) holds each layer's shifts and norm exponents as 32-bit words.

The same file must mean the same model here and in the prover, so the reader enforces the grammar
and the fixed order of lines that verifier/docs/parameter-file.md defines. Parameters.read states
where it is stricter than the verifier and where it is laxer.

Parameters.read reads both files. given_digests validates the caller's SHA-256 digests, and
read_bytes checks a file against one; other modules call both.
"""

import hashlib
import json
import os
import re
import stat
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

CLAMPED_MAX = 2047  # the largest magnitude of a clamped activation (docs/model.md, "5 The norm")
WEIGHT_BITS = 13  # the width of a matrix weight, fixed by the format (every line wb_* but wb_emb)
WEIGHT_MIN, WEIGHT_MAX = -(1 << (WEIGHT_BITS - 1)), (1 << (WEIGHT_BITS - 1)) - 1

_WEIGHT_WIDTH_LINES = (
    "wb_qk wb_v wb_o wb_g wb_u wb_gs wb_us wb_gr wb_ur wb_dn wb_dnr wb_r wb_head".split()
)
_TABLE_DIGEST_LINES = ("tsilu", "tsig", "trope", "texp")

# The grammar of a value line: one space, no sign, no leading zero, no other white space.
_VALUE_LINE = re.compile(r"([!-~]{1,63}) (0|[1-9][0-9]{0,9})")
_CALL_LINE = re.compile(r"(call|callA|callH)( [A-Za-z0-9._-]+){3,}")
# The lines on the proof system. Each is a name of one or two words and a fixed number of values
# (None: a multiple of four, from 4 to 32). The value of the first, `version`, is four characters;
# every other value is a decimal number of at most 15 digits.
_VERSIONS = ("H821",)
_GROUPS = "link folded omega own once-folded once-omega once-own once-ranges".split()
_PROOF_SYSTEM_LINES: list[tuple[str, int | None]] = [
    *((name, 1) for name in "sec reading rowcost slots answers record".split()),
    *((f"work {kind}", 3) for kind in "field sample closeness positions smallness".split()),
    ("salt", 1),
    ("listed", 1),
    *(line for group in _GROUPS for line in ((f"group {group}", 5), (f"stages {group}", None))),
    ("ranges-part", 1),
    ("table", 6),
]
_PROOF_SYSTEM_VALUE = re.compile(r"0|[1-9][0-9]{0,14}")
# The names of the value lines that follow, in their fixed order. Names ending in "?" are optional.
# nminH is optional too, and may appear only right after lidb, where _read_lines inserts it.
_LINE_ORDER = (
    "prof lt L ll d H G dh f fe NE V eps1_lo eps1_hi eps2_lo eps2_hi cs theta0_lo "
    "theta0_hi hq hv ho hgD huD hdD hr ut la nmin D tau lmaf NAf lmad NAd lmar lmb lmq lmx wb_qk "
    "wb_v wb_o wb_g wb_u wb_gs wb_us wb_gr wb_ur wb_dn wb_dnr wb_r iota_0 iota_1 iota_2 iota_3 "
    "iota_4 iota_5 iota_6 iota_7 seed_lo seed_hi lB S rx kFM kFD lmS NAs lmh lmo lmbA iotaA_0 MF "
    "KF epsF_lo epsF_hi toksrc lmhf NHf lmhd lmhr lmbH wb_emb wb_head iotaH_0 iotaH_1 iotaH_2 "
    "iotaH_3 iotaH_4 iotaH_5 C theta cap s sh sT F lidb? eg? smax? rcl? gcl? xl? "
).split()
_LINE_ORDER += [f"{name}{i}" for name in _TABLE_DIGEST_LINES for i in range(8)]
# Lines that allow one value only. An optional line must not give its default value. wb_emb is a
# constant of the format, and this package only checks it: it is no bound on the embedding's
# entries (for that, see EMBEDDING_LIMIT in block_weights.py).
_ONLY_VALUE = {
    "reading": 1, "rowcost": 1, "answers": 1, "listed": 0,
    "prof": 20, "ut": 1, "wb_emb": 21, "rcl": 1, "gcl": 1, "xl": 5, "tau": 128, "toksrc": 1,
    "F": 24,
    **dict.fromkeys(_WEIGHT_WIDTH_LINES, WEIGHT_BITS),
}  # fmt: skip
# An optional line that only concerns the proof: the smallest and largest allowed value. Its default
# is one below the smallest value.
_RANGE_IF_PRESENT = {"lidb": (23, 30)}
_MOST_CALL_LINES = {"call": 8, "callH": 6}
_NO_EXPERTS = (
    "NE is 0, a model without routed experts. A parameter file for such a model cannot carry the "
    "digests of the public tables, which this package requires."
)
_NO_DIGESTS = (
    " The 32 lines tsilu0 to texp7 carry the digests of the public tables. Without them nothing "
    "shows that this machine's tables are those of the proved model. scales/policy_tables.py "
    "adds them to a parameter file."
)
_PLACE_OF_NMINH = " The line nminH is allowed once, directly after a line lidb."


class ParameterError(ValueError):
    """The parameter file or the file of quantization scales does not describe a model that this
    package can evaluate.

    Also raised for a mistake in integer_model.json or in the options, and for a file that does not
    have the given SHA-256.
    """


@dataclass(frozen=True)
class LayerScales:
    """The quantization scales of one layer, in the order of `KL.bin`, under the names of
    docs/model.md, "3 Weights and quantization scales"."""

    kq: int  # right shifts of the seven matrix products
    kk: int
    kv: int
    ko: int
    kg: int
    ku: int
    kd: int
    M1: int  # exponents of the layer's first norm
    K1: int
    M2: int  # exponents of the layer's second norm
    K2: int
    sh: int  # right shift that gives a softmax key
    rp: int  # 1 if the layer applies rotary position encoding, 0 otherwise
    # Mixture-of-experts layers only: shifts of the router and of the routed expert's gate and up.
    kr: int = 0
    kgR: int = 0
    kuR: int = 0


@dataclass(frozen=True)
class Parameters:
    """Shape, table parameters, norm parameters and quantization scales, under the names of
    docs/model.md, "2 Shape parameters" and "3 Weights and quantization scales", where that
    document names them."""

    T: int  # number of positions of a window
    L: int  # number of layers
    d: int  # width of the residual stream
    n_h: int  # query heads
    n_g: int  # key and value heads
    dh: int  # head dimension
    f: int  # hidden width of a feed-forward layer; with experts, f = 2 fe (shared plus routed)
    fe: int  # hidden width of one expert
    NE: int  # number of routed experts; 0 if no layer has experts
    V: int  # vocabulary size
    cs: int  # scale exponent of the rotary tables
    theta0: int  # base of the rotary tables
    expS: int  # scale exponent of the softmax table
    rx: int  # step exponent of the softmax table
    eg: int  # key step exponent of the SiLU table
    smax: int  # bound of the SiLU table
    NX: int  # balanced limbs (signed bytes) of a residual-stream value: 4 or 5 (the line xl)
    router_clamp: bool  # the line rcl: the router's logits are clamped
    gate_clamp: bool  # the line gcl: the gate's keys are clamped
    eps1: int  # regularizers of a layer's first norm, of its second norm and of the final norm
    eps2: int
    epsF: int
    MF: int  # exponents of the final norm
    KF: int
    seed: int  # the seed from which a synthetic model's weights are generated
    layers: tuple[LayerScales, ...]
    # For each of tsilu, tsig, trope, texp: the eight words in the file.
    table_digests: dict[str, list[int]]
    # A logit is in units of 2^-s. The model reads s only for the noise of a seeded request
    # (noise.py); integrations/vllm/integer_model/examples/fit_limits.py prints it.
    s: int

    @property
    def dq(self) -> int:
        return self.n_h * self.dh

    @property
    def dkv(self) -> int:
        return self.n_g * self.dh

    def has_experts(self, layer: int) -> bool:
        """If NE > 0, the odd-numbered layers are mixture-of-experts layers."""
        return self.NE > 0 and layer % 2 == 1

    @classmethod
    def read(
        cls,
        parameter_file: str | Path,
        scales_file: str | Path | None = None,
        sha256: dict[str, str] | None = None,
    ) -> "Parameters":
        """Reads both files. scales_file defaults to KL.bin in the parameter file's directory.

        sha256, if given, is {"policy.txt": digest, "KL.bin": digest}, whatever the two files are
        called (see given_digests). Each file is read once, so the bytes that are hashed are the
        bytes that are parsed. Without sha256 the files are not verified.

        Accepts only version H821 (blocks/frame/version.h): format 8 over the field
        2^64 - 2^32 + 1.

        Stricter than the verifier:
        - only prof 20, the profile with attention;
        - the 32 lines with the public tables' digests are required, and with them NE > 0;
        - theta0 must be 10,000 or 500,000, dh <= 128 and lt <= 10 (otherwise the verifier ends at
          best with NOT-A-VERIFICATION: its condition `rotary-table`);
        - eps must be below 2^48 (the verifier, without the line xl: below 2^59);
        - with xl 5 and K <= 23 a norm needs K >= M - 13, as in the prover (the verifier: M - 14);
        - tau must be 128, as in the prover (the verifier: one of seven values, set by the line
          sec).

        Laxer than the verifier on the lines that only the proof reads. Of the lines on the proof
        system this reader checks the names, the order, the number of values and their spelling,
        and no value but the version and those of reading, rowcost, answers and listed. Of the other
        proof-only lines it checks names and order; the
        values of those in _ONLY_VALUE and _RANGE_IF_PRESENT and of nminH; and the number of call
        and callH lines. Of the rest it checks only that each value is a decimal below 2^32. Of a
        call line it checks only the grammar.

        Laxer also on other rules, for example: d = H dh is not checked, V may be up to 2^24 (the
        verifier: 2^23), and a product's shift up to 32 (the verifier: 16, 24 or 32, set by lines
        that this package does not use). So this reader accepts some files that the prover and the
        verifier reject (integrations/vllm/integer_model/OPERATING.md, "Verify what was loaded").
        """
        parameter_file = Path(parameter_file)
        scales_file = Path(scales_file) if scales_file else parameter_file.parent / "KL.bin"
        digests = given_digests(sha256, ("policy.txt", "KL.bin"))
        value = _read_lines(parameter_file, read_bytes(parameter_file, digests["policy.txt"]))

        def line(name: str, if_absent: int | None = None) -> int:
            return value[name] if if_absent is None else value.get(name, if_absent)

        def long_line(name: str) -> int:
            return line(name + "_lo") | line(name + "_hi") << 32

        layers = _read_scales(
            scales_file, read_bytes(scales_file, digests["KL.bin"]), line("L"), line("NE")
        )
        try:
            return cls(
                # The cap avoids a huge integer; __post_init__ rejects T > 2^10.
                T=1 << min(line("lt"), 32),
                L=line("L"),
                d=line("d"),
                n_h=line("H"),
                n_g=line("G"),
                dh=line("dh"),
                f=line("f"),
                fe=line("fe"),
                NE=line("NE"),
                V=line("V"),
                cs=line("cs"),
                theta0=long_line("theta0"),
                expS=line("S"),
                rx=line("rx"),
                eg=line("eg", 8),
                smax=line("smax", 4095),
                NX=line("xl", 4),
                router_clamp="rcl" in value,
                gate_clamp="gcl" in value,
                eps1=long_line("eps1"),
                eps2=long_line("eps2"),
                epsF=long_line("epsF"),
                MF=line("MF"),
                KF=line("KF"),
                seed=long_line("seed"),
                layers=layers,
                table_digests={
                    name: [line(f"{name}{i}") for i in range(8)] for name in _TABLE_DIGEST_LINES
                },
                s=line("s"),
            )
        except ParameterError as error:
            raise ParameterError(
                f"{error} (Read from {parameter_file} and {scales_file}.)"
            ) from None

    def __post_init__(self) -> None:
        """Rejects parameters that describe no model, or a model whose tables or tensors would have
        no reasonable size.

        It allows a norm with K > M, as the verifier does. Such a file is not in the standard
        parameter form, and no call of the verifier under it ends with YES
        (docs/model.md, "9 Range conditions"). Parameters.read lists how these rules differ from
        the verifier's.
        """

        def require(holds: bool, rule: str) -> None:
            if not holds:
                raise ParameterError(f"Invalid parameters: {rule}.")

        def power_of_two(value: int, low: int, high: int) -> bool:
            return low <= value <= high and value & (value - 1) == 0

        require(power_of_two(self.T, 1 << 4, 1 << 10), "lt must be in 4 .. 10")
        require(self.L >= 1, "L must be at least 1")
        require(power_of_two(self.dh, 2, 128), "dh must be a power of two from 2 to 128")
        require(
            power_of_two(self.n_g, 1, 64) and self.n_h >= 1,
            "G must be a power of two of at most 64, and H at least 1",
        )
        require(
            self.n_h % self.n_g == 0 and self.n_h + self.n_g <= 64,
            "G must divide H, and H + G must be at most 64",
        )
        require(1 <= self.d <= 1 << 20 and 1 <= self.f <= 1 << 20, "d and f must be in 1 .. 2^20")
        require(
            self.NE == 0 or power_of_two(self.NE, 2, 4096),
            "NE must be a power of two from 2 to 4,096",
        )
        require(
            self.NE == 0 or (self.f == 2 * self.fe and power_of_two(self.fe, 1, 1 << 19)),
            "fe must be a power of two, and f = 2 fe",
        )
        require(
            self.NE > 0 or not (self.router_clamp or self.gate_clamp),
            "rcl and gcl need a model with experts",
        )
        require(2 <= self.V <= 1 << 24, "V must be in 2 .. 2^24")
        require(self.cs == 12, "cs must be 12")
        require(self.theta0 in (10_000, 500_000), "theta0 must be 10,000 or 500,000")
        require(
            1 <= self.expS <= 14 and self.T << (self.expS + 1) <= 1 << 24,
            "S must be in 1 .. 14, and lt + S + 1 must be at most 24",
        )
        require(1 <= self.rx <= 12, "rx must be in 1 .. 12")
        require(4 <= self.eg <= 8, "eg must be in 4 .. 7 or be omitted")
        require(
            self.smax in (4095, 2047, 1023, 511, 255),
            "smax must be 2,047, 1,023, 511 or 255 or be omitted",
        )
        require(self.NX in (4, 5), "xl must be 5 or be omitted")
        require(
            len(self.layers) == self.L,
            "there must be one set of quantization scales for each layer",
        )

        def require_norm(M: int, K: int, eps: int, which: str) -> None:
            least = M - (13 if self.NX == 5 and K <= 23 else 14)
            require(
                24 <= M <= 47 and least <= K and 17 <= K <= 32,
                f"{which} needs 24 <= M <= 47, 17 <= K <= 32 and K >= M - 14 (M - 13 with xl 5 and "
                "K <= 23)",
            )
            require(1 <= eps < 1 << 48, f"{which} needs 1 <= eps < 2^48")

        require_norm(self.MF, self.KF, self.epsF, "the final norm")
        for layer, k in enumerate(self.layers):
            require_norm(k.M1, k.K1, self.eps1, f"the first norm of layer {layer}")
            require_norm(k.M2, k.K2, self.eps2, f"the second norm of layer {layer}")
            shifts = [k.kq, k.kk, k.kv, k.ko, k.kg, k.ku, k.kd]
            if self.has_experts(layer):
                shifts += [k.kr, k.kgR, k.kuR]
            require(
                all(1 <= shift <= 32 for shift in shifts),
                f"every product's shift in layer {layer} must be in 1 .. 32",
            )
            require(4 <= k.sh <= 12, f"sh in layer {layer} must be in 4 .. 12")
            require(k.rp in (0, 1), f"rp in layer {layer} must be 0 or 1")


def digests_from_file(path: object, folder: str | os.PathLike) -> object:
    """Reads the option "sha256_file": the absolute path of a file, outside the model directory
    folder, that holds what the option "sha256" would hold, as one JSON object. A caller passes
    that path in place of the digests, so that they appear on no command line and in no log.

    Raises ParameterError if the path is not absolute, leads into folder, or does not name a
    regular file of at most 64 KiB, not a symbolic link, that holds one JSON object, or if the
    file's mode lets its group or others read or write it (the digest of wsums.txt lets a guess at
    the weights be tested). given_digests checks the object. The file's owner is not looked at, and
    a hard link to a file in the model directory is not noticed: keep the file where only you can
    read or write.
    """
    where = 'The option "sha256_file" (with vLLM, an entry of --model-loader-extra-config)'
    if not isinstance(path, str) or not os.path.isabs(path):
        raise ParameterError(f"{where} must be the absolute path of a file.")
    inside = os.path.realpath(folder)
    if os.path.commonpath([os.path.realpath(path), inside]) == inside:
        raise ParameterError(f"{where}: {path} lies in the model directory; keep it outside.")
    try:
        handle = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            mode = os.fstat(handle).st_mode
            regular = stat.S_ISREG(mode)
            data = b""
            while regular and not mode & 0o077 and len(data) <= 65536:
                if not (piece := os.read(handle, 65537)):
                    break
                data += piece
        finally:
            os.close(handle)
    except OSError as error:
        raise ParameterError(f"{where}: cannot read {path}: {error.strerror}.") from None
    if not regular:
        raise ParameterError(f"{where}: {path} is not a regular file.")
    if mode & 0o077:
        raise ParameterError(
            f"{where}: {path} can be read or written by other accounts (mode {mode & 0o777:o}); "
            "make it yours alone: chmod 600 it."
        )
    if len(data) > 65536:
        raise ParameterError(f"{where}: {path} is larger than 64 KiB.")
    try:
        digests = json.loads(data.decode())
    except ValueError:
        digests = None
    if not isinstance(digests, dict):
        raise ParameterError(f"{where}: {path} does not hold one JSON object.")
    return digests


def given_digests(
    sha256: object, names: tuple[str, ...], with_gguf: tuple[str, ...] = ()
) -> dict[str, str | None]:
    """Validates the option "sha256" and returns it. If it is None, returns a dictionary with None
    under each of names.

    Raises ParameterError unless it is None or a dictionary whose keys are exactly names, or exactly
    names and with_gguf, and whose values are strings of 64 lower-case hexadecimal digits. with_gguf
    is for a caller that does not know yet whether the weights are in GGUF files.
    """
    if sha256 is None:
        return dict.fromkeys(names)
    if (
        not isinstance(sha256, dict)
        or set(sha256) not in (set(names), set(names + with_gguf))
        or not all(
            isinstance(digest, str) and re.fullmatch("[0-9a-f]{64}", digest)
            for digest in sha256.values()
        )
    ):
        raise ParameterError(
            'The option "sha256", or the file that the option "sha256_file" names (with vLLM, '
            "entries of --model-loader-extra-config), must be a JSON object with exactly these "
            "entries: "
            + ", ".join(f'"{name}"' for name in names)
            + (
                ", and for weights in GGUF files also "
                + ", ".join(f'"{name}"' for name in with_gguf)
                if with_gguf
                else ""
            )
            + ". Each is that file's SHA-256 as 64 lower-case hexadecimal digits, as sha256sum "
            "prints it."
        )
    return sha256


def read_bytes(file: Path, sha256: str | None) -> bytes:
    """Reads the file once and, if sha256 is given, raises ParameterError unless the bytes have that
    digest.

    The error names neither digest, on purpose: logs are read more widely than these files, the path
    may be a symlink to any readable file, and the digest of wsums.txt (the weights' checksums)
    reveals something about the weights.
    """
    data = file.read_bytes()
    if sha256 is not None and hashlib.sha256(data).hexdigest() != sha256:
        raise ParameterError(
            f"{file} is not the file whose SHA-256 was given, so the model was not loaded. The "
            'digest came from the option "sha256", or from the file that the option "sha256_file" '
            "names (with vLLM, entries of --model-loader-extra-config). Compare the file with the "
            "copy that the digest was taken from."
        )
    return data


def _read_lines(parameter_file: Path, data: bytes) -> dict[str, int]:
    """The value lines of the parameter file as a dictionary, once the whole file has passed the
    rules on grammar and order."""

    def reject(number: int, fault: str) -> NoReturn:
        raise ParameterError(f"{parameter_file}, line {number}: {fault}")

    try:
        text = data.decode("ascii")
    except UnicodeDecodeError:
        raise ParameterError(f"{parameter_file} is not an ASCII text file.") from None
    if not text.endswith("\n"):
        raise ParameterError(f"{parameter_file}: the last line does not end with a newline.")
    lines = text[:-1].split("\n")

    if lines[0] not in [f"version {version}" for version in _VERSIONS]:
        reject(
            1,
            f'expected "version {_VERSIONS[0]}". This package reads no other format.',
        )
    for number, (name, count) in enumerate(_PROOF_SYSTEM_LINES, start=2):
        words = lines[number - 1].split(" ") if number <= len(lines) else []
        values = words[len(name.split(" ")) :]
        if (
            " ".join(words[: len(words) - len(values)]) != name
            or not all(_PROOF_SYSTEM_VALUE.fullmatch(word) for word in values)
            or (len(values) != count if count else len(values) not in range(4, 33, 4))
        ):
            how_many = {None: "4, 8, .. or 32 decimal values", 1: "1 decimal value"}.get(
                count, f"{count} decimal values"
            )
            reject(number, f'expected "{name}" and {how_many}, with one space between them.')
        if name in _ONLY_VALUE and int(values[0]) != _ONLY_VALUE[name]:
            reject(number, f"{name} is {values[0]}, expected {_ONLY_VALUE[name]}.")
    first = len(_PROOF_SYSTEM_LINES) + 2  # the number of the line prof

    value: dict[str, int] = {}
    expected = list(_LINE_ORDER)
    for number, line in enumerate(lines[first - 1 :], start=first):
        if line.startswith("call"):
            break
        match = _VALUE_LINE.fullmatch(line)
        if match is None or int(match[2]) >= 1 << 32:
            reject(number, 'expected "name value", with one space and a decimal value below 2^32.')
        name = match[1]
        skipped = []  # the lines that may be absent here
        while True:
            wanted = expected.pop(0) if expected else "call"
            if wanted.rstrip("?") == name:
                break
            if wanted.endswith("?"):
                skipped.append(wanted.rstrip("?"))
                continue
            what = "a call line" if wanted == "call" else f"the line {wanted}"
            if skipped:
                names = ", ".join(skipped)
                what += f" (or before it, in this order, any of the optional lines {names})"
            hint = ""
            if name == "nminH":
                hint = _PLACE_OF_NMINH
            elif name.rstrip("0123456789") in _TABLE_DIGEST_LINES:
                hint = " Each table's digest has 8 lines."
            reject(number, f"expected {what}, found {name}.{hint}")
        value[name] = int(match[2])
        # Checked first, because a file of another profile also has other lines.
        if name == "prof" and value[name] != _ONLY_VALUE[name]:
            reject(
                number,
                f"prof is {value[name]}, expected {_ONLY_VALUE[name]}, the profile with attention.",
            )
        if name == "NE" and value[name] == 0:
            reject(number, _NO_EXPERTS)
        if name == "lidb":
            expected.insert(0, "nminH?")
    missing = [wanted for wanted in expected if not wanted.endswith("?")]
    if missing:
        hint = _NO_DIGESTS if missing[0] == "tsilu0" else ""
        reject(first + len(value), f"expected the line {missing[0]}.{hint}")

    kinds = []
    for number, line in enumerate(lines[first - 1 + len(value) :], start=first + len(value)):
        match = _CALL_LINE.fullmatch(line)
        if match is None or len(line) > 1022:
            reject(
                number,
                "expected a call line: call, callA or callH, then at least three names, with one "
                "space between them, in at most 1,022 characters.",
            )
        kinds.append(match[1])
    if kinds != sorted(kinds) or "call" not in kinds or "callA" not in kinds:
        raise ParameterError(
            f"{parameter_file} must end with at least one call line, then at least one callA "
            "line, then any callH lines."
        )
    for kind, most in _MOST_CALL_LINES.items():
        if kinds.count(kind) > most:
            raise ParameterError(
                f"{parameter_file} has {kinds.count(kind)} {kind} lines. It may have at most "
                f"{most}."
            )

    for name, only in _ONLY_VALUE.items():
        if name in value and value[name] != only:
            raise ParameterError(f"{parameter_file}: {name} is {value[name]}, expected {only}.")
    if value.get("eg") == 8 or value.get("smax") == 4095:
        raise ParameterError(
            f"{parameter_file}: eg 8 and smax 4095 are the defaults and must be omitted."
        )
    optional = _RANGE_IF_PRESENT | {"nminH": (max(value["nmin"] + 1, 2), 28)}
    for name, (least, largest) in optional.items():
        if name in value and not least <= value[name] <= largest:
            default = value["nmin"] if name == "nminH" else least - 1
            raise ParameterError(
                f"{parameter_file}: {name} is {value[name]}, expected {least} .. {largest}."
                f" For the value {default}, omit the line."
            )
    return value


def _read_scales(scales_file: Path, data: bytes, L: int, NE: int) -> tuple[LayerScales, ...]:
    """Reads 13 little-endian 32-bit words per layer, then 3 per mixture-of-experts layer in the
    layers' order."""
    expert_layers = L // 2 if NE > 0 else 0
    if len(data) != 4 * (13 * L + 3 * expert_layers):
        raise ParameterError(
            f"{scales_file} must hold 13 words for each of the {L} layers and 3 more for each of "
            f"the {expert_layers} mixture-of-experts layers, 4 bytes each. It holds {len(data)} "
            "bytes."
        )
    word = struct.unpack(f"<{len(data) // 4}I", data)
    layers = []
    for layer in range(L):
        extra = word[13 * L + 3 * (layer // 2) :][:3] if NE > 0 and layer % 2 == 1 else ()
        layers.append(LayerScales(*word[13 * layer : 13 * layer + 13], *extra))
    return tuple(layers)
