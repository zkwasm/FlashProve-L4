# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""Run integrations/check_proof.py with the real cp_verify on the examples.

In integrations/tests/, test_check_proof.py and test_round_trip.py use fake_cp_verify.py, a fake
verifier that checks nothing in a proof. The tests here check that the script and cp_verify agree
on a pass, under exact match and a clipped-gap policy, and on a rejection. Each is skipped if
bin/cp_verify or its example's proof is missing.
"""

import json
import os
import pathlib
import shutil
import struct
import subprocess
import sys

import proof_middleware
import pytest
import spool

REPOSITORY = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = REPOSITORY / "integrations" / "check_proof.py"
VERIFIER = REPOSITORY / "bin" / "cp_verify"
EXAMPLES = REPOSITORY / "examples"

pytestmark = pytest.mark.skipif(
    not VERIFIER.is_file(), reason="bin/cp_verify is not built (run verifier/build.sh)"
)


def check(example, tmp_path, proof=None, timeout=600):
    """Check `proof`, by default the example's proof, against the example's values.

    Return the exit code and the last two output lines.
    """
    if not (example / "example.proof").is_file():
        pytest.skip("%s has no files" % example)
    (tmp_path / "tmp").mkdir(exist_ok=True)
    request = (example / "example.req").read_bytes()
    start, end = struct.unpack("<2I", request[-8:])
    words = [sys.executable, "-B", str(SCRIPT), str(proof or example / "example.proof")]
    words += [str(example / "example.req"), "--public-key-dir", str(example / "ivk")]
    words += ["--parameter-file", str(example / "config" / "policy.txt")]
    words += ["--scales-file", str(example / "config" / "KL.bin")]
    words += ["--commitment-id", (example / "ID.txt").read_text().strip()]
    words += ["--range", "%d:%d" % (start, end), "--end-token", "none"]
    if end == len(request) // 4 - 2:
        words += ["--next-token", "%d" % struct.unpack("<I", (example / "example.y").read_bytes())]
    pairs = [line.split() for line in (example / "config" / "policy.txt").read_text().splitlines()]
    values = dict(pair for pair in pairs if len(pair) == 2)
    if (values["C"], values["theta"], values["cap"]) != ("256", "256", "0"):
        # Not exact match. A test may take the limits from the file; a client must not.
        words += ["--policy", "clipped-gap"]
        for name in ("C", "theta", "cap", "s"):
            words += ["--" + name, values[name]]
    environment = {"PATH": "/usr/bin:/bin", "TMPDIR": str(tmp_path / "tmp")}
    done = subprocess.run(words, capture_output=True, text=True, env=environment, timeout=timeout)
    return done.returncode, *done.stdout.splitlines()[-2:]


def changed_proof(example, tmp_path):
    """Return the path of a copy of the example's proof with one bit changed."""
    for name in ("example.proof", "example.y"):
        shutil.copyfile(example / name, tmp_path / name)  # not the mode: it may be read-only
    data = bytearray((tmp_path / "example.proof").read_bytes())
    data[len(data) // 2] ^= 1
    (tmp_path / "example.proof").write_bytes(data)
    return tmp_path / "example.proof"


def test_small_example_is_verified(tmp_path):
    code, result, last_line = check(EXAMPLES / "accepted-proof", tmp_path)
    assert code == 0, last_line
    assert result.startswith("result: outcome=verified ")
    assert last_line.startswith("check_proof: verified: ")


def test_small_clipped_gap_example_passes_within_its_own_limits(tmp_path):
    code, result, last_line = check(EXAMPLES / "accepted-proof-clipped-gap", tmp_path)
    assert code == 0, last_line
    assert result.startswith("result: outcome=within-limits ") and " claim=policy " in result
    assert last_line.startswith("check_proof: within limits: clipped gap: ")


@pytest.mark.parametrize("name", ["accepted-proof", "accepted-proof-clipped-gap"])
def test_changed_proof_is_rejected(tmp_path, name):
    if not (EXAMPLES / name / "example.proof").is_file():
        pytest.skip("examples/%s has no files" % name)
    proof = changed_proof(EXAMPLES / name, tmp_path)
    code, result, last_line = check(EXAMPLES / name, tmp_path, proof)
    assert (code, result) == (1, "result: outcome=rejected")
    assert last_line.startswith("check_proof: rejected: cp_verify printed NO: ")


def test_full_size_example_is_verified(tmp_path):
    """This example needs more time and memory than the small ones."""
    folder = EXAMPLES / "full-size"
    # The example's page has this sentence if the release ships its three large files as a separate
    # archive and not in the repository.
    sentence = "Three files of this example are not in the repository"
    page = folder / "README.md"
    if not (folder / "example.proof").is_file() and page.is_file() and sentence in page.read_text():
        pytest.skip("full-size: not run: the set is not in this tree")
    code, result, last_line = check(folder, tmp_path, timeout=3600)
    assert code == 0, last_line
    assert result.startswith("result: outcome=verified ")


def test_full_size_clipped_gap_example_passes_within_its_own_limits(tmp_path):
    """Like the full-size example, this one needs more time and memory than the small ones."""
    folder = EXAMPLES / "full-size-clipped-gap"
    if not (folder / "example.proof").is_file():
        pytest.skip("full-size-clipped-gap: not run: the set is not in this tree")
    code, result, last_line = check(folder, tmp_path, timeout=3600)
    assert code == 0, last_line
    assert result.startswith("result: outcome=within-limits ") and " claim=policy " in result
    assert last_line.startswith("check_proof: within limits: clipped gap: ")
    assert "; the next token, " in last_line  # the stored request also covers the token after it


# ---- The middleware's check of a batch, with the real cp_verify ----


def spool_with_the_example(example, tmp_path, proof):
    """Lay the example's files, and one job whose response in out/ has this proof.

    Return the middleware, its proof ID and the spool directory."""
    if not (example / "example.proof").is_file():
        pytest.skip("%s has no files" % example)
    tmp_path.chmod(0o700)
    shutil.copytree(example / "config", tmp_path / "config")
    shutil.copytree(example / "ivk", tmp_path / "ivk")
    shutil.copyfile(VERIFIER, tmp_path / "cp_verify")
    (tmp_path / "cp_verify").chmod(0o755)
    for path in (tmp_path / "config", tmp_path / "ivk"):
        path.chmod(0o755)
    entry = {
        "served_model": "example",
        "commitment_id": (example / "ID.txt").read_text().strip(),
        "parameter_file": str(tmp_path / "config" / "policy.txt"),
        "spool_dir": str(tmp_path / "spool"),
        "public_key_dir": str(tmp_path / "ivk"),
        "verifier": str(tmp_path / "cp_verify"),
    }
    config = {"jobs_dir": str(tmp_path / "jobs"), "policies": {"example": entry}}
    (tmp_path / "middleware.json").write_text(json.dumps(config))
    middleware = proof_middleware.ProofMiddleware(None, config=str(tmp_path / "middleware.json"))
    assert middleware.fault is None, middleware.fault
    request = (example / "example.req").read_bytes()
    request_id, key, y = "ab" * 32, bytes(range(32)), (example / "example.y").read_bytes()
    out = tmp_path / "spool" / "out"
    out.mkdir(mode=0o700, exist_ok=True)
    for suffix, data in (
        ("y", y),
        ("tag", spool.response_tag(key, request_id, y, proof)),
        ("proof", proof),
    ):
        path = out / ("%s.%s" % (request_id, suffix))
        path.write_bytes(data)
        os.utime(path, ns=(10**18, 10**18))  # a response's files have one modification time
    job = {
        "policy": "example",
        "commitment_id": entry["commitment_id"],
        "request_id": request_id,
        "tag_key": key.hex(),
        "request": {},
        "request_bytes": request.hex(),
    }
    proof_id = "cd" * 32
    (tmp_path / "jobs" / proof_middleware.job_file(proof_id)).write_text(json.dumps(job))
    return middleware, proof_id, tmp_path / "spool"


@pytest.mark.parametrize("name", ["accepted-proof", "accepted-proof-clipped-gap"])
def test_the_middleware_hands_on_a_proof_that_passes(tmp_path, monkeypatch, name):
    monkeypatch.setattr(proof_middleware, "WAIT_S", 600)
    proof = (EXAMPLES / name / "example.proof")
    data = proof.read_bytes() if proof.is_file() else b""
    middleware, proof_id, spool_dir = spool_with_the_example(EXAMPLES / name, tmp_path, data)
    assert middleware.job_state(proof_id)[0]["state"] == "proved"
    assert not (spool_dir / "hold").exists()


@pytest.mark.parametrize("name", ["accepted-proof", "accepted-proof-clipped-gap"])
def test_the_middleware_holds_back_a_proof_that_cp_verify_rejects(tmp_path, monkeypatch, name):
    monkeypatch.setattr(proof_middleware, "WAIT_S", 600)
    proof = (EXAMPLES / name / "example.proof")
    data = bytearray(proof.read_bytes() if proof.is_file() else b"x")
    data[len(data) // 2] ^= 1  # one bit of the proof: see examples/rejected-proofs/
    middleware, proof_id, spool_dir = spool_with_the_example(EXAMPLES / name, tmp_path, bytes(data))
    assert middleware.job_state(proof_id)[0]["state"] == "waiting"
    assert (spool_dir / "hold").is_file()
    assert sorted(path.suffix for path in (spool_dir / "out").iterdir()) == [".failed", ".tag"]
    (spool_dir / "hold").unlink()
    assert middleware.job_state(proof_id)[0]["state"] == "no_proof"
