-- Copyright 2026 Anthropic PBC
-- SPDX-License-Identifier: Apache-2.0

/-
tools/lean-verify/LeanVerify.lean: the entry of the command `tools/lean-verify/lean_verify.sh`: one
call of `PV.Program.run` (lean/Johnson/PVR/Program.lean). Call the script, not this file.

It is no part of the Lean library: no build compiles it, and no check of the library reads it. It is
a short unproved program: it hands the arguments over and turns the exit code into Lean's status.

Reads: the environment variable LEAN_VERIFY_ARGS: the seven arguments of `PV.Program.usage`, one to
a line. Why not a `main`: the library's module OnBytes.Main, which lies beneath the verifier's
files, has declared that name.

It ends Lean with the status 10 for accept, 11 for reject, 12 for no verdict: never 0, with which
Lean ends whenever a file merely elaborates. Any other status means that the call did not end as it
should. The lines of a call go to the error stream: inside `#eval` Lean holds standard output back.
-/
import Johnson.PVR.Program

set_option stderrAsMessages false

#eval show IO Unit from do
  let args := ((← IO.getEnv "LEAN_VERIFY_ARGS").getD "").splitOn "\n"
  let code ← PV.Program.run args
  IO.Process.exit (match code with | 0 => 10 | 1 => 11 | _ => 12)
