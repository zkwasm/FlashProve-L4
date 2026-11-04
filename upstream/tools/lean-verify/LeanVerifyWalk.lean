-- Copyright 2026 Anthropic PBC
-- SPDX-License-Identifier: Apache-2.0

/-
tools/lean-verify/LeanVerifyWalk.lean: works out anew what tools/lean-verify/native-modules.txt
should hold: the names of the modules whose compiled code goes into the native helper library of
`tools/lean-verify/lean_verify.sh`. The script `tools/lean-verify/check_list.sh` calls it, and
compares what it writes with that list's own lines. It is no part of the Lean library, and it judges
no proof. Paths on this page count from the repository's top directory.

What it does. (1) It starts at `PV.Program.run` (lean/Johnson/PVR/Program.lean) and walks the
compiler's own records of compiled code (Lean's intermediate representation): from each declaration
to every name that its code mentions. (2) It takes the modules of this library in which these names
lie. (3) It strikes each module that is one of `leftOut`, or imports one, directly or indirectly, by
Lean's own record of imports. (4) It takes the rest and every module that they import, directly or
indirectly, except Lean's own (Init, Std, Lean, Lake), whose code the program `lean` has already.
(5) It writes these modules' names, one to a line, sorted by their bytes, to a new file, whose path
the environment variable LEAN_VERIFY_LIST names, and prints the counts. It overwrites nothing: if
something lies at that path already, it stops.

Why seven modules are left out. A compiled Lean program first runs the start-up code of every module
that it imports, and that code evaluates every definition that takes no argument. For each of the
last six names of `leftOut` we built a program that imports that module and prints one line: each
ran out of memory (24 GB) before it printed (one trial each, on one machine). We take it that the
same code runs when `lean` loads a library that holds such a module; that has not been tried. The
first name, OnBytes.Main, declares a `main`; every helper library made for lean_verify.sh has left
it out, and that script has not been tried with one that holds it. That these
seven are enough is known by trial, not by proof: runs with the helper library have ended with
their verdicts.

What follows for a run. The code of a struck module is not in the helper library: Lean interprets
it. A wrong list can cost time, or cost a run its verdict. By Lean's design compiled and interpreted
code give one answer, so a wrong list should not change a verdict; nothing here proves that.

What the result depends on, beside the sources: Lean's version (what its compiler inlines), the
compiled files as they lie, `leftOut`, and the directory in which Lean is started: a module counts
as this library's if its source file lies below that directory, so start it from lean/. A name that
the compiler made itself and that Lean's record assigns to no module is passed over.
-/
import Johnson.PVR.Program

open Lean

namespace LeanVerifyWalk

/-- Modules that the helper library must neither hold nor import. The comment at the top of this
file says why. -/
def leftOut : List Name :=
  [`OnBytes.Main, `Inst.Model.Params.Setup, `OnBytes.Slice.LinkAt, `OnBytes.Slice.SimSample,
   `Code.FieldEnum, `Inst.R0.Numbering, `Cb.GlVal.GlGuardTower]

/-- The roots, and every name that the compiled code of a name already found mentions. A name
without compiled code of its own is kept too; nothing is followed from it. -/
def reached (env : Environment) (roots : List Name) : NameSet := Id.run do
  let mut seen : NameSet := {}
  let mut todo : Array Name := roots.toArray
  while !todo.isEmpty do
    let n := todo.back!
    todo := todo.pop
    if seen.contains n then continue
    seen := seen.insert n
    if let some d := IR.findEnvDecl env n then
      for m in IR.collectUsedDecls env [d] do
        if !seen.contains m then todo := todo.push m
  return seen

end LeanVerifyWalk

open LeanVerifyWalk in
#eval show CoreM Unit from do
  let some out ← IO.getEnv "LEAN_VERIFY_LIST"
    | throwError "LEAN_VERIFY_LIST is not set: set it to the path of a new file, to which the \
        list of modules should go, or run tools/lean-verify/check_list.sh"
  if ← (System.FilePath.mk out).pathExists then
    throwError "LEAN_VERIFY_LIST names a path at which something lies already: name a new file"
  let env ← getEnv
  let names := env.header.moduleNames
  let data := env.header.moduleData
  -- Lean lists a module behind all that it imports, so one pass from the front is enough. An
  -- import that has not been seen yet would break that order: it counts as left out.
  let mut place : Std.HashMap Name Nat := {}
  let mut bad : Array Bool := #[]
  for i in [0:names.size] do
    let b := leftOut.contains names[i]! ||
      data[i]!.imports.any fun im => match place.get? im.module with
        | some j => bad[j]!
        | none => true
    place := place.insert names[i]! i
    bad := bad.push b
  let seen := reached env [``PV.Program.run]
  let mut code : NameSet := {}
  for n in seen.toList do
    if let some i := env.getModuleIdxFor? n then
      let m := names[i.toNat]!
      if ← (System.FilePath.mk ((m.toString.replace "." "/") ++ ".lean")).pathExists then
        code := code.insert m
  let roots := code.toList.filter fun m => (place.get? m).any fun j => !bad[j]!
  -- from the back: whatever a wanted module imports is wanted
  let mut want : Array Bool := names.map fun m => roots.contains m
  for k in [0:names.size] do
    let i := names.size - 1 - k
    if want[i]! then
      for im in data[i]!.imports do
        if let some j := place.get? im.module then want := want.set! j true
  let mut keep : Array String := #[]
  for i in [0:names.size] do
    if want[i]! && !([`Init, `Std, `Lean, `Lake].contains names[i]!.getRoot) then
      keep := keep.push names[i]!.toString
  IO.FS.writeFile out (String.intercalate "\n" (keep.qsort (· < ·)).toList ++ "\n")
  IO.println s!"names reached from PV.Program.run: {seen.size}; modules of this library that hold \
    them: {code.size}; of these, not importing a module that is left out on purpose: \
    {roots.length}; with everything that they import, except Lean's own: {keep.size}; the list \
    is written to {out}"
