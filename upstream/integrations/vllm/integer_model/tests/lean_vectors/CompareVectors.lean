-- Copyright 2026 Anthropic PBC
-- SPDX-License-Identifier: Apache-2.0

import Lean.Data.Json
import IntegerModel.Model
import VerifiableInference.ModelWeights

/-!
# Compares lean/IntegerModel/Model.lean with test vectors from the Python package

    compare_vectors OUT

`OUT` is a directory of test vectors; README.md beside this file describes its layout. For every
case below `OUT/cases` this program reads the settings, the tables, the weights and the prompts,
evaluates the definitions of Model.lean on them, and compares the outcome and every listed value
with what the Python package reported. It prints one line for each run, and one for each array that
differs, with the file, the index and both values. The exit status is 0 if nothing differs, 1 if
something differs, and 2 if the files cannot be read or hold no case and no function to compare.
run.py beside this file builds and runs the program.

Where tables.json says so, its tables are also compared, entry for entry, with the five tables of
the library's theorems (`lookupTables`, lean/VerifiableInference/ModelWeights.lean). On each file
below `OUT/settings` the library's test `settingsAreValid` is evaluated. On each case's settings the
library's tests of the quantization parameters are evaluated, and the cases that pass are counted.

A case that holds the whole model is evaluated with `model` and `trace`. A case in which matrices
are left out is evaluated function by function: see `compareByFunction`.

This is a test program. It is no part of the Lean library: nothing below lean/ imports it, and it
states no theorem but one `example` on a test of its own. What runs is compiled Lean code, not an
evaluation by Lean's kernel, and the comparison proves nothing.
-/


open Lean IntegerTransformer
open System (FilePath)

namespace CompareVectors

def fail {α : Type} (message : String) : IO α :=
  throw (IO.userError message)

/-! ## Reading the files -/

/-- One entry of a list "arrays" in case.json, tables.json or function.json. -/
structure Listed where
  name : String
  symbol : String
  layer : Option Nat
  shape : Array Nat
  type : String
  file : FilePath
  lowestKey : Option Int
  rowsAreTokens : Option (Array Nat)

/-- The list "arrays" of a JSON object. The files' paths are made to begin with `folder`. -/
def listedArrays (folder : FilePath) (j : Json) : Except String (Array Listed) := do
  (← j.getObjValAs? (Array Json) "arrays").mapM fun a => do
    let name ← a.getObjValAs? String "name"
    return {
      name
      symbol := (a.getObjValAs? String "symbol").toOption.getD name
      layer := (a.getObjValAs? Nat "layer").toOption
      shape := ← a.getObjValAs? (Array Nat) "shape"
      type := ← a.getObjValAs? String "type"
      file := folder / (← a.getObjValAs? String "file")
      lowestKey := (a.getObjValAs? Int "lowest_key").toOption
      rowsAreTokens := (a.getObjValAs? (Array Nat) "rows_are_tokens").toOption }

def readJson (path : FilePath) : IO Json := do
  IO.ofExcept (Json.parse (← IO.FS.readFile path))

def find (arrays : Array Listed) (name : String) : IO Listed :=
  match arrays.find? (·.name == name) with
  | some a => pure a
  | none => fail s!"no array {name} is listed"

/-- Entry `i` of a raw file of little-endian two's complement numbers of `width` bytes. -/
@[inline] def entryAt (bytes : ByteArray) (width i : Nat) : Int := Id.run do
  let mut unsigned : Nat := 0
  for b in [0:width] do
    unsigned := unsigned + (bytes.get! (i * width + b)).toNat <<< (8 * b)
  return if unsigned < 2 ^ (8 * width - 1) then unsigned else (unsigned : Int) - 2 ^ (8 * width)

/-- A file's bytes and the width of one number. The file must have the size that its listed shape
and type ask for. -/
def readBytes (a : Listed) : IO (ByteArray × Nat) := do
  let width ← match a.type with
    | "int16" => pure 2
    | "int32" => pure 4
    | "int64" => pure 8
    | other => fail s!"{a.file}: unknown type {other}"
  let bytes ← IO.FS.readBinFile a.file
  let wanted := a.shape.foldl (· * ·) width
  if bytes.size ≠ wanted then
    fail s!"{a.file}: {bytes.size} bytes, but {a.type} of shape {a.shape.toList} has {wanted}"
  return (bytes, width)

/-- A file's numbers in the file's order. -/
def readFlat (a : Listed) : IO IntegerVector := do
  let (bytes, width) ← readBytes a
  return tabulate (bytes.size / width) fun i => entryAt bytes width i

/-- A file of shape [rows, columns] as the array of its rows. -/
def readMatrix (a : Listed) : IO Matrix := do
  let #[rows, columns] := a.shape
    | fail s!"{a.file}: a matrix has two sizes, but the shape is {a.shape.toList}"
  let (bytes, width) ← readBytes a
  return tabulate rows fun r => tabulate columns fun c => entryAt bytes width (r * columns + c)

/-! ## Settings, tables and weights -/

def layerSettings (top j : Json) : Except String LayerSettings := do
  let n (key : String) := j.getObjValAs? Nat key
  return {
    attentionBits := ← n "M1", attentionShift := ← n "K1"
    attentionEps := ← top.getObjValAs? Int "eps1"
    queryShift := ← n "kq", keyShift := ← n "kk", valueShift := ← n "kv"
    rotary := (← n "rp") != 0, softmaxShift := ← n "sh", outputShift := ← n "ko"
    feedForwardBits := ← n "M2", feedForwardShift := ← n "K2"
    feedForwardEps := ← top.getObjValAs? Int "eps2"
    gateShift := ← n "kg", upShift := ← n "ku", downShift := ← n "kd"
    routerShift := ← n "kr", routedGateShift := ← n "kgR", routedUpShift := ← n "kuR" }

def settings (j : Json) : Except String Settings := do
  let n (key : String) := j.getObjValAs? Nat key
  return {
    positions := ← n "T", layers := ← n "L", width := ← n "d"
    queryHeads := ← n "n_h", keyValueHeads := ← n "n_g", headDim := ← n "dh"
    hidden := ← n "f", expertHidden := ← n "fe", experts := ← n "NE", vocabulary := ← n "V"
    rotaryScale := ← n "cs"
    finalEps := ← j.getObjValAs? Int "epsF", finalBits := ← n "MF", finalShift := ← n "KF"
    perLayer := ← (← j.getObjValAs? (Array Json) "layers").mapM (layerSettings j) }

def readTables (arrays : Array Listed) : IO Tables := do
  let keyed (name : String) : IO KeyedTable := do
    let a ← find arrays name
    let some lowestKey := a.lowestKey | fail s!"{a.file}: no lowest_key is listed"
    return { lowestKey, entries := ← readFlat a }
  return {
    cosine := ← readMatrix (← find arrays "cosine")
    sine := ← readMatrix (← find arrays "sine")
    exponential := ← readFlat (← find arrays "exponential")
    silu := ← keyed "silu"
    sigmoid := ← keyed "sigmoid" }

/-- One case: what case.json says of the model, and the tables that it names. -/
structure Case where
  name : String
  settings : Settings
  tables : Tables
  arrays : Array Listed
  /-- Is every matrix written? If not, a matrix that is not listed is read as an empty array. -/
  whole : Bool

def Case.matrix (c : Case) (name : String) : IO Matrix :=
  match c.arrays.find? (·.name == name) with
  | some a => readMatrix a
  | none => if c.whole then fail s!"{c.name}: no array {name} is listed" else pure #[]

/-- The weights of layer `l`. A matrix whose name `wanted` turns down is not read and stays empty.
-/
def Case.layerWeights (c : Case) (l : Nat) (wanted : String → Bool := fun _ => true) :
    IO LayerWeights := do
  let m (name : String) : IO Matrix := if wanted name then c.matrix s!"L{l}.{name}" else pure #[]
  let mixture := c.settings.isMixture l
  let routed : Array GatedWeights ← if mixture then
      (Array.range c.settings.experts).mapM fun e => do
        return { gate := ← m s!"E{e}.Wgr", up := ← m s!"E{e}.Wur", down := ← m s!"E{e}.Wdr" }
    else pure #[]
  return {
    query := ← m "Wq", key := ← m "Wk", value := ← m "Wv", output := ← m "Wo"
    shared := { gate := ← m "Wg", up := ← m "Wu", down := ← m "Wd" }
    router := ← if mixture then m "Wr" else pure #[]
    routed }

/-- The embedding, V rows. Where only some tokens' rows are written, the other rows are empty. -/
def Case.embedding (c : Case) : IO Matrix := do
  let a ← find c.arrays "Emb"
  let rows ← readMatrix a
  match a.rowsAreTokens with
  | none => return rows
  | some tokens =>
    if tokens.size ≠ rows.size then
      fail s!"{a.file}: {rows.size} rows, but rows_are_tokens lists {tokens.size} tokens"
    let mut all : Matrix := Array.replicate c.settings.vocabulary #[]
    for token in tokens, row in rows do
      all := all.set! token row
    return all

/-! ## From a name in the files to a value of Model.lean -/

/-- `v` followed by zeros, `n` entries in all. -/
def padded (v : IntegerVector) (n : Nat) : IntegerVector :=
  tabulate n fun k => v.entry k

/-- A norm's value, by the ending of its name. The sum of squares can pass 64 bits, so the files
would have it as two numbers: Vx = high 2^40 + low with 0 ≤ low < 2^40. -/
def normValue (n : Norm) (ending : String) : Option IntegerVector :=
  match ending with
  | "_Vx_high" => some #[n.sumSquares / 2 ^ 40]
  | "_Vx_low" => some #[n.sumSquares % 2 ^ 40]
  | "_sigma" => some #[n.root]
  | "_Rc" => some #[n.reciprocal]
  | "_w" => some n.quotient
  | "" => some n.out
  | _ => none

/-- The value that the files call `symbol`, in one layer at position `t`, as one row. Values of
hidden units: the shared block's, then the routed expert's (none in a dense layer). Values by head:
head after head, and those over the positions j are filled with zeros up to T entries. The package
reports 21 of these names in a dense layer and 25 in a layer with experts; the others are here so
that every field of `Layer` has a name. -/
def layerValue (s : Settings) (L : Layer) (symbol : String) (t : Nat) : Option IntegerVector :=
  let p := L.attention.projected.entry t
  let ff := L.feedForward.entry t
  let units (f : HiddenUnits → IntegerVector) : Option IntegerVector := some (f ff.shared ++ f ff.routed.units)
  let byHead (f : Head → IntegerVector) : Option IntegerVector := some ((L.attention.heads.entry t).flatMap f)
  match symbol with
  | "x" => some (L.input.entry t)
  | "Aq" => some p.queryProducts
  | "Ak" => some p.keyProducts
  | "pq" => some p.rotatedQuery
  | "pk" => some p.rotatedKey
  | "qw" => some p.wideQuery
  | "q" => some p.query
  | "kw" => some p.wideKey
  | "k" => some p.key
  | "vw" => some p.wideValue
  | "v" => some p.value
  | "s_ij" => byHead fun h => padded h.scores s.positions
  | "m_i" => byHead fun h => #[h.top]
  | "u_ij" => byHead fun h => padded h.distances s.positions
  | "e_ij" => byHead fun h => padded h.exponentials s.positions
  | "Z_i" => byHead fun h => #[h.total]
  | "acc" => byHead fun h => h.mix
  | "o" => some (L.attention.mixed.entry t)
  | "ad_attention" => some (L.attention.addition.entry t)
  | "x_mid" => some (L.attention.out.entry t)
  | "lw" => some ff.routed.wide
  | "ly" => some ff.routed.high
  | "l3" => some ff.routed.low
  | "lr" => some ff.routed.logits
  | "e_star" => some #[ff.routed.chosen]
  | "sc" => some #[ff.routed.scale]
  | "xr" => some ff.routed.input
  | "gw" => units (·.wideGate)
  | "g0" => units (·.lowByte)
  | "gh" => units (·.highPart)
  | "gc" => units (·.clampedHigh)
  | "key" => units (·.key)
  | "s" => units (·.activation)
  | "u" => units (·.up)
  | "ad_feed_forward" => some ff.addition
  | other =>
    if other.startsWith "a" then normValue (L.attention.norm.entry t) (other.drop 1).toString
    else if other.startsWith "m" then normValue ff.norm (other.drop 1).toString
    else none

/-- The values behind the layers. -/
structure Top where
  last : Matrix
  finalNorm : Array Norm
  logits : Matrix
  choices : Array Nat

def topValue (top : Top) (symbol : String) (t : Nat) : Option IntegerVector :=
  match symbol with
  | "x_last" => some (top.last.entry t)
  | "lam" => some (top.logits.entry t)
  | "first_choice" => some #[top.choices.entry t]
  | other =>
    if other.startsWith "aF" then normValue (top.finalNorm.entry t) (other.drop 2).toString
    else none

/-! ## Comparing -/

/-- What has been compared so far. -/
structure Count where
  cases : Nat := 0
  runs : Nat := 0
  runsThatDiffer : Nat := 0
  arrays : Nat := 0
  values : Nat := 0
  /-- Cases whose settings pass the library's `settingsInRange`. -/
  inRange : Nat := 0
  /-- Cases whose settings pass `ownLayersInRange`. -/
  ownLayersInRange : Nat := 0

/-- What has been compared in the run at hand. -/
structure RunCount where
  arrays : Nat := 0
  values : Nat := 0
  same : Bool := true

/-- Compares one listed array with Lean's rows for the positions below `positions`, and prints the
first difference. A name for which `row` has no value counts as a difference. -/
def compareArray (a : Listed) (positions : Nat) (row : Nat → Option IntegerVector) (sofar : RunCount) :
    IO RunCount := do
  let close (values : Nat) (difference : Option String) : IO RunCount := do
    if let some words := difference then IO.println s!"  DIFFERS {a.file}: {words}"
    return {
      arrays := sofar.arrays + 1, values := sofar.values + values
      same := sofar.same && difference.isNone }
  let want ← readFlat a
  if a.shape[0]? ≠ some positions then
    return ← close 0 s!"shape {a.shape.toList}: its first size should be {positions}"
  let mut got : IntegerVector := Array.mkEmpty want.size
  for t in [0:positions] do
    match row t with
    | some v => got := got ++ v
    | none => return ← close 0 s!"this program knows no value of the name {a.symbol}"
  if got.size ≠ want.size then
    return ← close 0 s!"the file has {want.size} values, Lean has {got.size}"
  let places := (Array.range want.size).filter fun i => want[i]! ≠ got[i]!
  close want.size <| places[0]?.map fun i =>
    let each := want.size / positions
    s!"index {i} (position {i / each}, entry {i % each}): the file has {want[i]!}, " ++
      s!"Lean has {got[i]!}; {places.size} of {want.size} values differ"

/-- The names that a run must list if it compares any position: the 32 values that the package
reports, 21 in a dense layer, 25 in a layer with experts, and 7 behind the layers. -/
def requiredNames (s : Settings) : Array String := Id.run do
  let mut names : Array String := #[]
  for l in [0:s.layers] do
    let own := #["x", "a_sigma", "a_Rc", "a_w", "a", "qw", "q", "kw", "k", "vw", "v", "o", "x_mid",
        "m_sigma", "m_Rc", "m_w", "m"]
      ++ (if s.isMixture l then #["lr", "e_star", "sc", "xr"] else #[]) ++ #["gw", "key", "s", "u"]
    names := names ++ own.map fun name => s!"L{l}.{name}"
  return names ++ #["x_last", "aF_sigma", "aF_Rc", "aF_w", "aF", "lam", "first_choice"]

/-- What is wrong with a run's own description, if anything. A run that expects a token, or whose
values are the package's with its interval widened, compares all positions, another run those before
the package's first failing position. A run that compares any position lists all of
`requiredNames`, so a maker that drops values does not pass. The tokens must be a prompt
(`isPrompt`), so that a "none" does not come from there. -/
def faultOfTheList (s : Settings) (run : Json) (theirs : Array Listed) (prompt : Array Nat)
    (positions : Nat) : Except String (Option String) := do
  let expected ← run.getObjVal? "expected"
  let all := (← expected.getObjValAs? String "kind") == "token"
    || (expected.getObjValAs? Bool "values_with_the_interval_widened").toOption == some true
  let due ← if all then pure prompt.size else expected.getObjValAs? Nat "position"
  if positions ≠ due then
    return some s!"positions_compared is {positions}, but {due} positions are to be compared"
  let missing := if positions = 0 then #[] else
    (requiredNames s).filter fun name => !theirs.any (·.name == name)
  if missing.size > 0 then return some s!"the run does not list {missing.toList}"
  if !isPrompt s prompt then return some "Lean's isPrompt says false of the run's prompt"
  return none

/-- Does Lean agree with a run's "expected"? With words for the run's line. `result` is `model` on
the prompt, and `defined n` says whether `model` is defined on the prompt's first `n` tokens. Where
"none" is expected, the model must be defined on the tokens before the position "undefined_from" and
undefined with that position, so that the failure is a range condition's, at that place. -/
def outcomeAgrees (run : Json) (result : Option Nat) (defined : Nat → Bool) :
    Except String (Bool × String) := do
  let expected ← run.getObjVal? "expected"
  let shown := match result with
    | some token => s!"Lean: some {token}"
    | none => "Lean: none"
  match ← expected.getObjValAs? String "kind" with
  | "token" =>
    let token ← expected.getObjValAs? Nat "token"
    return (result == some token, s!"expected: token {token}; {shown}")
  | "none" =>
    let q ← expected.getObjValAs? Nat "undefined_from"
    let before := q == 0 || defined q
    let there := !defined (q + 1)
    let words := s!"expected: none, from position {q} on; {shown}, " ++
      (if before then "defined before it, " else "NOT defined before it, ") ++
      (if there then "undefined with it" else "STILL defined with it")
    return (result == none && before && there, words)
  | "package refuses" =>
    match (expected.getObjValAs? Nat "token_with_the_interval_widened").toOption with
    | some token => return (result == some token, s!"expected: defined, token {token}; {shown}")
    | none => return (result.isSome, s!"expected: defined; {shown}")
  | other => throw s!"unknown kind of expected result: {other}"

/-- Prints one run's line and adds the run to the count. -/
def closeRun (count : Count) (name words : String) (run : RunCount) : IO Count := do
  let verdict := if run.same then "same   " else "DIFFERS"
  IO.println s!"{verdict} {name}: {words}; {run.arrays} arrays, {run.values} values compared"
  (← IO.getStdout).flush
  return { count with
    runs := count.runs + 1, runsThatDiffer := count.runsThatDiffer + (if run.same then 0 else 1)
    arrays := count.arrays + run.arrays, values := count.values + run.values }

/-- One run of a case with the whole model: `model` gives the outcome and `trace` the values. -/
def compareWhole (c : Case) (weights : Weights) (run : Json) (theirs : Array Listed)
    (prompt : Array Nat) (positions : Nat) : IO (RunCount × String) := do
  let s := c.settings
  let (agrees, words) ← IO.ofExcept <| outcomeAgrees run (model s c.tables weights prompt) fun n =>
    (model s c.tables weights (prompt.extract 0 n)).isSome
  let values := trace s c.tables weights prompt
  let top : Top := { values with }
  let mut sofar : RunCount := { same := agrees }
  for a in theirs do
    sofar ← compareArray a positions (sofar := sofar) fun t =>
      match a.layer with
      | some l => layerValue s (values.layer.entry l) a.symbol t
      | none => topValue top a.symbol t
  return (sofar, words)

/-- One run of a case in which matrices are left out: the routed experts that no position chose, and
the embedding's rows of other tokens. `model` would say `none` of it, because `wellFormed` asks for
all of them; and at Llama 4 Maverick's sizes two layers as arrays of `Int` need more memory than a
small machine has. So Model.lean's functions are evaluated one after the other, each with only the
matrices that it reads: `attention` on the stream; `feedForward` on each row of its result, with the
shared block, the router and the one routed expert that the files name for that position (if Lean
chose another, it would read an empty matrix, and the values would differ); behind the layers
`rmsNorm`, `product` and `firstArgmax`. They are put together here as in `layer`, `firstLayers` and
`trace`: each function's input is the result of the one before, not a value from the files. The
range conditions are evaluated with `attentionOk`, `feedForwardOk` and `normOk`, put together as in
`allConditionsHold`. `wellFormed` is not evaluated. -/
def compareByFunction (c : Case) (run : Json) (theirs : Array Listed) (prompt : Array Nat)
    (positions : Nat) : IO (RunCount × String) := do
  let s := c.settings
  if (run.getObjVal? "expected" >>= (·.getObjValAs? String "kind")).toOption ≠ some "token" then
    fail s!"{c.name}: a case with matrices left out must expect a token"
  let mut sofar : RunCount := {}
  let mut holds := isPrompt s prompt
  let embedding ← c.embedding
  let mut stream : Matrix := prompt.map fun token => embedding.entry token
  for l in [0:s.layers] do
    let p := s.perLayer.entry l
    let mixture := s.isMixture l
    let attended := attention s p c.tables
      (← c.layerWeights l fun m => ["Wq", "Wk", "Wv", "Wo"].contains m) stream
    let shared ← c.layerWeights l fun m => ["Wg", "Wu", "Wd", "Wr"].contains m
    let named ← if mixture then readFlat (← find theirs s!"L{l}.e_star") else pure #[]
    let mut fed : Array FeedForward := #[]
    for t in [0:positions] do
      let one ← c.layerWeights l fun m => m.startsWith s!"E{named.entry t}."
      let weights := { shared with routed := one.routed }
      fed := fed.push (feedForward s p c.tables weights mixture (attended.out.entry t))
    let L : Layer :=
      { input := stream, attention := attended, feedForward := fed, out := fed.map (·.out) }
    for a in theirs do
      if a.layer == some l then
        sofar ← compareArray a positions (layerValue s L a.symbol) sofar
    holds := holds && (List.range positions).all fun t =>
      attentionOk p attended t && feedForwardOk p (fed.entry t)
    stream := L.out
  let head ← c.matrix "Wh"
  let finalNorm := stream.map fun row => rmsNorm s.finalBits s.finalShift s.finalEps row
  let logits := finalNorm.map fun norm => product norm.out head s.vocabulary
  let choices := logits.map fun lam => firstArgmax (fun v => lam.entry v) s.vocabulary
  let top : Top := { last := stream, finalNorm, logits, choices }
  for a in theirs do
    if a.layer.isNone then
      sofar ← compareArray a positions (topValue top a.symbol) sofar
  holds := holds && finalNorm.all (normOk s.finalEps)
  let result := if holds then some (choices.entry (prompt.size - 1)) else none
  let (agrees, words) ← IO.ofExcept (outcomeAgrees run result fun _ => false)
  return ({ sofar with same := sofar.same && agrees }, words ++ " (function by function)")

/-- The clauses of the library's `settingsInRange`, asked of the layers that the model has.
`settingsInRange` asks them of 48 layers, so it says false of every model with fewer. The tests of a
norm and of a layer are the library's own. The clause on the softmax shifts is written again here,
and that all are asked of `s.layers` layers is this program's doing. -/
def ownLayersInRange (s : Settings) : Bool :=
  VerifiableInference.normInRange s.finalBits s.finalShift s.finalEps
  && (List.range s.layers).all fun l =>
      VerifiableInference.layerSettingsInRange (s.isMixture l) (s.perLayer.entry l)
      && (s.perLayer.entry l).softmaxShift == (s.perLayer.entry (4 * (l / 4))).softmaxShift

/-- At 48 layers the two are the same test. If the library's differs, this does not compile. -/
example (s : Settings) (h : s.layers = 48) :
    ownLayersInRange s = VerifiableInference.settingsInRange s := by
  simp [ownLayersInRange, VerifiableInference.settingsInRange, h]

/-- `OUT/cases/NAME`. -/
def compareCase (out folder : FilePath) (count : Count) : IO Count := do
  let j ← readJson (folder / "case.json")
  let name ← IO.ofExcept (j.getObjValAs? String "name")
  if (j.getObjValAs? Nat "format").toOption ≠ some 1 then
    fail s!"{name}: this program reads format 1"
  let tables := out / "tables" / (← IO.ofExcept (j.getObjValAs? String "tables"))
  let tableArrays ← IO.ofExcept (listedArrays tables (← readJson (tables / "tables.json")))
  let c : Case := {
    name
    settings := ← IO.ofExcept (settings (← IO.ofExcept (j.getObjVal? "settings")))
    tables := ← readTables tableArrays
    arrays := ← IO.ofExcept (listedArrays folder j)
    whole := ← IO.ofExcept (j.getObjValAs? Bool "every_matrix_is_written") }
  let weights : Weights ← if c.whole then
      pure {
        embedding := ← c.embedding
        perLayer := ← (Array.range c.settings.layers).mapM (c.layerWeights ·)
        head := ← c.matrix "Wh" }
    else pure default
  let inRange := VerifiableInference.settingsInRange c.settings
  let own := ownLayersInRange c.settings
  IO.println <| s!"        {name}: settingsInRange, which asks 48 layers: {inRange}; " ++
    s!"its clauses on this model's {c.settings.layers} layers: {own}"
  let mut count := { count with
    cases := count.cases + 1, inRange := count.inRange + inRange.toNat
    ownLayersInRange := count.ownLayersInRange + own.toNat }
  let fits := !c.whole || wellFormed c.settings c.tables weights
  if !fits then IO.println s!"  DIFFERS {folder / "case.json"}: Lean's wellFormed says false"
  let runs ← IO.ofExcept (j.getObjValAs? (Array Json) "runs")
  if runs.isEmpty then fail s!"{name}: the case has no runs"
  for run in runs, number in [0:runs.size] do
    let prompt ← IO.ofExcept (run.getObjValAs? (Array Nat) "prompt")
    let positions ← IO.ofExcept (run.getObjValAs? Nat "positions_compared")
    let theirs ← IO.ofExcept (listedArrays folder run)
    let (compared, words) ←
      if c.whole then compareWhole c weights run theirs prompt positions
      else compareByFunction c run theirs prompt positions
    let fault ← IO.ofExcept (faultOfTheList c.settings run theirs prompt positions)
    if let some words := fault then IO.println s!"  DIFFERS {folder / "case.json"}: {words}"
    let compared := { compared with same := compared.same && fits && fault.isNone }
    count ← closeRun count s!"{name} run {number}" words compared
  return count

/-- `OUT/functions/first_argmax.*`: `firstArgmax` on given rows of logits. -/
def compareFirstArgmax (folder : FilePath) (count : Count) : IO Count := do
  let arrays ← IO.ofExcept (listedArrays folder (← readJson (folder / "function.json")))
  let lam ← readMatrix (← find arrays "lam")
  let compared ← compareArray (← find arrays "first_choice") lam.size (sofar := {}) fun t =>
    some #[firstArgmax (fun v => (lam.entry t).entry v) (lam.entry t).size]
  closeRun { count with cases := count.cases + 1 } (folder.fileName.getD "")
    "firstArgmax on given rows" compared

/-- `OUT/tables/ID`: if tables.json has "compare_with_the_librarys_tables": true, its five tables
are compared with `VerifiableInference.lookupTables`, entry for entry, and the lowest keys too. -/
def compareTables (folder : FilePath) (count : Count) : IO Count := do
  let j ← readJson (folder / "tables.json")
  if (j.getObjValAs? Bool "compare_with_the_librarys_tables").toOption ≠ some true then
    return count
  let arrays ← IO.ofExcept (listedArrays folder j)
  let ours := VerifiableInference.lookupTables
  let mut sofar : RunCount := {}
  for (name, rows, lowestKey) in [
      ("cosine", ours.cosine, none), ("sine", ours.sine, none),
      ("exponential", ours.exponential.map (#[·]), some 0),
      ("silu", ours.silu.entries.map (#[·]), some ours.silu.lowestKey),
      ("sigmoid", ours.sigmoid.entries.map (#[·]), some ours.sigmoid.lowestKey)] do
    let a ← find arrays name
    sofar ← compareArray a rows.size (fun t => some (rows.entry t)) sofar
    if a.lowestKey ≠ lowestKey then
      IO.println s!"  DIFFERS {a.file}: lowest key: listed {a.lowestKey}, Lean has {lowestKey}"
      sofar := { sofar with same := false }
  closeRun { count with cases := count.cases + 1 } s!"tables {folder.fileName.getD ""}"
    "the library's lookupTables, entry for entry" sofar

/-- `OUT/settings/NAME.json`: the library's `settingsAreValid`, the test on the settings at which
its theorems are stated, must give what the file lists. -/
def compareSettings (file : FilePath) (count : Count) : IO Count := do
  let j ← readJson file
  let s ← IO.ofExcept (settings (← IO.ofExcept (j.getObjVal? "settings")))
  let listed ← IO.ofExcept (j.getObjValAs? Bool "settingsAreValid")
  let found := VerifiableInference.settingsAreValid s
  closeRun { count with cases := count.cases + 1 } s!"settings {file.fileName.getD ""}"
    s!"expected: settingsAreValid {listed}; Lean: {found}" { values := 1, same := listed == found }

def sortedFolders (folder : FilePath) : IO (Array FilePath) := do
  if !(← folder.isDir) then return #[]
  return ((← folder.readDir).map (·.path)).qsort (·.toString < ·.toString)

end CompareVectors

open CompareVectors in
def main (arguments : List String) : IO UInt32 := do
  let [out] := arguments | do IO.eprintln "usage: compare_vectors OUT"; return 2
  try
    let out : FilePath := out
    let mut count : Count := {}
    for folder in ← sortedFolders (out / "cases") do
      count ← compareCase out folder count
    let models := count.cases
    for folder in ← sortedFolders (out / "functions") do
      count ← compareFirstArgmax folder count
    let nothing := count.runs = 0
    for folder in ← sortedFolders (out / "tables") do
      count ← compareTables folder count
    for file in ← sortedFolders (out / "settings") do
      count ← compareSettings file count
    IO.println <| s!"settings in range, of {models} cases: {count.inRange} by settingsInRange, " ++
      s!"{count.ownLayersInRange} by its clauses on their own layers alone"
    IO.println <| s!"{count.cases} cases, {count.runs} runs, {count.arrays} arrays, " ++
      s!"{count.values} values compared; {count.runsThatDiffer} runs differ"
    if nothing then
      IO.eprintln "no case and no function to compare"
      return 2
    return if count.runsThatDiffer = 0 then 0 else 1
  catch e =>
    IO.eprintln s!"cannot go on: {e}"
    return 2
