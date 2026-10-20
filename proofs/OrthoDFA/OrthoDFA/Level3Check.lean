import OrthoDFA.Level3

/-!
# Level 3 with the start check

Level 3 (`runProbes`) runs its probes, walked from `walkStart`. Where it would end consistent the
round checks the hypothesis from `ε` instead: it walks each of a fresh block of strings from `0`
and counts those whose probe does not come out `agree` (`startFails`), testing that count by
`seqRate` against half the check's rate. The `i`-th check runs at level `δ / (8 (i + 1) (i + 2))`.
A passing check ends the round consistent. A failing one takes the first of its strings whose
probe from `0` crosses a wrong learned edge or stops at an unlearned one, changes the hypothesis as
level 3 does for such a probe, never counting it, and runs level 3 again from there on a fresh
block of probes. Where no string does, or the blocks run out, the round fails.

Known modelling gap. The Python leaves the strings whose sift is undecided out of the check's
denominator; here they fail it. Counting them makes the check certify a rate for every string, but
it needs the family's indecision far below the check's rate, which the model's indecision limit
is and the Python's FNR limit is not.

Known modelling gap. The Python's check (`estimate_agreement_rate`) asks `rate_side` after every
draw from the 30th on, at one level, and at its cap of 2000 draws compares its point estimate with
the bar. Here the check is the sequential test level 3's success test uses, anytime valid, and a
check that does not fire within its block fails.

Known modelling gap. The Python probes every string of a failed check from `ε` and repeats the pass
after every change until none changes the hypothesis (`settle`), and its search goes on where none
does. Here one string's change is made, and a check whose strings change nothing ends the round
without a population.
-/

namespace OrthoDFA

namespace Random

variable {α : Type*} [DecidableEq α]

structure CheckSettings where
  /-- The share of its strings a passing check may fail. -/
  rate : ℝ
  /-- The chance the checks may pass a hypothesis failing more than `rate`. -/
  δ : ℝ
  /-- The strings each check reads. -/
  size : ℕ
  /-- The level-3 runs a round may make. -/
  runs : ℕ

/-- The `i`-th check's level. -/
noncomputable def CheckSettings.level (K : CheckSettings) (i : ℕ) : ℝ :=
  K.δ / (8 * (i + 1) * (i + 2))

section

variable (read : FreeMonoid α → ARU)

/-- Walked from `0` under the hypothesis, the string's probe does not come out `agree`. -/
def startFails (T : DTree α) (E : Edges α) (x : FreeMonoid α) : Bool :=
  match probe read T E 0 x with
  | .agree => false
  | _ => true

/-- After some number of its strings the check's test fires below its rate. -/
def checkPasses (K : CheckSettings) (i : ℕ) (T : DTree α) (E : Edges α)
    (xs : List (FreeMonoid α)) : Prop :=
  ∃ n ≤ xs.length,
    seqRate K.rate (K.rate / 2) (K.level i) n ((xs.take n).countP (startFails read T E))

/-- The change level 3 makes for a probe of `x` from `0` that crosses a wrong learned edge or
stops at an unlearned one. -/
def fixOf (s : Level3State α) (x : FreeMonoid α) : Option (Level3State α) :=
  match probe read s.tree s.edges 0 x with
  | .unlearnedEdge p c u t =>
    some (newHypothesis s.tree (Function.update s.edges (p, c) (some (t, u))) (s.hypotheses + 1))
  | .wrongEdge p c _ t =>
    match s.edges (p, c) with
    | some (t₀, _) => some (splitLeaf read s p c t t₀)
    | none => none
  | _ => none

/-- The first of `xs` with a change. -/
def firstFix (s : Level3State α) : List (FreeMonoid α) → Option (Level3State α)
  | [] => none
  | x :: xs => (fixOf read s x).elim (firstFix s xs) some

variable (C : Level3Settings) (K : CheckSettings)

open scoped Classical in
/-- Level 3's runs from the `i`-th on, one per block: the block's probes, then its check
strings. -/
noncomputable def checkedRun :
    List (List (FreeMonoid α) × List (FreeMonoid α)) → ℕ → Level3State α
      → Level3State α × RoundEnd Level3Site
  | [], _, s => (s, .failed)
  | b :: bs, i, s =>
    let r := runProbes read C s b.1
    match r.2 with
    | .consistent =>
      if checkPasses read K i r.1.tree r.1.edges b.2 then r
      else match firstFix read r.1 b.2 with
        | some s' => checkedRun bs (i + 1) s'
        | none => (r.1, .failed)
    | e => (r.1, e)

end

/-- A block of the round's draws: a run's probes and its check's strings. -/
abbrev CheckBlock (α : Type*) (C : Level3Settings) (K : CheckSettings) :=
  (Fin C.probeBudget → FreeMonoid α) × (Fin K.size → FreeMonoid α)

/-- The round: at most `K.runs` runs of level 3, one per block. -/
noncomputable def indecisiveLStarCheck (C : Level3Settings) (K : CheckSettings)
    (read : FreeMonoid α → ARU) (blocks : Fin K.runs → CheckBlock α C K) : Level3Result α :=
  let r := checkedRun read C K
    (List.ofFn fun i => (List.ofFn (blocks i).1, List.ofFn (blocks i).2)) 0 initialState
  ⟨⟨r.1.tree, r.1.edges⟩, r.2⟩

end Random

end OrthoDFA
