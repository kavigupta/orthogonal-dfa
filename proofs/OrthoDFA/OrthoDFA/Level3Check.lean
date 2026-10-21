import OrthoDFA.Level3

/-!
# Level 3 with the start check

Each of level 3's draws is a probe and a block of check strings. Level 3 runs its probes, walked
from `walkStart`; where its success test would end the round consistent, the round checks the
hypothesis from `ε` on that draw's block instead. It walks each string of the block from `0` and
counts those whose probe parts from its sift or stops undecided (`startFails`), testing the count
by `seqRate` against half the check's rate, the `j`-th hypothesis's check at level
`δ / (8 (j + 1) (j + 2))`.
A passing check ends the round consistent. A failing one takes the first string of the block
whose probe from `0` crosses a wrong learned edge or stops at an unlearned one and changes the
hypothesis as level 3 does for such a probe, never counting it; level 3 goes on from there on its
next draws. Where no string does, the round fails.

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

Known modelling gap. Where a failing check string's probe from `ε` stops undecided, the Python
feeds the string to the next round's search, which harvests the undecided prefix. Here it changes
nothing, so a check whose strings all stop undecided ends the round without a population.
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

/-- The `j`-th hypothesis's check's level. -/
noncomputable def CheckSettings.level (K : CheckSettings) (j : ℕ) : ℝ :=
  K.δ / (8 * (j + 1) * (j + 2))

/-- A draw: a probe and a block of check strings. -/
abbrev CheckDraw (α : Type*) (K : CheckSettings) := FreeMonoid α × (Fin K.size → FreeMonoid α)

section

variable (read : FreeMonoid α → ARU)

/-- Walked from `0` under the hypothesis, the string's probe parts from its sift or stops
undecided: it does not come out `agree`, or undecided with no undecided string and no
disagreement. -/
def startFails (T : DTree α) (E : Edges α) (x : FreeMonoid α) : Bool :=
  match probe read T E 0 x with
  | .agree => false
  | .undecided [] true => false
  | _ => true

/-- After some number of its strings the check's test fires below its rate. -/
def checkPasses (K : CheckSettings) (j : ℕ) (T : DTree α) (E : Edges α)
    (xs : List (FreeMonoid α)) : Prop :=
  ∃ n ≤ xs.length,
    seqRate K.rate (K.rate / 2) (K.level j) n ((xs.take n).countP (startFails read T E))

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
/-- The check of the hypothesis on a block. -/
noncomputable def checkStep (s : Level3State α) (b : Fin K.size → FreeMonoid α) :
    Level3State α ⊕ RoundEnd Level3Site :=
  if checkPasses read K s.hypotheses s.tree s.edges (List.ofFn b) then .inr .consistent
  else match firstFix read s (List.ofFn b) with
    | some s' => .inl s'
    | none => .inr .failed

/-- Level 3's step on the draw's probe, the check on its block where the success test fires. -/
noncomputable def applyCheck (s : Level3State α) (d : CheckDraw α K) :
    Level3State α ⊕ RoundEnd Level3Site :=
  match applyProbe read C s d.1 with
  | .inr .consistent => checkStep read K s d.2
  | r => r

/-- The state the draws end in, and how; `failed` where they ran out. -/
noncomputable def runChecked :
    Level3State α → List (CheckDraw α K) → Level3State α × RoundEnd Level3Site
  | s, [] => (s, .failed)
  | s, d :: ds =>
    match applyCheck read C K s d with
    | .inl s' => runChecked s' ds
    | .inr e => (s, e)

end

noncomputable def indecisiveLStarCheck (C : Level3Settings) (K : CheckSettings)
    (read : FreeMonoid α → ARU) (draws : Fin C.probeBudget → CheckDraw α K) : Level3Result α :=
  let r := runChecked read C K initialState (List.ofFn draws)
  ⟨⟨r.1.tree, r.1.edges⟩, r.2⟩

end Random

end OrthoDFA
