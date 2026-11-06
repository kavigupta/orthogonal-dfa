import OrthoDFA.Code.CoveredStart
import OrthoDFA.Code.FamilyRead
import OrthoDFA.Code.LearnerLoop

/-! Choosing the leaf the final hypothesis starts at, and the learner: the loop, then that
choice on held-out strings. -/

noncomputable section

namespace OrthoDFA

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

open scoped Classical in
/-- How many of `xs` the hypothesis, run from the leaf `s`, classifies as `read` decides them. -/
def leafScore (H : HypothesisDFA α) (read : FreeMonoid α → ARU) {n : ℕ}
    (xs : Fin n → FreeMonoid α) (s : List Bool) : ℕ :=
  (Finset.univ.filter fun i => read (xs i) ≠ .undecided
    ∧ H.Classifies s (xs i) (read (xs i) = .accept)).card

/-- The leaf of highest `leafScore`, the first of those tied. -/
def pickLeaf (H : HypothesisDFA α) (read : FreeMonoid α → ARU) {n : ℕ}
    (xs : Fin n → FreeMonoid α) : List Bool :=
  (H.tree.leaves.argmax (leafScore H read xs)).getD []

/-- The final hypothesis, started at the leaf that scores best on `xs`, classifies `x` as `b`. -/
def HypothesisDFA.rerootedDFA (H : HypothesisDFA α) (read : FreeMonoid α → ARU) {n : ℕ}
    (xs : Fin n → FreeMonoid α) (x : FreeMonoid α) (b : Prop) : Prop :=
  H.Classifies (pickLeaf H read xs) x b

/-- `pickLeaf` as a program: it asks the read at each of `xs`. -/
def pickLeafQ (H : HypothesisDFA α) {n : ℕ} (xs : Fin n → FreeMonoid α) :
    Query (FreeMonoid α) ARU (List Bool) :=
  (Query.table (List.ofFn xs)).map fun r => pickLeaf H r xs

/-- The learner: the loop, then the leaf its last round's hypothesis starts at, picked on the
held-out strings `xs` as that round's family reads them; `none` if the loop fails. -/
def learnerQ {K : ℕ} (settings : LoopSettings K) (D Dsf : Measure (FreeMonoid α))
    (draws : Fin K → RoundDraws α K settings.level3.harvestReadCap settings.level3.probeBudget)
    {n : ℕ} (xs : Fin n → FreeMonoid α) :
    Query (FreeMonoid α) ℝ (Option (List (LoopRound α) × List Bool)) :=
  (loopQ settings D Dsf draws).bind fun
  | none => .pure none
  | some rounds =>
    match rounds.getLast? with
    | some o => ((pickLeafQ o.result.hypothesis xs).subst
        (familyReadQ o.family o.state.lo (o.state.hi + 1))).map fun s => some (rounds, s)
    | none => .pure (some (rounds, []))

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}

/-- The rounds and the start leaf the learner returns at the noise `p.1`, the draws `p.2` and
the held-out strings `xs`: `learnerQ` asking `O`'s answers at that noise. -/
def indecisiveLStar {K : ℕ} (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α))
    (D Dsf : Measure (FreeMonoid α))
    (p : LoopDraws Ω α K settings.level3.harvestReadCap settings.level3.probeBudget) {n : ℕ}
    (xs : Fin n → FreeMonoid α) : Option (List (LoopRound α) × List Bool) :=
  (learnerQ settings D Dsf p.2 xs).eval fun w => O.mq w p.1

end OrthoDFA

end
