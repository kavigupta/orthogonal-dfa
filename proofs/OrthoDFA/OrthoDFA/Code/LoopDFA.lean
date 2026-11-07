import OrthoDFA.Code.CoveredStart
import OrthoDFA.Code.FamilyRead

/-! Choosing the leaf the final hypothesis starts at. -/

noncomputable section

namespace OrthoDFA

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

end OrthoDFA

end
