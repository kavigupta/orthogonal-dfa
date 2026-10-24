import OrthoDFA.Code.HypothesisDFA

noncomputable section

namespace OrthoDFA

open MeasureTheory
open scoped ENNReal

variable {α : Type*} [Fintype α] [DecidableEq α]

/-- Uniform over the strings of length `n`. -/
def uniformStrings (n : ℕ) : Measure (FreeMonoid α) :=
  ((Fintype.card α : ℝ≥0∞) ^ n)⁻¹
    • ∑ v : List.Vector α n, Measure.dirac (FreeMonoid.ofList v.toList)

/-- From the leaf `s`, `H` follows learned edges along all of `x` to a leaf, which is on the root's
accept side exactly when `b`. -/
def HypothesisDFA.Classifies (H : HypothesisDFA α) (s : List Bool) (x : FreeMonoid α) (b : Prop) :
    Prop :=
  ∃ p, walkEnd H.edges s x.toList = some p ∧ (p.head? = some true ↔ b)

end OrthoDFA

end
