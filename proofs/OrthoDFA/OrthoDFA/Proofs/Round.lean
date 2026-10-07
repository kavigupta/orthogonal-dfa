import OrthoDFA.Round
import OrthoDFA.Proofs.Replay

namespace OrthoDFA

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α] {Ω : Type*} [MeasurableSpace Ω]
variable {μ : Measure Ω} {Q : Type*}

/-- Believed true: `RoundOutcome`'s yield gives `d ≤ L · P(harvest)`, its harvest spread gives
`P(t harvested) ≤ ∑ min(i+1,L)/L · D(first i letters are t's)`, and for `i ≤ |t|` that last chance
is `L · ν(t's first i letters) ≤ L · κ · V(its state)`. -/
theorem harvestSpread_of (A : DFA (FreeMonoid α) Q) (R : CutReads α) (H : Hypothesis α)
    (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (L : ℕ) (κ : ℝ)
    (hb : R.B.lo ≤ R.B.hi)
    (hκ : ∀ y, prefixWeight D L y ≤ κ * stateWeight A D L (A.state y)) :
    HarvestSpread A R H D L κ := by
  sorry

end OrthoDFA
