import Mathlib.MeasureTheory.Constructions.Pi

/-! An anytime-valid sequential test of a count of hits against a rate, and what it promises. -/

namespace OrthoDFA

open MeasureTheory

/-- The likelihood ratio of a rate `p₁` against `p₀` after `h` hits in `n` draws. -/
noncomputable def likelihoodRatio (p₀ p₁ : ℝ) (n h : ℕ) : ℝ :=
  (p₁ / p₀) ^ h * ((1 - p₁) / (1 - p₀)) ^ (n - h)

/-- The sequential test of a rate `p₀` against `p₁` at level `a` fires after `h` hits in `n`
draws: the likelihood ratio has reached `1 / a`. -/
def seqRate (p₀ p₁ a : ℝ) (n h : ℕ) : Prop := 1 / a ≤ likelihoodRatio p₀ p₁ n h

end OrthoDFA
