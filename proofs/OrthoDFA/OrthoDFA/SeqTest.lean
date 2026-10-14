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

open scoped Classical in
/-- Over independent draws from `D`, testing how many of the first `n` satisfy `P` against `p₀`,
at every `n`, fires at some look with chance at most `a` when `P`'s chance is on the far side of
`p₀` from `p₁`. -/
def SeqRateSound : Prop :=
  ∀ {X : Type*} [MeasurableSpace X] [Countable X] [MeasurableSingletonClass X]
    (D : Measure X) [IsProbabilityMeasure D] (P : X → Prop) (p₀ p₁ a : ℝ) (N : ℕ),
    0 < p₀ → p₀ < 1 → 0 ≤ p₁ → p₁ ≤ 1 → 0 < a →
    (D.real {x | P x} - p₀) * (p₁ - p₀) ≤ 0 →
    (Measure.pi fun _ : Fin N => D).real
      {xs | ∃ n ≤ N, seqRate p₀ p₁ a n (((List.ofFn xs).take n).countP fun x => decide (P x))}
      ≤ a

end OrthoDFA
