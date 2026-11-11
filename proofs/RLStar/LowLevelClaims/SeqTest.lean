import RLStar.Code.SeqTest

/-! The sequential test's promise. -/

namespace OrthoDFA

open MeasureTheory

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

open scoped Classical in
/-- `SeqRateSound` where the test may count any sublist of the first `n` draws missing at most `d`
of them: it fires at some look with chance at most `a Rᵈ`, `R = max(p₀/p₁, (1 − p₀)/(1 − p₁))`. -/
def SeqRateRobustSound : Prop :=
  ∀ {X : Type*} [MeasurableSpace X] [Countable X] [MeasurableSingletonClass X]
    (D : Measure X) [IsProbabilityMeasure D] (P : X → Prop) (p₀ p₁ a : ℝ) (N d : ℕ),
    0 < p₀ → p₀ < 1 → 0 < p₁ → p₁ < 1 → 0 < a →
    (D.real {x | P x} - p₀) * (p₁ - p₀) ≤ 0 →
    (Measure.pi fun _ : Fin N => D).real
      {xs | ∃ n ≤ N, ∃ l : List X, l.Sublist ((List.ofFn xs).take n) ∧ n ≤ l.length + d
        ∧ seqRate p₀ p₁ a l.length (l.countP fun x => decide (P x))}
      ≤ a * max (p₀ / p₁) ((1 - p₀) / (1 - p₁)) ^ d

end OrthoDFA
