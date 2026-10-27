import OrthoDFA.Code.RateTest

/-! The binomial test's promise. -/

namespace OrthoDFA

open MeasureTheory

open scoped Classical in
/-- Over `n` independent draws from `D`, testing how many satisfy `P` against `θ` says above with
chance at most `a` when `P` has chance at most `θ`, and below with chance at most `a` when it has
chance at least `θ`. -/
def RateTestSound : Prop :=
  ∀ {X : Type*} [MeasurableSpace X] [Countable X] [MeasurableSingletonClass X]
    (D : Measure X) [IsProbabilityMeasure D] (P : X → Prop) (θ a : ℝ) (n₀ n : ℕ),
    0 ≤ θ → θ ≤ 1 → 0 ≤ a →
    (D.real {x | P x} ≤ θ →
      (Measure.pi fun _ : Fin n => D).real
        {xs | rateSide θ a n₀ n (Finset.univ.filter fun i => P (xs i)).card = some true} ≤ a)
    ∧ (θ ≤ D.real {x | P x} →
      (Measure.pi fun _ : Fin n => D).real
        {xs | rateSide θ a n₀ n (Finset.univ.filter fun i => P (xs i)).card = some false} ≤ a)

end OrthoDFA
