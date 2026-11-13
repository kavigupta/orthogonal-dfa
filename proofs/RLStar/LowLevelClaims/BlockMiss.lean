import Mathlib.MeasureTheory.Constructions.Pi

namespace OrthoDFA

open MeasureTheory

/-- `b` draws from `D`, after a past `p` drawn from `ν` that fixes a set `A p` holding at least
`q` of `D`, all miss it with chance at most

    (1 − q)^b. -/
def BlockMiss : Prop :=
  ∀ {P S : Type*} [MeasurableSpace P] [MeasurableSpace S] [Countable S]
    [MeasurableSingletonClass S] (ν : Measure P) [IsProbabilityMeasure ν]
    (D : Measure S) [IsProbabilityMeasure D] (b : ℕ) (q : ℝ) (A : P → Set S),
    (∀ s, MeasurableSet {p | s ∈ A p}) → q ≤ 1 →
    (ν.prod (Measure.pi fun _ : Fin b => D)).real
        {pw | q ≤ D.real (A pw.1) ∧ ∀ i, pw.2 i ∉ A pw.1} ≤ (1 - q) ^ b

end OrthoDFA
