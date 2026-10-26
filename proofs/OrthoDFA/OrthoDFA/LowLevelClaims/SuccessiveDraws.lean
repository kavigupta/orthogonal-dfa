import OrthoDFA.LowLevelClaims.Clustering

/-! Sampling without replacement, read off independent draws.

A draw is rejected when it lands in `T` or repeats an earlier draw; the distinct draws outside
`T` are what a sampler without replacement keeps. Each draw lands in a set the earlier draws fix,
with chance that set's mass, so the counts compare with binomials whatever repeats.
-/

namespace OrthoDFA

open MeasureTheory

open scoped Classical in
/-- Over `n` independent draws from `D`, where `T` together with any fewer than `n` further
strings holds at most `t` of `D`: the rejected draws reach `c` no more often than `Bin(n, t)`
does; the kept draws in `P` reach `c` no more often than `Bin(n, D(P))` does, and stay at or below
`c` no more often than `Bin(n, D(P) − t)` does. -/
def SuccessiveDrawsTails : Prop :=
  ∀ {S : Type*} [MeasurableSpace S] [Countable S] [MeasurableSingletonClass S] [DecidableEq S]
    (D : Measure S) [IsProbabilityMeasure D] (n : ℕ) (T : Finset S) (t : ℝ) (P : Set S),
    t ≤ 1 → (∀ F : Finset S, F.card < n → D.real ↑(T ∪ F) ≤ t) →
    (∀ c : ℕ, (Measure.pi fun _ : Fin n => D).real
        {y | c + (Finset.univ.image y \ T).card ≤ n} ≤ binomSfGe n t c)
    ∧ (∀ c : ℕ, (Measure.pi fun _ : Fin n => D).real
        {y | c ≤ ((Finset.univ.image y \ T).filter (· ∈ P)).card} ≤ binomSfGe n (D.real P) c)
    ∧ (∀ c : ℕ, t ≤ D.real P → (Measure.pi fun _ : Fin n => D).real
        {y | ((Finset.univ.image y \ T).filter (· ∈ P)).card ≤ c}
          ≤ 1 - binomSfGe n (D.real P - t) (c + 1))

end OrthoDFA
