import OrthoDFA.Clustering

/-! Hoeffding's (1956) comparison of a sum of independent bits with a binomial. -/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

/-- A sum of independent bits over `s`, whose rates may differ, lies at or below a count `b` no
more often than `Bin(n, q)` does, once its mean is at least `nq ≥ b + 2`; and at or above `c` no
more often than `Bin(n, q)` does, once its mean is at most `nq ≤ c − 2`.  The binomial may have
more trials than `s` has bits. -/
def BitSumTails : Prop :=
  ∀ {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ] {ι : Type*}
    (X : ι → Ω → ℝ) (s : Finset ι) (n : ℕ) (q : ℝ),
    (∀ i, Measurable (X i)) → iIndepFun X μ → (∀ i, ∀ᵐ ω ∂μ, X i ω = 0 ∨ X i ω = 1) →
    s.card ≤ n → 0 ≤ q → q ≤ 1 →
    (∀ b : ℕ, n * q ≤ ∑ i ∈ s, μ[X i] → (b : ℝ) + 2 ≤ n * q →
      μ.real {ω | ∑ i ∈ s, X i ω ≤ b} ≤ 1 - binomSfGe n q (b + 1))
    ∧ (∀ c : ℕ, ∑ i ∈ s, μ[X i] ≤ n * q → n * q + 2 ≤ c →
      μ.real {ω | (c : ℝ) ≤ ∑ i ∈ s, X i ω} ≤ binomSfGe n q c)

end OrthoDFA
