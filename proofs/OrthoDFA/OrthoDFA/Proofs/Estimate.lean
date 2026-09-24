import OrthoDFA.Proofs.Discharge

/-!
# Sample → distribution: two-sided concentration

The gate estimates a rate (FNR, or misclassification) on `m` fresh prefixes drawn
from a population's distribution, then resamples more when it needs a tighter read
(PR #257). Each fresh prefix contributes an independent `[0,1]` indicator whose
mean is the *distributional* rate `p`. So the empirical mean concentrates on `p`:
this is the same Hoeffding as the vote, pointed at the prefix axis.

`twoSided` bounds `P[|empirical − p| ≥ γ]`; the one-sided pieces are the already
discharged `wrongDecisive_le` (upper) and `misplacedMember_le` (lower).
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory
open scoped ENNReal NNReal

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]

/-- Upper tail: total mean ≤ `k·b` ⇒ the sum exceeds `k(b+γ)` w.p. ≤ `exp(-2kγ²)`. -/
theorem sumUpper_le {ι : Type*} (X : ι → Ω → ℝ) (idx : Finset ι) (b γ : ℝ)
    (hmeas : ∀ i, AEMeasurable (X i) μ) (h_indep : iIndepFun X μ)
    (hIcc : ∀ i, ∀ᵐ ω ∂μ, X i ω ∈ Set.Icc (0 : ℝ) 1)
    (hmean : ∑ i ∈ idx, μ[X i] ≤ (idx.card : ℝ) * b) (hγ : 0 ≤ γ) :
    μ.real {ω | (idx.card : ℝ) * (b + γ) ≤ ∑ i ∈ idx, X i ω}
      ≤ Real.exp (-2 * (idx.card : ℝ) * γ ^ 2) :=
  wrongDecisive_le X idx b γ hmeas h_indep hIcc hmean hγ

/-- Lower tail: total mean ≥ `k·b` ⇒ the sum falls below `k(b-γ)` w.p. ≤ `exp(-2kγ²)`. -/
theorem sumLower_le {ι : Type*} (X : ι → Ω → ℝ) (idx : Finset ι) (b γ : ℝ)
    (hmeas : ∀ i, AEMeasurable (X i) μ) (h_indep : iIndepFun X μ)
    (hIcc : ∀ i, ∀ᵐ ω ∂μ, X i ω ∈ Set.Icc (0 : ℝ) 1)
    (hmean : (idx.card : ℝ) * b ≤ ∑ i ∈ idx, μ[X i]) (hγ : 0 ≤ γ) :
    μ.real {ω | ∑ i ∈ idx, X i ω ≤ (idx.card : ℝ) * (b - γ)}
      ≤ Real.exp (-2 * (idx.card : ℝ) * γ ^ 2) := by
  have hmean' : (idx.card : ℝ) * ((b - γ) + γ) ≤ ∑ i ∈ idx, μ[X i] := by
    rw [show (b - γ) + γ = b by ring]; exact hmean
  have h := misplacedMember_le X idx (b - γ) γ 0 hmeas h_indep hIcc hmean' hγ
  simpa only [add_zero, sub_zero] using h

end OrthoDFA
