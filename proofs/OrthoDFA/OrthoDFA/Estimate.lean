import OrthoDFA.Discharge

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
theorem sumUpper_le (X : ℕ → Ω → ℝ) (k : ℕ) (b γ : ℝ)
    (hmeas : ∀ i, AEMeasurable (X i) μ) (h_indep : iIndepFun X μ)
    (hIcc : ∀ i, ∀ᵐ ω ∂μ, X i ω ∈ Set.Icc (0 : ℝ) 1)
    (hmean : ∑ i ∈ Finset.range k, μ[X i] ≤ (k : ℝ) * b) (hγ : 0 ≤ γ) :
    μ.real {ω | (k : ℝ) * (b + γ) ≤ ∑ i ∈ Finset.range k, X i ω}
      ≤ Real.exp (-2 * (k : ℝ) * γ ^ 2) :=
  wrongDecisive_le X k b γ hmeas h_indep hIcc hmean hγ

/-- Lower tail: total mean ≥ `k·b` ⇒ the sum falls below `k(b-γ)` w.p. ≤ `exp(-2kγ²)`. -/
theorem sumLower_le (X : ℕ → Ω → ℝ) (k : ℕ) (b γ : ℝ)
    (hmeas : ∀ i, AEMeasurable (X i) μ) (h_indep : iIndepFun X μ)
    (hIcc : ∀ i, ∀ᵐ ω ∂μ, X i ω ∈ Set.Icc (0 : ℝ) 1)
    (hmean : (k : ℝ) * b ≤ ∑ i ∈ Finset.range k, μ[X i]) (hγ : 0 ≤ γ) :
    μ.real {ω | ∑ i ∈ Finset.range k, X i ω ≤ (k : ℝ) * (b - γ)}
      ≤ Real.exp (-2 * (k : ℝ) * γ ^ 2) := by
  have hmean' : (k : ℝ) * ((b - γ) + γ) ≤ ∑ i ∈ Finset.range k, μ[X i] := by
    rw [show (b - γ) + γ = b by ring]; exact hmean
  have h := misplacedMember_le X k (b - γ) γ 0 hmeas h_indep hIcc hmean' hγ
  simpa only [add_zero, sub_zero] using h

/-- **Two-sided concentration.**  Total mean `= k·p`: the sum deviates from `k·p`
by `k·γ` w.p. ≤ `2·exp(-2kγ²)`.  With per-prefix `[0,1]` indicators of mean `p`
(the distributional rate), this is empirical-mean-concentrates-on-`p`. -/
theorem twoSided (X : ℕ → Ω → ℝ) (k : ℕ) (p γ : ℝ)
    (hmeas : ∀ i, AEMeasurable (X i) μ) (h_indep : iIndepFun X μ)
    (hIcc : ∀ i, ∀ᵐ ω ∂μ, X i ω ∈ Set.Icc (0 : ℝ) 1)
    (hmean : ∑ i ∈ Finset.range k, μ[X i] = (k : ℝ) * p) (hγ : 0 ≤ γ) :
    μ.real {ω | (k : ℝ) * γ ≤ |(∑ i ∈ Finset.range k, X i ω) - (k : ℝ) * p|}
      ≤ 2 * Real.exp (-2 * (k : ℝ) * γ ^ 2) := by
  set S : Ω → ℝ := fun ω => ∑ i ∈ Finset.range k, X i ω with hS
  have hup : μ.real {ω | (k : ℝ) * (p + γ) ≤ S ω} ≤ Real.exp (-2 * (k : ℝ) * γ ^ 2) :=
    sumUpper_le X k p γ hmeas h_indep hIcc (le_of_eq hmean) hγ
  have hlo : μ.real {ω | S ω ≤ (k : ℝ) * (p - γ)} ≤ Real.exp (-2 * (k : ℝ) * γ ^ 2) :=
    sumLower_le X k p γ hmeas h_indep hIcc (ge_of_eq hmean) hγ
  have hsub : {ω | (k : ℝ) * γ ≤ |S ω - (k : ℝ) * p|}
      ⊆ {ω | (k : ℝ) * (p + γ) ≤ S ω} ∪ {ω | S ω ≤ (k : ℝ) * (p - γ)} := by
    intro ω hω
    simp only [Set.mem_setOf_eq, Set.mem_union] at hω ⊢
    rcases le_abs.mp hω with h | h
    · left; nlinarith [h]
    · right; nlinarith [h]
  calc μ.real {ω | (k : ℝ) * γ ≤ |S ω - (k : ℝ) * p|}
      ≤ μ.real ({ω | (k : ℝ) * (p + γ) ≤ S ω} ∪ {ω | S ω ≤ (k : ℝ) * (p - γ)}) :=
        measureReal_mono hsub
    _ ≤ μ.real {ω | (k : ℝ) * (p + γ) ≤ S ω} + μ.real {ω | S ω ≤ (k : ℝ) * (p - γ)} :=
        measureReal_union_le _ _
    _ ≤ Real.exp (-2 * (k : ℝ) * γ ^ 2) + Real.exp (-2 * (k : ℝ) * γ ^ 2) :=
        add_le_add hup hlo
    _ = 2 * Real.exp (-2 * (k : ℝ) * γ ^ 2) := by ring

#print axioms twoSided

end OrthoDFA
