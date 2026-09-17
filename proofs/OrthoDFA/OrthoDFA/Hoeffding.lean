import Mathlib.Probability.Moments.SubGaussian

namespace OrthoDFA
open MeasureTheory ProbabilityTheory
open scoped ENNReal NNReal

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]

/-- Reads `X 0 … X (k-1)` independent, each in `[0,1]` a.e., with total mean at
most `k β` (equivalently average mean `≤ β`).  The event that their sum exceeds
`k(β+τ)` then has probability at most `exp(-2 k τ²)`.

Stated with the sum condition rather than a per-read one so it serves both the
false-positive case (a single string's `k` reads, each mean `≤ β`) and the
certification case (`n` prefixes of mixed classes whose average mean `≤ β+τ`).

For a vote (`k` i.i.d. reads of one string) the count is exactly `Bin(k, β±s)`,
so this sub-Gaussian bound is a conservative overestimate of the exact binomial
tail the code computes; it is used to keep the trusted base at Lean core, since
Mathlib lacks a "sum of i.i.d. Bernoulli = `Bin(k,p)`" law. -/
theorem wrongDecisive_le {ι : Type*}
    (X : ι → Ω → ℝ) (s : Finset ι) (β τ : ℝ)
    (hmeas : ∀ i, AEMeasurable (X i) μ)
    (h_indep : iIndepFun X μ)
    (hIcc : ∀ i, ∀ᵐ ω ∂μ, X i ω ∈ Set.Icc (0 : ℝ) 1)
    (hmean_sum : ∑ i ∈ s, μ[X i] ≤ (s.card : ℝ) * β) (hτ : 0 ≤ τ) :
    μ.real {ω | (s.card : ℝ) * (β + τ) ≤ ∑ i ∈ s, X i ω}
      ≤ Real.exp (-2 * (s.card : ℝ) * τ ^ 2) := by
  -- Centered reads.
  set Y : ι → Ω → ℝ := fun i ω => X i ω - μ[X i] with hYdef
  -- Independence of the centered reads.
  have hYindep : iIndepFun Y μ := by
    have h := h_indep.comp (fun i => fun x : ℝ => x - μ[X i])
      (fun i => measurable_id.sub_const _)
    exact h
  -- Each centered read is sub-Gaussian with constant 1/4.
  have hc4 : ((‖(1:ℝ) - 0‖₊) / 2) ^ 2 = (1 / 4 : ℝ≥0) := by
    norm_num
  have hsub : ∀ i ∈ s, HasSubgaussianMGF (Y i) (1 / 4 : ℝ≥0) μ := by
    intro i _
    have h := hasSubgaussianMGF_of_mem_Icc (hmeas i) (hIcc i)
    rwa [hc4] at h
  -- Hoeffding bound for the sum of the centered reads.
  have hε : (0:ℝ) ≤ (s.card:ℝ) * τ := by positivity
  have hmain :=
    HasSubgaussianMGF.measure_sum_ge_le_of_iIndepFun hYindep
      (c := fun _ => (1 / 4 : ℝ≥0)) (s := s) hsub hε
  -- Sum of the means is at most k·β (the hypothesis).
  have hsum_mean : ∑ i ∈ s, μ[X i] ≤ (s.card : ℝ) * β := hmean_sum
  -- Event inclusion.
  have hsubset :
      {ω | (s.card:ℝ) * (β + τ) ≤ ∑ i ∈ s, X i ω}
        ⊆ {ω | (s.card:ℝ) * τ ≤ ∑ i ∈ s, Y i ω} := by
    intro ω hω
    simp only [Set.mem_setOf_eq] at hω ⊢
    have hpush : ∑ i ∈ s, Y i ω
        = (∑ i ∈ s, X i ω) - ∑ i ∈ s, μ[X i] := by
      simp only [hYdef, Finset.sum_sub_distrib]
    rw [hpush]
    nlinarith [hω, hsum_mean]
  -- Monotonicity gives the bound on the ACCEPT event.
  have hmono := measureReal_mono (μ := μ) hsubset
  refine (hmono.trans hmain).trans_eq ?_
  -- Simplify the exponent.
  congr 1
  push_cast [Finset.sum_const, nsmul_eq_mul]
  rcases Finset.eq_empty_or_nonempty s with rfl | hs
  · simp
  · have hkne : (s.card:ℝ) ≠ 0 := Nat.cast_ne_zero.mpr (Finset.card_pos.mpr hs).ne'
    field_simp
    ring

end OrthoDFA

#print axioms OrthoDFA.wrongDecisive_le
