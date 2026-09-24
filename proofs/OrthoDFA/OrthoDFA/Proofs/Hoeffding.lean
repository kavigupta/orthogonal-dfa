import Mathlib.Probability.Moments.SubGaussian

/-! # Hoeffding's two tails for independent reads in `[0, 1]` -/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory
open scoped NNReal

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]

/-- Total mean at most `k·β`: the sum reaches `k(β + τ)` with probability at most
`exp(-2kτ²)`, where `k = #s`. -/
theorem sumUpper_le {ι : Type*}
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

/-- Total mean at least `k·b`: the sum falls to `k(b − γ)` with probability at most
`exp(-2kγ²)`, where `k = #idx`.  The upper tail, for `1 − X`. -/
theorem sumLower_le {ι : Type*} (X : ι → Ω → ℝ) (idx : Finset ι) (b γ : ℝ)
    (hmeas : ∀ i, AEMeasurable (X i) μ) (h_indep : iIndepFun X μ)
    (hIcc : ∀ i, ∀ᵐ ω ∂μ, X i ω ∈ Set.Icc (0 : ℝ) 1)
    (hmean : (idx.card : ℝ) * b ≤ ∑ i ∈ idx, μ[X i]) (hγ : 0 ≤ γ) :
    μ.real {ω | ∑ i ∈ idx, X i ω ≤ (idx.card : ℝ) * (b - γ)}
      ≤ Real.exp (-2 * (idx.card : ℝ) * γ ^ 2) := by
  set X' : ι → Ω → ℝ := fun i ω => 1 - X i ω with hX'
  have hmeas' : ∀ i, AEMeasurable (X' i) μ := fun i => (hmeas i).const_sub 1
  have hindep' : iIndepFun X' μ :=
    h_indep.comp (fun _ => fun x : ℝ => 1 - x) (fun _ => measurable_const.sub measurable_id)
  have hIcc' : ∀ i, ∀ᵐ ω ∂μ, X' i ω ∈ Set.Icc (0 : ℝ) 1 := by
    intro i; filter_upwards [hIcc i] with ω hω
    simp only [hX', Set.mem_Icc]; constructor <;> [linarith [hω.2]; linarith [hω.1]]
  have hint : ∀ i, Integrable (X i) μ := fun i =>
    MeasureTheory.Integrable.of_mem_Icc 0 1 (hmeas i) (hIcc i)
  have hmeanX' : ∀ i, μ[X' i] = 1 - μ[X i] := by
    intro i; simp only [hX']
    rw [integral_sub (integrable_const (1 : ℝ)) (hint i), integral_const]
    have huniv : μ.real Set.univ = 1 := by
      simp [MeasureTheory.measureReal_def, measure_univ]
    rw [huniv]; ring
  have hmean' : ∑ i ∈ idx, μ[X' i] ≤ (idx.card : ℝ) * (1 - b) := by
    have hrw : ∑ i ∈ idx, μ[X' i] = (idx.card : ℝ) - ∑ i ∈ idx, μ[X i] := by
      rw [Finset.sum_congr rfl (fun i _ => hmeanX' i), Finset.sum_sub_distrib,
        Finset.sum_const]
      simp [nsmul_eq_mul]
    rw [hrw]; nlinarith [hmean]
  have h := sumUpper_le X' idx (1 - b) γ hmeas' hindep' hIcc' hmean' hγ
  refine le_trans (le_of_eq ?_) h
  congr 1
  ext ω
  simp only [Set.mem_setOf_eq, hX', Finset.sum_sub_distrib, Finset.sum_const,
    nsmul_eq_mul, mul_one]
  constructor
  · intro hle; nlinarith [hle]
  · intro hge; nlinarith [hge]

end OrthoDFA
