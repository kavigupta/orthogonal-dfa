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


/-- `e^{-l} ≤ 1 − l + l²/2` for `l ≥ 0`. -/
lemma exp_neg_le_quadratic {l : ℝ} (hl : 0 ≤ l) : Real.exp (-l) ≤ 1 - l + l ^ 2 / 2 := by
  have h4 := Real.sum_le_exp_of_nonneg hl 4
  simp only [Finset.sum_range_succ, Finset.sum_range_zero, Nat.factorial] at h4
  norm_num at h4
  have hq : 0 < 1 - l + l ^ 2 / 2 := by nlinarith [sq_nonneg (l - 1)]
  have hprod : 1 ≤ (1 - l + l ^ 2 / 2) * Real.exp l := by
    have : 1 ≤ (1 - l + l ^ 2 / 2) * (1 + l + l ^ 2 / 2 + l ^ 3 / 6) := by
      nlinarith [pow_nonneg hl 3, pow_nonneg hl 4, pow_nonneg hl 5]
    refine le_trans this (mul_le_mul_of_nonneg_left (by linarith) hq.le)
  rw [Real.exp_neg, inv_eq_one_div, div_le_iff₀ (Real.exp_pos l)]
  linarith

/-- The lower tail relative to the mean.  A sum of independent reads in `[0, 1]` whose means add
to at least `ν` falls to `ν − t` with probability at most

    exp(−t²/(2ν)),

which is Hoeffding's `exp(−2t²/#s)` with the sample size replaced by the mean. -/
theorem sumLower_rel {ι : Type*} (X : ι → Ω → ℝ) (s : Finset ι) (ν t : ℝ)
    (hmeas : ∀ i, Measurable (X i)) (h_indep : iIndepFun X μ)
    (hIcc : ∀ i, ∀ᵐ ω ∂μ, X i ω ∈ Set.Icc (0 : ℝ) 1)
    (hν : 0 < ν) (hmean : ν ≤ ∑ i ∈ s, μ[X i]) (ht : 0 ≤ t) :
    μ.real {ω | ∑ i ∈ s, X i ω ≤ ν - t} ≤ Real.exp (-(t ^ 2 / (2 * ν))) := by
  set l : ℝ := t / ν with hl
  have hl0 : 0 ≤ l := div_nonneg ht hν.le
  set a : ℝ := 1 - Real.exp (-l) with ha
  have ha0 : 0 ≤ a := by
    rw [ha, sub_nonneg]; exact Real.exp_le_one_iff.2 (by linarith)
  have hint : ∀ i, Integrable (X i) μ := fun i =>
    MeasureTheory.Integrable.of_mem_Icc 0 1 (hmeas i).aemeasurable (hIcc i)
  have hexpint : ∀ i, Integrable (fun ω => Real.exp (-l * X i ω)) μ := by
    intro i
    refine MeasureTheory.Integrable.of_mem_Icc 0 1
      ((measurable_const.mul (hmeas i)).exp).aemeasurable ?_
    filter_upwards [hIcc i] with ω hω
    refine ⟨(Real.exp_pos _).le, Real.exp_le_one_iff.2 ?_⟩
    nlinarith [hω.1]
  -- one read's moment generating function at `−l`
  have hone : ∀ i, mgf (X i) μ (-l) ≤ Real.exp (-(a * μ[X i])) := by
    intro i
    have hpt : ∀ᵐ ω ∂μ, Real.exp (-l * X i ω) ≤ 1 - a * X i ω := by
      filter_upwards [hIcc i] with ω hω
      have h := convexOn_exp.2 (Set.mem_univ 0) (Set.mem_univ (-l))
        (by linarith [hω.2] : 0 ≤ 1 - X i ω) hω.1 (by ring : 1 - X i ω + X i ω = 1)
      simp only [smul_eq_mul, mul_zero, zero_add, Real.exp_zero, mul_one] at h
      rw [ha]
      calc Real.exp (-l * X i ω) = Real.exp (X i ω * -l) := by ring_nf
        _ ≤ 1 - X i ω + X i ω * Real.exp (-l) := h
        _ = 1 - (1 - Real.exp (-l)) * X i ω := by ring
    have hle : mgf (X i) μ (-l) ≤ 1 - a * μ[X i] := by
      unfold mgf
      calc μ[fun ω => Real.exp (-l * X i ω)] ≤ μ[fun ω => 1 - a * X i ω] :=
            integral_mono_ae (f := fun ω => Real.exp (-l * X i ω)) (g := fun ω => 1 - a * X i ω)
              (hexpint i) ((integrable_const 1).sub ((hint i).const_mul a)) hpt
        _ = 1 - a * μ[X i] := by
            rw [integral_sub (integrable_const 1) ((hint i).const_mul a), integral_const,
              integral_const_mul]
            simp
    refine le_trans hle ?_
    have := Real.add_one_le_exp (-(a * μ[X i]))
    linarith
  -- the sum's
  have hsum : mgf (∑ i ∈ s, X i) μ (-l) ≤ Real.exp (-(a * ν)) := by
    rw [h_indep.mgf_sum hmeas s]
    calc ∏ i ∈ s, mgf (X i) μ (-l) ≤ ∏ i ∈ s, Real.exp (-(a * μ[X i])) :=
          Finset.prod_le_prod (fun i _ => mgf_nonneg) (fun i _ => hone i)
      _ = Real.exp (-(a * ∑ i ∈ s, μ[X i])) := by
          rw [← Real.exp_sum, Finset.mul_sum, Finset.sum_neg_distrib]
      _ ≤ Real.exp (-(a * ν)) :=
          Real.exp_le_exp.2 (by nlinarith [mul_le_mul_of_nonneg_left hmean ha0])
  have hSint : Integrable (fun ω => Real.exp (-l * (∑ i ∈ s, X i) ω)) μ := by
    have hSm : Measurable (fun ω => Real.exp (-l * (∑ i ∈ s, X i) ω)) := by
      have : Measurable (∑ i ∈ s, X i) := by
        rw [show (∑ i ∈ s, X i) = fun ω => ∑ i ∈ s, X i ω from by ext ω; simp]
        exact Finset.measurable_sum s (fun i _ => hmeas i)
      fun_prop
    refine MeasureTheory.Integrable.of_mem_Icc 0 1 hSm.aemeasurable ?_
    have hall : ∀ᵐ ω ∂μ, ∀ i ∈ s, X i ω ∈ Set.Icc (0 : ℝ) 1 :=
      (Filter.eventually_all_finset s).2 (fun i _ => hIcc i)
    filter_upwards [hall] with ω hω
    refine ⟨(Real.exp_pos _).le, Real.exp_le_one_iff.2 ?_⟩
    have : 0 ≤ (∑ i ∈ s, X i) ω := by
      rw [Finset.sum_apply]; exact Finset.sum_nonneg (fun i hi => (hω i hi).1)
    nlinarith
  have hch := measure_le_le_exp_mul_mgf (μ := μ) (X := ∑ i ∈ s, X i) (ν - t)
    (by linarith : -l ≤ 0) hSint
  have hset : {ω | ∑ i ∈ s, X i ω ≤ ν - t} = {ω | (∑ i ∈ s, X i) ω ≤ ν - t} := by
    ext ω; simp [Finset.sum_apply]
  rw [hset]
  refine le_trans hch ?_
  calc Real.exp (- -l * (ν - t)) * mgf (∑ i ∈ s, X i) μ (-l)
      ≤ Real.exp (- -l * (ν - t)) * Real.exp (-(a * ν)) :=
        mul_le_mul_of_nonneg_left hsum (Real.exp_pos _).le
    _ = Real.exp (l * (ν - t) - a * ν) := by rw [← Real.exp_add]; ring_nf
    _ ≤ Real.exp (-(t ^ 2 / (2 * ν))) := by
        refine Real.exp_le_exp.2 ?_
        have hq := exp_neg_le_quadratic hl0
        have haq : l - l ^ 2 / 2 ≤ a := by rw [ha]; linarith
        have hlν : l * ν = t := by rw [hl]; field_simp
        have e : t ^ 2 / (2 * ν) = l ^ 2 * ν / 2 := by
          rw [← hlν]; field_simp
        rw [e]
        nlinarith [mul_le_mul_of_nonneg_right haq hν.le]

end OrthoDFA
