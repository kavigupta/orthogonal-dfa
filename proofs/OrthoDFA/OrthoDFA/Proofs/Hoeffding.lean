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

lemma exp_le_cubic {l : ℝ} (h0 : 0 ≤ l) (h1 : l ≤ 1) :
    Real.exp l ≤ 1 + l + l ^ 2 / 2 + 2 * l ^ 3 / 9 := by
  have h := Real.exp_bound' h0 h1 (n := 3) (by norm_num)
  simp only [Finset.sum_range_succ, Finset.sum_range_zero, Nat.factorial] at h
  norm_num at h
  linarith

/-- The upper tail relative to the mean.  For independent reads in `[0, 1]` whose means add to
at most `ν`, and `t ≥ 0`,

    P(Σ X ≥ ν + t) ≤ exp(−t²/(2(ν + t))). -/
theorem sumUpper_rel {ι : Type*} (X : ι → Ω → ℝ) (s : Finset ι) (ν t : ℝ)
    (hmeas : ∀ i, Measurable (X i)) (h_indep : iIndepFun X μ)
    (hIcc : ∀ i, ∀ᵐ ω ∂μ, X i ω ∈ Set.Icc (0 : ℝ) 1)
    (hν : 0 ≤ ν) (hmean : ∑ i ∈ s, μ[X i] ≤ ν) (ht : 0 ≤ t) :
    μ.real {ω | ν + t ≤ ∑ i ∈ s, X i ω} ≤ Real.exp (-(t ^ 2 / (2 * (ν + t)))) := by
  rcases eq_or_lt_of_le ht with ht0 | htpos
  · rw [← ht0]; simp only [ne_eq, OfNat.ofNat_ne_zero, not_false_eq_true, zero_pow, zero_div,
      neg_zero, Real.exp_zero]
    exact measureReal_le_one
  set w : ℝ := ν + t with hw
  have hwpos : 0 < w := by rw [hw]; linarith
  set l : ℝ := t / w with hl
  have hl0 : 0 ≤ l := div_nonneg ht hwpos.le
  have hl1 : l ≤ 1 := by rw [hl, div_le_one hwpos]; linarith
  have htl : t = l * w := by rw [hl]; field_simp
  have hνl : ν = (1 - l) * w := by rw [sub_mul, one_mul, ← htl, hw]; ring
  set a : ℝ := Real.exp l - 1 with ha
  have ha0 : 0 ≤ a := by rw [ha, sub_nonneg]; exact Real.one_le_exp hl0
  have hint : ∀ i, Integrable (X i) μ := fun i =>
    MeasureTheory.Integrable.of_mem_Icc 0 1 (hmeas i).aemeasurable (hIcc i)
  have hexpint : ∀ i, Integrable (fun ω => Real.exp (l * X i ω)) μ := by
    intro i
    refine MeasureTheory.Integrable.of_mem_Icc 0 (Real.exp l)
      ((measurable_const.mul (hmeas i)).exp).aemeasurable ?_
    filter_upwards [hIcc i] with ω hω
    exact ⟨(Real.exp_pos _).le, Real.exp_le_exp.2 (by nlinarith [hω.2])⟩
  have hone : ∀ i, mgf (X i) μ l ≤ Real.exp (a * μ[X i]) := by
    intro i
    have hpt : ∀ᵐ ω ∂μ, Real.exp (l * X i ω) ≤ 1 + a * X i ω := by
      filter_upwards [hIcc i] with ω hω
      have h := convexOn_exp.2 (Set.mem_univ 0) (Set.mem_univ l)
        (by linarith [hω.2] : 0 ≤ 1 - X i ω) hω.1 (by ring : 1 - X i ω + X i ω = 1)
      simp only [smul_eq_mul, mul_zero, zero_add, Real.exp_zero, mul_one] at h
      rw [ha]
      calc Real.exp (l * X i ω) = Real.exp (X i ω * l) := by ring_nf
        _ ≤ 1 - X i ω + X i ω * Real.exp l := h
        _ = 1 + (Real.exp l - 1) * X i ω := by ring
    have hle : mgf (X i) μ l ≤ 1 + a * μ[X i] := by
      unfold mgf
      calc μ[fun ω => Real.exp (l * X i ω)] ≤ μ[fun ω => 1 + a * X i ω] :=
            integral_mono_ae (f := fun ω => Real.exp (l * X i ω)) (g := fun ω => 1 + a * X i ω)
              (hexpint i) ((integrable_const 1).add ((hint i).const_mul a)) hpt
        _ = 1 + a * μ[X i] := by
            rw [integral_add (integrable_const 1) ((hint i).const_mul a), integral_const,
              integral_const_mul]
            simp
    refine le_trans hle ?_
    have := Real.add_one_le_exp (a * μ[X i])
    linarith
  have hsum : mgf (∑ i ∈ s, X i) μ l ≤ Real.exp (a * ν) := by
    rw [h_indep.mgf_sum hmeas s]
    calc ∏ i ∈ s, mgf (X i) μ l ≤ ∏ i ∈ s, Real.exp (a * μ[X i]) :=
          Finset.prod_le_prod (fun i _ => mgf_nonneg) (fun i _ => hone i)
      _ = Real.exp (a * ∑ i ∈ s, μ[X i]) := by
          rw [← Real.exp_sum, Finset.mul_sum]
      _ ≤ Real.exp (a * ν) :=
          Real.exp_le_exp.2 (mul_le_mul_of_nonneg_left hmean ha0)
  have hSint : Integrable (fun ω => Real.exp (l * (∑ i ∈ s, X i) ω)) μ := by
    have hSm : Measurable (fun ω => Real.exp (l * (∑ i ∈ s, X i) ω)) := by
      have : Measurable (∑ i ∈ s, X i) := by
        rw [show (∑ i ∈ s, X i) = fun ω => ∑ i ∈ s, X i ω from by ext ω; simp]
        exact Finset.measurable_sum s (fun i _ => hmeas i)
      fun_prop
    have hall : ∀ᵐ ω ∂μ, ∀ i ∈ s, X i ω ∈ Set.Icc (0 : ℝ) 1 :=
      (Filter.eventually_all_finset s).2 (fun i _ => hIcc i)
    refine MeasureTheory.Integrable.of_mem_Icc 0 (Real.exp (l * s.card)) hSm.aemeasurable ?_
    filter_upwards [hall] with ω hω
    refine ⟨(Real.exp_pos _).le, Real.exp_le_exp.2 ?_⟩
    have : (∑ i ∈ s, X i) ω ≤ s.card := by
      rw [Finset.sum_apply]
      calc ∑ i ∈ s, X i ω ≤ ∑ _i ∈ s, (1 : ℝ) := Finset.sum_le_sum (fun i hi => (hω i hi).2)
        _ = s.card := by simp
    exact mul_le_mul_of_nonneg_left this hl0
  have hch := measure_ge_le_exp_mul_mgf (μ := μ) (X := ∑ i ∈ s, X i) w hl0 hSint
  have hset : {ω | w ≤ ∑ i ∈ s, X i ω} = {ω | w ≤ (∑ i ∈ s, X i) ω} := by
    ext ω; simp [Finset.sum_apply]
  rw [hset]
  refine le_trans hch ?_
  calc Real.exp (-l * w) * mgf (∑ i ∈ s, X i) μ l
      ≤ Real.exp (-l * w) * Real.exp (a * ν) :=
        mul_le_mul_of_nonneg_left hsum (Real.exp_pos _).le
    _ = Real.exp (-l * w + a * ν) := by rw [← Real.exp_add]
    _ ≤ Real.exp (-(t ^ 2 / (2 * w))) := by
        refine Real.exp_le_exp.2 ?_
        have hq := exp_le_cubic hl0 hl1
        have haq : a ≤ l + l ^ 2 / 2 + 2 * l ^ 3 / 9 := by rw [ha]; linarith
        have e : t ^ 2 / (2 * w) = l ^ 2 * w / 2 := by
          rw [htl]; field_simp
        rw [e]
        have h1 := mul_le_mul_of_nonneg_right haq hν
        rw [hνl] at h1 ⊢
        have hw3 : 0 ≤ l ^ 3 * w := by positivity
        have hw4 : 0 ≤ l ^ 4 * w := by positivity
        nlinarith [h1, hw3, hw4]

/-- The exponent rate of a count of independent events, each at chance at most `E`, reaching
`l` of its size. -/
noncomputable def relRate (l E : ℝ) : ℝ := (l - E) ^ 2 / (2 * l)

/-- `sumUpper_rel` for a count: independent reads in `[0, 1]` whose means average at most `E`
reach `l` of their number `n` with probability at most `exp(−n·relRate l E)`. -/
theorem sumUpper_count_rel {ι : Type*} (X : ι → Ω → ℝ) (s : Finset ι) (E l : ℝ)
    (hmeas : ∀ i, Measurable (X i)) (h_indep : iIndepFun X μ)
    (hIcc : ∀ i, ∀ᵐ ω ∂μ, X i ω ∈ Set.Icc (0 : ℝ) 1)
    (hE : 0 ≤ E) (hEl : E ≤ l) (hmean : ∑ i ∈ s, μ[X i] ≤ (s.card : ℝ) * E) :
    μ.real {ω | (s.card : ℝ) * l ≤ ∑ i ∈ s, X i ω}
      ≤ Real.exp (-((s.card : ℝ) * relRate l E)) := by
  rcases Nat.eq_zero_or_pos s.card with hc | hc
  · simp only [hc, Nat.cast_zero, zero_mul, neg_zero, Real.exp_zero]
    exact measureReal_le_one
  have hcR : (0 : ℝ) < (s.card : ℝ) := by exact_mod_cast hc
  rcases eq_or_lt_of_le (le_trans hE hEl) with hl0 | hlpos
  · have hr : relRate l E = 0 := by unfold relRate; rw [← hl0]; simp
    rw [hr, mul_zero, neg_zero, Real.exp_zero]
    exact measureReal_le_one
  have h := sumUpper_rel X s ((s.card : ℝ) * E) ((s.card : ℝ) * (l - E)) hmeas h_indep hIcc
    (by positivity) hmean (mul_nonneg hcR.le (by linarith))
  have hset : {ω | (s.card : ℝ) * l ≤ ∑ i ∈ s, X i ω}
      = {ω | (s.card : ℝ) * E + (s.card : ℝ) * (l - E) ≤ ∑ i ∈ s, X i ω} := by
    ext ω; simp only [Set.mem_setOf_eq]; ring_nf
  rw [hset]
  refine le_trans h (le_of_eq ?_)
  congr 1
  unfold relRate
  rw [show (s.card : ℝ) * E + (s.card : ℝ) * (l - E) = (s.card : ℝ) * l by ring]
  field_simp

/-- `relRate` grows with the limit and falls with the chance. -/
lemma relRate_anti {l l' E E' : ℝ} (hE : 0 ≤ E) (hEE : E ≤ E') (hE'l' : E' ≤ l') (hl : l' ≤ l) :
    relRate l' E' ≤ relRate l E := by
  unfold relRate
  have hE'0 : 0 ≤ E' := le_trans hE hEE
  rcases eq_or_lt_of_le (le_trans hE'0 hE'l') with h0 | hpos
  · rw [← h0]; simp only [mul_zero, div_zero]
    exact div_nonneg (sq_nonneg _) (by linarith)
  have hlpos : 0 < l := lt_of_lt_of_le hpos hl
  have h1 : (l' - E') ^ 2 / (2 * l') ≤ (l - E') ^ 2 / (2 * l) := by
    rw [div_le_div_iff₀ (by linarith) (by linarith)]
    have ha0 : 0 ≤ l' - E' := by linarith
    have hab : l' - E' ≤ l - E' := by linarith
    have hb : l = (l - E') + E' := by ring
    have hb' : l' = (l' - E') + E' := by ring
    nlinarith [mul_nonneg (mul_nonneg ha0 (by linarith : (0 : ℝ) ≤ l - E'))
      (by linarith : (0 : ℝ) ≤ (l - E') - (l' - E')),
      mul_le_mul_of_nonneg_right (pow_le_pow_left₀ ha0 hab 2) hE'0]
  refine le_trans h1 (div_le_div_of_nonneg_right ?_ (by linarith))
  exact pow_le_pow_left₀ (by linarith) (by linarith) 2

/-- A count's excess `t` over a larger mean is the less likely one. -/
lemma relRate_shift_anti {E E' t : ℝ} (hE : 0 ≤ E) (hEE : E ≤ E') (ht : 0 ≤ t) :
    relRate (E' + t) E' ≤ relRate (E + t) E := by
  unfold relRate
  rw [show E' + t - E' = t by ring, show E + t - E = t by ring]
  rcases eq_or_lt_of_le ht with h0 | hpos
  · rw [← h0]; simp
  exact div_le_div_of_nonneg_left (sq_nonneg _) (by positivity) (by linarith)

end OrthoDFA
