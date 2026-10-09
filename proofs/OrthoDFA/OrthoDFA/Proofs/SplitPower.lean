import OrthoDFA.Proofs.Hoeffding

/-! # The split test's power on two blocks of independent reads -/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory Real
open scoped NNReal

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]

/-- With `H = 2|a||b|/(|a| + |b|)`, the statistic `H·(mean over a − mean over b)²` of independent
reads in `[0, 1]`, of mean at least `μa` over `a` and at most `μb` over `b`, falls short of `θ`
with chance at most `exp(−(√H·(μa − μb) − √θ)²)` once `√θ ≤ √H·(μa − μb)`. -/
theorem two_sample_power {ι : Type*} (X : ι → Ω → ℝ) (a b : Finset ι)
    (μa μb θ : ℝ) (hmeas : ∀ i, AEMeasurable (X i) μ) (h_indep : iIndepFun X μ)
    (hIcc : ∀ i, ∀ᵐ ω ∂μ, X i ω ∈ Set.Icc (0 : ℝ) 1) (hab : Disjoint a b)
    (ha : a.Nonempty) (hb : b.Nonempty)
    (hμa : (a.card : ℝ) * μa ≤ ∑ i ∈ a, μ[X i]) (hμb : ∑ i ∈ b, μ[X i] ≤ (b.card : ℝ) * μb)
    (hgap : √θ ≤ √(2 * a.card * b.card / (a.card + b.card)) * (μa - μb)) :
    μ.real {ω | 2 * a.card * b.card / (a.card + b.card)
        * ((∑ i ∈ a, X i ω) / a.card - (∑ i ∈ b, X i ω) / b.card) ^ 2 < θ}
      ≤ exp (-(√(2 * a.card * b.card / (a.card + b.card)) * (μa - μb) - √θ) ^ 2) := by
  classical
  set na : ℝ := (a.card : ℝ) with hna_def
  set nb : ℝ := (b.card : ℝ) with hnb_def
  have hna : 0 < na := by simp [na, ha.card_pos]
  have hnb : 0 < nb := by simp [nb, hb.card_pos]
  set H : ℝ := 2 * na * nb / (na + nb) with hH_def
  have hH : 0 < H := by positivity
  set G := √H with hG_def
  have hG : 0 < G := Real.sqrt_pos.2 hH
  have hGG : G * G = H := Real.mul_self_sqrt hH.le
  set ε := (μa - μb) - √θ / G with hε_def
  have hε : 0 ≤ ε := by
    have : √θ / G ≤ μa - μb := (div_le_iff₀ hG).2 (by linarith [hgap])
    linarith
  set w : ι → ℝ := fun i => if i ∈ a then -1 / na else 1 / nb with hw
  set Y : ι → Ω → ℝ := fun i ω => w i * (X i ω - μ[X i]) with hY
  have hYi : iIndepFun Y μ := by
    have h := h_indep.comp (fun i => fun x : ℝ => w i * (x - μ[X i]))
      (fun i => (measurable_id.sub_const _).const_mul _)
    exact h
  have hc4 : ((‖(1:ℝ) - 0‖₊) / 2) ^ 2 = (1 / 4 : ℝ≥0) := by norm_num
  set c : ι → ℝ≥0 := fun i => ⟨w i ^ 2, sq_nonneg _⟩ * (1 / 4 : ℝ≥0) with hc
  have hsub : ∀ i ∈ a ∪ b, HasSubgaussianMGF (Y i) (c i) μ := by
    intro i _
    have h := hasSubgaussianMGF_of_mem_Icc (hmeas i) (hIcc i)
    rw [hc4] at h
    exact h.const_mul (w i)
  have hmain := HasSubgaussianMGF.measure_sum_ge_le_of_iIndepFun hYi hsub hε
  have hsum : ∀ (g : ι → ℝ), ∑ i ∈ a ∪ b, w i * g i
      = -(∑ i ∈ a, g i) / na + (∑ i ∈ b, g i) / nb := by
    intro g
    rw [Finset.sum_union hab]
    have h1 : ∑ i ∈ a, w i * g i = -((∑ i ∈ a, g i) / na) := by
      rw [Finset.sum_div, ← Finset.sum_neg_distrib]
      refine Finset.sum_congr rfl fun i hi => ?_
      simp only [w, if_pos hi]; ring
    have h2 : ∑ i ∈ b, w i * g i = (∑ i ∈ b, g i) / nb := by
      rw [Finset.sum_div]
      refine Finset.sum_congr rfl fun i hi => ?_
      simp only [w, if_neg (Finset.disjoint_right.1 hab hi)]; ring
    rw [h1, h2, neg_div]
  have hcsum : ((∑ i ∈ a ∪ b, c i : ℝ≥0) : ℝ) = 1 / (2 * H) := by
    have hci : ∀ i, (c i : ℝ) = w i * (w i / 4) := fun i => by
      change w i ^ 2 * ((1 / 4 : ℝ≥0) : ℝ) = _
      push_cast; ring
    rw [NNReal.coe_sum]
    simp only [hci]
    rw [hsum fun i => w i / 4]
    have ha' : ∑ i ∈ a, w i / 4 = -na / (4 * na) := by
      rw [Finset.sum_congr rfl (g := fun _ => -1 / na / 4) fun i hi => by simp only [w, if_pos hi]]
      simp only [Finset.sum_const, nsmul_eq_mul, ← hna_def]
      field_simp
    have hb' : ∑ i ∈ b, w i / 4 = nb / (4 * nb) := by
      rw [Finset.sum_congr rfl (g := fun _ => 1 / nb / 4) fun i hi => by
        simp only [w, if_neg (Finset.disjoint_right.1 hab hi)]]
      simp only [Finset.sum_const, nsmul_eq_mul, ← hnb_def]
      field_simp
    rw [ha', hb', hH_def]
    field_simp
    ring
  have hsubset : {ω | H * ((∑ i ∈ a, X i ω) / na - (∑ i ∈ b, X i ω) / nb) ^ 2 < θ}
      ⊆ {ω | ε ≤ ∑ i ∈ a ∪ b, Y i ω} := by
    intro ω hω
    simp only [Set.mem_ofPred_eq] at hω ⊢
    set Δ := (∑ i ∈ a, X i ω) / na - (∑ i ∈ b, X i ω) / nb
    have hθ0 : 0 ≤ θ := le_trans (by positivity) hω.le
    have hlt : G * Δ < √θ := by
      calc G * Δ ≤ |G * Δ| := le_abs_self _
        _ = √(H * Δ ^ 2) := by
          rw [Real.sqrt_mul hH.le, Real.sqrt_sq_eq_abs, abs_mul, abs_of_pos hG]
        _ < √θ := Real.sqrt_lt_sqrt (by positivity) hω
    have hΔ : Δ < √θ / G := (lt_div_iff₀ hG).2 (by linarith)
    have hY' : ∑ i ∈ a ∪ b, Y i ω
        = -(∑ i ∈ a, (X i ω - μ[X i])) / na + (∑ i ∈ b, (X i ω - μ[X i])) / nb :=
      hsum fun i => X i ω - μ[X i]
    rw [hY', Finset.sum_sub_distrib, Finset.sum_sub_distrib]
    have e1 : μa ≤ (∑ i ∈ a, μ[X i]) / na := (le_div_iff₀ hna).2 (by linarith)
    have e2 : (∑ i ∈ b, μ[X i]) / nb ≤ μb := (div_le_iff₀ hnb).2 (by linarith)
    have : -((∑ i ∈ a, X i ω) - ∑ i ∈ a, μ[X i]) / na
        + ((∑ i ∈ b, X i ω) - ∑ i ∈ b, μ[X i]) / nb
        = (∑ i ∈ a, μ[X i]) / na - (∑ i ∈ b, μ[X i]) / nb - Δ := by
      simp only [Δ]; ring
    rw [this]
    simp only [ε]
    linarith
  refine (measureReal_mono hsubset).trans (hmain.trans_eq ?_)
  rw [hcsum]
  congr 1
  have : G * (μa - μb) - √θ = G * ε := by
    simp only [ε]; field_simp
  rw [this, mul_pow, sq G, hGG]
  field_simp

end OrthoDFA
