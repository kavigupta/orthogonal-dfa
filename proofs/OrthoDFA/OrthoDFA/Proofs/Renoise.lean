import OrthoDFA.FamilyRead

/-!
# Re-noising an oracle on a set of strings

`O.renoise R` lives on two independent copies of the noise: it reads the first copy off `R` and
the second on it. It is an oracle with `O`'s language and rates, so every distributional claim
about oracles holds of it, and it agrees with `O` at the first copy off `R`.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
  {S : Type*} [MeasurableSpace S]

open scoped Classical in
/-- `O`'s noise from the first copy off `R` and from the second on it. -/
noncomputable def renoiseNoise (O : Oracle μ S) (R : Set S) (w : S) (p : Ω × Ω) : ℝ :=
  if w ∈ R then O.noise w p.2 else O.noise w p.1

omit [IsProbabilityMeasure μ] in
lemma measurable_renoiseNoise (O : Oracle μ S) (R : Set S) (w : S) :
    Measurable (renoiseNoise O R w) := by
  unfold renoiseNoise
  split_ifs
  · exact (O.noise_meas w).comp measurable_snd
  · exact (O.noise_meas w).comp measurable_fst

lemma renoise_indep (O : Oracle μ S) (R : Set S) :
    iIndepFun (renoiseNoise O R) (μ.prod μ) := by
  classical
  rw [iIndepFun_iff_measure_inter_preimage_eq_mul]
  intro T sets hsets
  have h0 := iIndepFun_iff_measure_inter_preimage_eq_mul.1 O.noise_indep
  set T₁ := T.filter (· ∉ R)
  set T₂ := T.filter (· ∈ R)
  have hset : (⋂ i ∈ T, renoiseNoise O R i ⁻¹' sets i)
      = (⋂ i ∈ T₁, O.noise i ⁻¹' sets i) ×ˢ (⋂ i ∈ T₂, O.noise i ⁻¹' sets i) := by
    ext p
    simp only [Set.mem_iInter, Set.mem_preimage, Set.mem_prod, T₁, T₂, Finset.mem_filter,
      renoiseNoise]
    constructor
    · intro h
      refine ⟨fun i ⟨hi, hR⟩ => ?_, fun i ⟨hi, hR⟩ => ?_⟩
      · have := h i hi; rwa [if_neg hR] at this
      · have := h i hi; rwa [if_pos hR] at this
    · rintro ⟨h1, h2⟩ i hi
      by_cases hR : i ∈ R
      · rw [if_pos hR]; exact h2 i ⟨hi, hR⟩
      · rw [if_neg hR]; exact h1 i ⟨hi, hR⟩
  have hone : ∀ i, (μ.prod μ) (renoiseNoise O R i ⁻¹' sets i) = μ (O.noise i ⁻¹' sets i) := by
    intro i
    by_cases hR : i ∈ R
    · have : renoiseNoise O R i ⁻¹' sets i = Set.univ ×ˢ (O.noise i ⁻¹' sets i) := by
        ext p; simp [renoiseNoise, hR]
      rw [this, Measure.prod_prod, measure_univ, one_mul]
    · have : renoiseNoise O R i ⁻¹' sets i = (O.noise i ⁻¹' sets i) ×ˢ Set.univ := by
        ext p; simp [renoiseNoise, hR]
      rw [this, Measure.prod_prod, measure_univ, mul_one]
  rw [hset, Measure.prod_prod, h0 T₁ (fun i hi => hsets i (Finset.mem_filter.1 hi).1),
    h0 T₂ (fun i hi => hsets i (Finset.mem_filter.1 hi).1)]
  rw [← Finset.prod_filter_not_mul_prod_filter (s := T) (p := (· ∈ R))]
  simp only [hone]
  rfl

lemma integral_renoiseNoise (O : Oracle μ S) (R : Set S) (w : S) :
    ∫ p, renoiseNoise O R w p ∂(μ.prod μ) = ∫ x, O.noise w x ∂μ := by
  classical
  by_cases hR : w ∈ R
  · simp only [renoiseNoise, hR, if_true]
    rw [integral_fun_snd, probReal_univ, one_smul]
  · simp only [renoiseNoise, hR, if_false]
    rw [integral_fun_fst, probReal_univ, one_smul]

/-- `O` with its noise on `R` read from a second, independent copy. -/
noncomputable def Oracle.renoise (O : Oracle μ S) (R : Set S) : Oracle (μ.prod μ) S where
  L := O.L
  L_meas := O.L_meas
  noise := renoiseNoise O R
  ηIn := O.ηIn
  ηOut := O.ηOut
  noise_meas := measurable_renoiseNoise O R
  noise_indep := renoise_indep O R
  noise_bit w := by
    classical
    by_cases hR : w ∈ R
    · have := (measurePreserving_snd (μ := μ) (ν := μ)).quasiMeasurePreserving.ae (O.noise_bit w)
      filter_upwards [this] with p hp
      simpa [renoiseNoise, hR] using hp
    · have := (measurePreserving_fst (μ := μ) (ν := μ)).quasiMeasurePreserving.ae (O.noise_bit w)
      filter_upwards [this] with p hp
      simpa [renoiseNoise, hR] using hp
  noise_mean_in w hw := by rw [integral_renoiseNoise]; exact O.noise_mean_in w hw
  noise_mean_out w hw := by rw [integral_renoiseNoise]; exact O.noise_mean_out w hw

lemma renoise_mq_of_notMem (O : Oracle μ S) {R : Set S} {w : S} (hw : w ∉ R) (p : Ω × Ω) :
    (O.renoise R).mq w p = O.mq w p.1 := by
  classical
  simp only [Oracle.mq, Oracle.label, Oracle.renoise, renoiseNoise, if_neg hw]

end OrthoDFA
