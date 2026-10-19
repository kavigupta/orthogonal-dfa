import OrthoDFA.Proofs.LoopLaw

/-!
# Drawing off a set

`offLaw ν T` is `ν` conditioned off `T`. A uniform below `1` draws from it a string it gives
positive mass (`drawFrom_pos`), so one outside `T` and in `ν`'s support. Where `ν (T) ≤ 1/2` it
at most doubles `ν`'s atoms, loses at most `ν (T)` of any set's mass, and an average under it
is within `ν (T)` of the average under `ν` (`integral_le_offLaw`).
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

/-- A uniform below `1` draws a string of positive mass. -/
lemma drawFrom_pos (ν : Measure (FreeMonoid α)) [IsProbabilityMeasure ν] {u : unitInterval}
    (hu : (u : ℝ) < 1) : 0 < ν.real {drawFrom ν u} := by
  rcases (drawFrom_eq_iff ν u (drawFrom ν u)).1 rfl with ⟨h1, h2⟩ | ⟨-, h⟩
  · rw [massBelow_succ] at h2
    have : {z : FreeMonoid α | strCode z = strCode (drawFrom ν u)} = {drawFrom ν u} := by
      ext z; simp [strCode_injective.eq_iff]
    rw [this] at h2
    linarith
  · obtain ⟨n, hn⟩ := exists_lt_massBelow ν (u := u) (by rw [probReal_univ]; exact hu)
    linarith [h n]

lemma offLaw_eq {ν : Measure (FreeMonoid α)} [IsFiniteMeasure ν] {T : Finset (FreeMonoid α)}
    (h0 : ν (↑T)ᶜ ≠ 0) : offLaw ν T = (ν (↑T)ᶜ)⁻¹ • ν.restrict (↑T)ᶜ := by
  have hu : ν.restrict (↑T)ᶜ Set.univ = ν (↑T)ᶜ := by
    rw [Measure.restrict_apply_univ]
  unfold offLaw popLaw
  rw [if_neg (by rw [hu]; exact not_or.2 ⟨h0, measure_ne_top _ _⟩), hu]

lemma offLaw_real {ν : Measure (FreeMonoid α)} [IsFiniteMeasure ν] {T : Finset (FreeMonoid α)}
    (h0 : ν (↑T)ᶜ ≠ 0) (A : Set (FreeMonoid α)) :
    (offLaw ν T).real A = ν.real (A ∩ (↑T)ᶜ) / ν.real (↑T)ᶜ := by
  rw [offLaw_eq h0, measureReal_def, Measure.smul_apply, smul_eq_mul, ENNReal.toReal_mul,
    ENNReal.toReal_inv, Measure.restrict_apply (measurableSet_of_countable A), div_eq_inv_mul]
  rfl

lemma compl_ne_zero {ν : Measure (FreeMonoid α)} [IsProbabilityMeasure ν]
    {T : Finset (FreeMonoid α)} (hT : ν.real T ≤ 1 / 2) : ν (↑T)ᶜ ≠ 0 := by
  intro h
  have : ν.real (↑T)ᶜ = 0 := by rw [measureReal_def, h, ENNReal.toReal_zero]
  have hc := probReal_compl_eq_one_sub (μ := ν) (measurableSet_of_countable (↑T : Set _))
  linarith

lemma half_le_compl {ν : Measure (FreeMonoid α)} [IsProbabilityMeasure ν]
    {T : Finset (FreeMonoid α)} (hT : ν.real T ≤ 1 / 2) : 1 / 2 ≤ ν.real (↑T)ᶜ := by
  have hc := probReal_compl_eq_one_sub (μ := ν) (measurableSet_of_countable (↑T : Set _))
  linarith

/-- A string `offLaw ν T` gives mass is outside `T`, and `ν` gives it mass. -/
lemma mem_of_offLaw_pos {ν : Measure (FreeMonoid α)} [IsFiniteMeasure ν]
    {T : Finset (FreeMonoid α)} (h0 : ν (↑T)ᶜ ≠ 0) {t : FreeMonoid α}
    (ht : 0 < (offLaw ν T).real {t}) : t ∉ T ∧ 0 < ν.real {t} := by
  rw [offLaw_real h0] at ht
  have hn : 0 < ν.real ({t} ∩ (↑T)ᶜ) := by
    by_contra h
    push Not at h
    have : ν.real ({t} ∩ (↑T)ᶜ) / ν.real (↑T)ᶜ ≤ 0 :=
      div_nonpos_of_nonpos_of_nonneg h measureReal_nonneg
    linarith
  refine ⟨fun htT => ?_, lt_of_lt_of_le hn (measureReal_mono Set.inter_subset_left)⟩
  have : ({t} ∩ (↑T)ᶜ : Set (FreeMonoid α)) = ∅ := by
    ext z
    simp only [Set.mem_inter_iff, Set.mem_singleton_iff, Set.mem_compl_iff, Finset.mem_coe,
      Set.mem_empty_iff_false, iff_false, not_and, not_not]
    rintro rfl
    exact htT
  rw [this, measureReal_empty] at hn
  exact lt_irrefl _ hn

/-- A uniform below `1` draws off `T`, a string `ν` gives mass. -/
lemma drawFrom_offLaw {ν : Measure (FreeMonoid α)} [IsProbabilityMeasure ν]
    {T : Finset (FreeMonoid α)} (h0 : ν (↑T)ᶜ ≠ 0) {u : unitInterval} (hu : (u : ℝ) < 1) :
    drawFrom (offLaw ν T) u ∉ T ∧ 0 < ν.real {drawFrom (offLaw ν T) u} :=
  mem_of_offLaw_pos h0 (drawFrom_pos _ hu)

lemma offLaw_null {ν : Measure (FreeMonoid α)} [IsFiniteMeasure ν] {T : Finset (FreeMonoid α)}
    (h0 : ν (↑T)ᶜ ≠ 0) {A : Set (FreeMonoid α)} (hA : ν A = 0) : offLaw ν T A = 0 := by
  rw [offLaw_eq h0, Measure.smul_apply, Measure.restrict_apply (measurableSet_of_countable A),
    measure_mono_null Set.inter_subset_left hA, smul_zero]

lemma offLaw_atom_le {ν : Measure (FreeMonoid α)} [IsProbabilityMeasure ν]
    {T : Finset (FreeMonoid α)} (hT : ν.real T ≤ 1 / 2) (t : FreeMonoid α) :
    (offLaw ν T).real {t} ≤ 2 * ν.real {t} := by
  rw [offLaw_real (compl_ne_zero hT), div_le_iff₀ (by linarith [half_le_compl hT])]
  have h1 := half_le_compl hT
  have h2 : ν.real ({t} ∩ (↑T)ᶜ) ≤ ν.real {t} := measureReal_mono Set.inter_subset_left
  nlinarith [measureReal_nonneg (μ := ν) (s := {t})]

lemma le_offLaw_real {ν : Measure (FreeMonoid α)} [IsProbabilityMeasure ν]
    {T : Finset (FreeMonoid α)} (hT : ν.real T ≤ 1 / 2) (A : Set (FreeMonoid α)) :
    ν.real A - ν.real T ≤ (offLaw ν T).real A := by
  rw [offLaw_real (compl_ne_zero hT), le_div_iff₀ (by linarith [half_le_compl hT])]
  have hA : ν.real A ≤ ν.real (A ∩ (↑T)ᶜ) + ν.real T := by
    refine (measureReal_mono (fun z hz => ?_)).trans (measureReal_union_le (A ∩ (↑T)ᶜ) (↑T))
    by_cases h : z ∈ (↑T : Set (FreeMonoid α))
    · exact Or.inr h
    · exact Or.inl ⟨hz, h⟩
  have h1 : ν.real (↑T)ᶜ ≤ 1 := measureReal_le_one
  have h3 : 0 ≤ ν.real (↑T)ᶜ := measureReal_nonneg
  rcases le_or_gt (ν.real A - ν.real T) 0 with h | h
  · nlinarith [measureReal_nonneg (μ := ν) (s := A ∩ (↑T)ᶜ)]
  · nlinarith

/-- An average of a `[0, 1]` score under `ν` is at most its average off `T` and `ν (T)`. -/
lemma integral_le_offLaw (ν : Measure (FreeMonoid α)) [IsProbabilityMeasure ν]
    (T : Finset (FreeMonoid α)) {f : FreeMonoid α → ℝ} (hf0 : ∀ p, 0 ≤ f p) (hf1 : ∀ p, f p ≤ 1) :
    ∫ p, f p ∂ν ≤ ∫ p, f p ∂(offLaw ν T) + ν.real T := by
  have hfm : Measurable f := measurable_of_countable f
  have hint : ∀ μ' : Measure (FreeMonoid α), IsFiniteMeasure μ' → Integrable f μ' :=
    fun μ' _ => Integrable.of_bound hfm.aestronglyMeasurable 1
      (Filter.Eventually.of_forall fun p => by
        rw [Real.norm_eq_abs, abs_of_nonneg (hf0 p)]; exact hf1 p)
  have hoff0 : 0 ≤ ∫ p, f p ∂(offLaw ν T) := integral_nonneg hf0
  have hTm : MeasurableSet (↑T : Set (FreeMonoid α)) := measurableSet_of_countable _
  rw [← integral_add_compl hTm (hint ν inferInstance)]
  have hT : ∫ p in ↑T, f p ∂ν ≤ ν.real T := by
    calc ∫ p in ↑T, f p ∂ν ≤ ∫ _p in ↑T, (1 : ℝ) ∂ν :=
          setIntegral_mono (hint _ inferInstance).integrableOn (integrable_const 1) hf1
      _ = ν.real T := by simp
  by_cases h0 : ν (↑T)ᶜ = 0
  · rw [Measure.restrict_zero_set h0, integral_zero_measure]
    linarith
  have hc : ∫ p in (↑T)ᶜ, f p ∂ν ≤ ∫ p, f p ∂(offLaw ν T) := by
    rw [offLaw_eq h0, integral_smul_measure, ENNReal.toReal_inv, smul_eq_mul]
    have hpos : 0 < (ν (↑T)ᶜ).toReal := ENNReal.toReal_pos h0 (measure_ne_top _ _)
    have hle : (ν (↑T)ᶜ).toReal ≤ 1 := by
      rw [← measureReal_def]; exact measureReal_le_one
    have hI : 0 ≤ ∫ p in (↑T)ᶜ, f p ∂ν := setIntegral_nonneg hTm.compl fun p _ => hf0 p
    rw [le_inv_mul_iff₀ hpos]
    nlinarith
  linarith

end OrthoDFA
