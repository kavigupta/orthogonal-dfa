import OrthoDFA.Round
import OrthoDFA.Proofs.Replay

namespace OrthoDFA

open MeasureTheory ProbabilityTheory
open scoped ENNReal

variable {α : Type*} [Fintype α] [DecidableEq α] {Ω : Type*} [MeasurableSpace Ω]
variable {μ : Measure Ω} {Q : Type*}

theorem withDensity_real_eq {β : Type*} [MeasurableSpace β] (X : Measure β) [IsFiniteMeasure X]
    (g : β → ℝ) (hm : Measurable g) (h0 : ∀ x, 0 ≤ g x) {S : Set β} (hS : MeasurableSet S) :
    (X.withDensity fun x => ENNReal.ofReal (g x)).real S = ∫ x in S, g x ∂X := by
  rw [measureReal_def, withDensity_apply _ hS,
    integral_eq_lintegral_of_nonneg_ae (Filter.Eventually.of_forall h0) hm.aestronglyMeasurable]

theorem stateIndecision_nonneg (A : DFA (FreeMonoid α) Q) (O : Oracle μ (FreeMonoid α))
    (B : State) (F : Finset (FreeMonoid α)) (q : Q) : 0 ≤ stateIndecision A O B F q :=
  Real.sSup_nonneg fun _ ⟨_, _, h⟩ => h ▸ measureReal_nonneg

theorem stateIndecision_le_one [IsProbabilityMeasure μ] (A : DFA (FreeMonoid α) Q)
    (O : Oracle μ (FreeMonoid α)) (B : State) (F : Finset (FreeMonoid α)) (q : Q) :
    stateIndecision A O B F q ≤ 1 :=
  Real.sSup_le (fun _ ⟨_, _, h⟩ => h ▸ measureReal_le_one) zero_le_one

theorem anchorLaw_singleton {L k : ℕ} (hk : k < L) : anchorLaw L {k} = (L : ℝ≥0∞)⁻¹ := by
  simp only [anchorLaw, Measure.smul_apply, Measure.coe_finsetSum, Finset.sum_apply,
    Measure.dirac_apply, Set.indicator_apply, Set.mem_singleton_iff, Pi.one_apply, smul_eq_mul]
  rw [Finset.sum_ite_eq' (Finset.range L) k fun _ => (1 : ℝ≥0∞), if_pos (Finset.mem_range.2 hk),
    mul_one]

/-- A string `x` shorter than `L` that `D` begins with is in the `ν` root with chance at least
`ν(x) = D(x is a prefix) / L`. -/
theorem nuRoot_pos (D : Measure (FreeMonoid α)) [IsFiniteMeasure D] {L : ℕ} {x : FreeMonoid α}
    (hx : x.toList.length < L) (hD : 0 < D.real {p | x.toList <+: p.toList})
    {S : Set (FreeMonoid α)} (hS : x ∈ S) : 0 < (nuRoot D L).real S := by
  have hm : Measurable fun q : FreeMonoid α × ℕ => prefixOf q.1 q.2 := measurable_of_countable _
  have hsub : {p : FreeMonoid α | x.toList <+: p.toList} ×ˢ ({x.toList.length} : Set ℕ)
      ⊆ (fun q : FreeMonoid α × ℕ => prefixOf q.1 q.2) ⁻¹' S := by
    rintro ⟨p, k⟩ ⟨hp, hk⟩
    simp only [Set.mem_singleton_iff] at hk
    subst hk
    obtain ⟨t, ht⟩ := hp
    show prefixOf p x.toList.length ∈ S
    have : prefixOf p x.toList.length = x := by
      simp [prefixOf, ← ht]
    rwa [this]
  have hpos : 0 < (D.prod (anchorLaw L))
      ({p : FreeMonoid α | x.toList <+: p.toList} ×ˢ ({x.toList.length} : Set ℕ)) := by
    rw [Measure.prod_prod, anchorLaw_singleton hx]
    refine ENNReal.mul_pos ?_ (ENNReal.inv_ne_zero.2 (ENNReal.natCast_ne_top L))
    intro h0
    rw [measureReal_def, h0, ENNReal.toReal_zero] at hD
    exact lt_irrefl _ hD
  rw [measureReal_def, nuRoot, Measure.map_apply hm (Set.to_countable S).measurableSet]
  exact ENNReal.toReal_pos (lt_of_lt_of_le hpos (measure_mono hsub)).ne'
    (measure_ne_top _ _)

omit [Fintype α] [DecidableEq α] in
/-- A chain filtered by any chance `g` advances by `β` once its source holds strings with
`g ≥ hi` and every other string has `g ≤ hi / β`. -/
theorem chainAdvancesBy_of_mass (X : Measure (FreeMonoid α)) [IsFiniteMeasure X]
    (g : FreeMonoid α → ℝ) (h0 : ∀ x, 0 ≤ g x) (h1 : ∀ x, g x ≤ 1) {β hi : ℝ}
    (hgap : ∀ x, g x ≤ hi / β ∨ hi ≤ g x) (hpos : 0 < X.real {x | hi ≤ g x}) :
    ChainAdvancesBy X g β hi := by
  have hm : Measurable g := measurable_from_top
  have hbad : MeasurableSet {x | hi ≤ g x} := MeasurableSpace.measurableSet_top
  have hint : Integrable g X :=
    Integrable.of_bound hm.aestronglyMeasurable 1 (Filter.Eventually.of_forall fun x => by
      rw [Real.norm_eq_abs, abs_of_nonneg (h0 x)]; exact h1 x)
  refine ⟨hpos, ?_, ?_⟩
  · rw [withDensity_real_eq X g hm h0 hbad]
    have := setIntegral_mono_on (integrableOn_const (measure_ne_top _ _)) hint.integrableOn hbad
      fun x hx => hx
    simpa [setIntegral_const, smul_eq_mul, mul_comm] using this
  · rw [withDensity_real_eq X g hm h0 hbad.compl]
    have := setIntegral_mono_on hint.integrableOn (integrableOn_const (measure_ne_top _ _))
      hbad.compl fun x hx => (hgap x).resolve_right hx
    simpa [setIntegral_const, smul_eq_mul, mul_comm] using this

theorem edgeDisagreeProb_nonneg (O : Oracle μ (FreeMonoid α)) (B : State)
    (F : Finset (FreeMonoid α)) (H : Hypothesis α) (x : FreeMonoid α) (c : α) :
    0 ≤ edgeDisagreeProb O B F H x c := measureReal_nonneg

theorem edgeDisagreeProb_le_one [IsProbabilityMeasure μ] (O : Oracle μ (FreeMonoid α))
    (B : State) (F : Finset (FreeMonoid α)) (H : Hypothesis α) (x : FreeMonoid α) (c : α) :
    edgeDisagreeProb O B F H x c ≤ 1 := measureReal_le_one

end OrthoDFA
