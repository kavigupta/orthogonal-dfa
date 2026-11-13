import RLStar.LowLevelClaims.BlockMiss
import Mathlib.MeasureTheory.Measure.Real

namespace OrthoDFA

open MeasureTheory

theorem blockMiss_holds : BlockMiss := by
  intro P S _ _ _ _ ν _ D _ b q A hA hq
  have hT : MeasurableSet {ps : P × S | ps.2 ∈ A ps.1} :=
    measurableSet_setOfPred.mpr
      (measurable_from_prod_countable_left fun s => measurableSet_setOfPred.mp (hA s))
  have hE : MeasurableSet {pw : P × (Fin b → S) | q ≤ D.real (A pw.1) ∧ ∀ i, pw.2 i ∉ A pw.1} := by
    simp only [Set.ofPred_and, Set.ofPred_forall]
    exact (measurableSet_le measurable_const
        ((measurable_measure_prodMk_left hT).ennreal_toReal.comp measurable_fst)).inter
      (MeasurableSet.iInter fun i =>
        (hT.preimage (measurable_fst.prodMk ((measurable_pi_apply i).comp measurable_snd))).compl)
  have h1q : 0 ≤ 1 - q := by linarith
  have hsec : ∀ p, (Measure.pi fun _ : Fin b => D)
      (Prod.mk p ⁻¹' {pw | q ≤ D.real (A pw.1) ∧ ∀ i, pw.2 i ∉ A pw.1})
        ≤ ENNReal.ofReal ((1 - q) ^ b) := by
    intro p
    by_cases h : q ≤ D.real (A p)
    · have hset : Prod.mk p ⁻¹' {pw : P × (Fin b → S) | q ≤ D.real (A pw.1) ∧ ∀ i, pw.2 i ∉ A pw.1}
          = Set.univ.pi fun _ => (A p)ᶜ := by
        ext w
        simp [h]
      rw [hset, Measure.pi_pi, Finset.prod_const, Finset.card_univ, Fintype.card_fin,
        ENNReal.ofReal_pow h1q]
      refine pow_le_pow_left₀ zero_le ?_ b
      rw [← ofReal_measureReal, probReal_compl_eq_one_sub (Set.to_countable _).measurableSet]
      exact ENNReal.ofReal_le_ofReal (by linarith)
    · have hset : Prod.mk p ⁻¹' {pw : P × (Fin b → S) | q ≤ D.real (A pw.1) ∧ ∀ i, pw.2 i ∉ A pw.1}
          = ∅ := by
        ext w
        simp [h]
      simp [hset]
  have hbound : (ν.prod (Measure.pi fun _ : Fin b => D))
      {pw | q ≤ D.real (A pw.1) ∧ ∀ i, pw.2 i ∉ A pw.1} ≤ ENNReal.ofReal ((1 - q) ^ b) := by
    rw [Measure.prod_apply hE]
    calc ∫⁻ p, (Measure.pi fun _ : Fin b => D)
          (Prod.mk p ⁻¹' {pw | q ≤ D.real (A pw.1) ∧ ∀ i, pw.2 i ∉ A pw.1}) ∂ν
        ≤ ∫⁻ _, ENNReal.ofReal ((1 - q) ^ b) ∂ν := lintegral_mono hsec
      _ = ENNReal.ofReal ((1 - q) ^ b) := by simp
  exact ENNReal.toReal_le_of_le_ofReal (pow_nonneg h1q b) hbound

end OrthoDFA
