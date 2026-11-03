import OrthoDFA.Proofs.RandomHarvest
import OrthoDFA.Proofs.BinomLaw
import OrthoDFA.Proofs.Hoeffding

/-! The `m`-draw harvest law concentrates to its ideal.

Over `m` independent probes, the sample's mass on a set is the number of probes that harvest into
it, so it is `Bin(m, harvestLaw A)`; `pi_count_ge` reads off the exact binomial tails and the
`Hoeffding` bounds turn them into `slack`.
-/

namespace OrthoDFA

namespace Random

open MeasureTheory

open scoped ENNReal

variable {α : Type*} [Fintype α] (read : FreeMonoid α → ARU) (D : Measure (FreeMonoid α)) (k : ℕ)
  (H : HypothesisDFA α)

open scoped Classical in
/-- The sample's mass on `A` is the number of its probes meeting `A`. -/
theorem harvestSample_apply_card (site : Level3Site) (A : Set (FreeMonoid α))
    {m : ℕ} (xs : Fin m → FreeMonoid α) :
    harvestSample read k H site xs A
      = ((Finset.univ.filter fun i => siteHit read k H site A (xs i)).card : ℝ≥0∞) := by
  rw [harvestSample_apply read k H site xs A]
  simp only [siteHarvest_filter_length read k H site A, Nat.cast_ite, Nat.cast_one, Nat.cast_zero]
  rw [Finset.sum_boole]

open scoped Classical in
/-- A probe drawn from `D` meets `A` exactly as often as the ideal law puts mass there. -/
theorem siteHit_prob [IsProbabilityMeasure D] (site : Level3Site) (A : Set (FreeMonoid α)) :
    D.real {x | siteHit read k H site A x} = (harvestLaw read D k H site).real A := by
  rw [harvestLaw_real read D k H site A]
  have hfun : (fun x => (((siteHarvest read k H site x).filter (· ∈ A)).length : ℝ))
      = {x | siteHit read k H site A x}.indicator 1 := by
    funext x
    rw [siteHarvest_filter_length read k H site A x]
    by_cases h : siteHit read k H site A x <;> simp [Set.indicator, h]
  rw [hfun, integral_indicator_one (Set.to_countable _).measurableSet]

open scoped Classical in
/-- The chance the sample finds at least `j` of its mass in `A` is `Bin(m, harvestLaw A)`'s. -/
theorem harvestSample_upper_tail [IsProbabilityMeasure D] (site : Level3Site)
    (A : Set (FreeMonoid α)) (m j : ℕ) :
    (Measure.pi fun _ : Fin m => D).real
        {xs | (j : ℝ≥0∞) ≤ harvestSample read k H site xs A}
      = binomSfGe m ((harvestLaw read D k H site).real A) j := by
  have hset : {xs : Fin m → FreeMonoid α | (j : ℝ≥0∞) ≤ harvestSample read k H site xs A}
      = {xs | j ≤ (Finset.univ.filter fun i => siteHit read k H site A (xs i)).card} := by
    ext xs
    simp only [Set.mem_ofPred_eq]
    rw [harvestSample_apply_card read k H site A xs, Nat.cast_le]
  rw [hset, pi_count_ge D (fun x => siteHit read k H site A x) Finset.univ j,
    siteHit_prob read D k H site A, Finset.card_univ, Fintype.card_fin]

open scoped Classical in
/-- The chance the sample finds at most `b` of its mass in `A` is `Bin(m, harvestLaw A)`'s lower
tail. -/
theorem harvestSample_lower_tail [IsProbabilityMeasure D] (site : Level3Site)
    (A : Set (FreeMonoid α)) (m b : ℕ) :
    (Measure.pi fun _ : Fin m => D).real
        {xs | harvestSample read k H site xs A ≤ (b : ℝ≥0∞)}
      = 1 - binomSfGe m ((harvestLaw read D k H site).real A) (b + 1) := by
  have hcompl : {xs : Fin m → FreeMonoid α | harvestSample read k H site xs A ≤ (b : ℝ≥0∞)}ᶜ
      = {xs | ((b + 1 : ℕ) : ℝ≥0∞) ≤ harvestSample read k H site xs A} := by
    ext xs
    simp only [Set.mem_compl_iff, Set.mem_ofPred_eq, not_le]
    rw [harvestSample_apply_card read k H site A xs]
    exact_mod_cast Nat.lt_iff_add_one_le
  have hm := measureReal_compl (μ := Measure.pi fun _ : Fin m => D)
    (s := {xs : Fin m → FreeMonoid α | harvestSample read k H site xs A ≤ (b : ℝ≥0∞)})
    (Set.to_countable _).measurableSet
  have huniv : (Measure.pi fun _ : Fin m => D).real Set.univ = 1 := by
    simp [measureReal_def, measure_univ]
  rw [hcompl, harvestSample_upper_tail read D k H site A m (b + 1), huniv] at hm
  linarith [hm]

end Random

end OrthoDFA
