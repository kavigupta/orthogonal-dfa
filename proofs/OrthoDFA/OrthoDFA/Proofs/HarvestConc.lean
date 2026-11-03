import OrthoDFA.Proofs.RandomHarvest
import OrthoDFA.Proofs.BinomLaw
import OrthoDFA.Proofs.Hoeffding
import OrthoDFA.Proofs.BitSum

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

/-- The ideal law's mass on a set is a probability. -/
theorem harvestLaw_real_le_one [IsProbabilityMeasure D] (site : Level3Site)
    (A : Set (FreeMonoid α)) : (harvestLaw read D k H site).real A ≤ 1 := by
  classical
  rw [← siteHit_prob read D k H site A]
  exact measureReal_le_one

open scoped Classical in
/-- Hoeffding's lower tail for the harvest: the sample finds at most `b` of its mass in `A` with
chance `exp(-2mγ²)` once `b ≤ m·(harvestLaw A − γ)`. With `A = Set.univ` this floors the sample's
yield; the derived draw budget `m` is what drives `exp(-2mγ²)` below the failure budget. -/
theorem harvestSample_lower_exp [IsProbabilityMeasure D] (site : Level3Site)
    (A : Set (FreeMonoid α)) (m b : ℕ) {γ : ℝ} (hγ : 0 ≤ γ)
    (hb : (b : ℝ) ≤ m * ((harvestLaw read D k H site).real A - γ)) :
    (Measure.pi fun _ : Fin m => D).real
        {xs | harvestSample read k H site xs A ≤ (b : ℝ≥0∞)}
      ≤ Real.exp (-2 * m * γ ^ 2) := by
  rw [harvestSample_lower_tail read D k H site A m b]
  exact one_sub_binomSfGe_le_exp m b measureReal_nonneg
    (harvestLaw_real_le_one read D k H site A) hγ hb

/-- Hoeffding's `exp(-2mγ²)` is below the failure budget `δ` once the draws reach
`m ≥ log(1/δ)/(2γ²)` -- the budget as a function of the slack `γ` and `δ`. -/
theorem exp_neg_two_mul_le {γ δ : ℝ} (m : ℝ) (hδ : 0 < δ) (hm : Real.log (1 / δ) ≤ 2 * m * γ ^ 2) :
    Real.exp (-2 * m * γ ^ 2) ≤ δ := by
  calc Real.exp (-2 * m * γ ^ 2) ≤ Real.exp (-Real.log (1 / δ)) :=
        Real.exp_le_exp.2 (by linarith)
    _ = δ := by rw [Real.exp_neg, Real.exp_log (by positivity), one_div, inv_inv]

open scoped Classical in
/-- The sample's yield-style undershoot on `A` stays within the failure budget `δ` once the draw
budget `m` reaches `log(1/δ)/(2γ²)`: the draws per harvested probe as a function of the slack. -/
theorem harvestSample_lower_budget [IsProbabilityMeasure D] (site : Level3Site)
    (A : Set (FreeMonoid α)) (m b : ℕ) {γ δ : ℝ} (hγ : 0 ≤ γ) (hδ : 0 < δ)
    (hm : Real.log (1 / δ) ≤ 2 * m * γ ^ 2)
    (hb : (b : ℝ) ≤ m * ((harvestLaw read D k H site).real A - γ)) :
    (Measure.pi fun _ : Fin m => D).real
        {xs | harvestSample read k H site xs A ≤ (b : ℝ≥0∞)} ≤ δ :=
  (harvestSample_lower_exp read D k H site A m b hγ hb).trans (exp_neg_two_mul_le m hδ hm)

end Random

end OrthoDFA
