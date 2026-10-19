import OrthoDFA.Proofs.RandomNoise
import OrthoDFA.Proofs.RoundEnd

/-!
# Level 3's sampler

Every string a probe harvests at a site is one it can read, undecided, so it starts as the probe
does for `k` letters, and a probe harvests at most one string: the sampler is spread from `k`, and
finds at good read-states at most the probes harvesting one there (`goodAt`). A harvest that does
not end well finds at most twice that, or less than the yield.
-/

namespace OrthoDFA

namespace Random

open MeasureTheory
open scoped ENNReal

variable {α : Type*} [Fintype α] (read : FreeMonoid α → ARU)
  (D : Measure (FreeMonoid α)) (k : ℕ) (H : HypothesisDFA α)

theorem siteHarvest_pot (site : Level3Site) (x : FreeMonoid α) :
    ∀ z ∈ siteHarvest read k H site x, z ∈ pot k x H.tree ∧ read z = .undecided := by
  cases site with
  | ends => exact probeOrAgree_ends read H.tree x H.edges k
  | middles => exact probeOrAgree_pt read H.tree x H.edges k

theorem siteHarvest_length (site : Level3Site) (x : FreeMonoid α) :
    (siteHarvest read k H site x).length ≤ 1 := by
  cases site with
  | ends => exact endsString_length _
  | middles => exact middleString_length _

theorem siteHarvest_nodup (site : Level3Site) (x : FreeMonoid α) :
    (siteHarvest read k H site x).Nodup := by
  have hl := siteHarvest_length read k H site x
  rcases h : siteHarvest read k H site x with _ | ⟨a, _ | ⟨b, l⟩⟩ <;> simp_all

theorem harvestLaw_apply (site : Level3Site) (A : Set (FreeMonoid α))
    [DecidablePred (· ∈ A)] :
    harvestLaw read D k H site A
      = ∫⁻ x, (((siteHarvest read k H site x).filter (· ∈ A)).length : ℝ≥0∞) ∂D :=
  bind_diracSum_apply D _ A

theorem harvestLaw_spread [IsFiniteMeasure D] (site : Level3Site) :
    Spread D k (harvestLaw read D k H site) :=
  spread_of_fresh D k _ (siteHarvest_nodup read k H site) fun x z hz =>
    congrArg FreeMonoid.toList (pot_pre (siteHarvest_pot read k H site x z hz).1).2

/-- The sampler's mass on `A` is the mean count of a probe's harvested strings in `A`. -/
theorem harvestLaw_real (site : Level3Site) (A : Set (FreeMonoid α)) [DecidablePred (· ∈ A)] :
    (harvestLaw read D k H site).real A
      = ∫ x, (((siteHarvest read k H site x).filter (· ∈ A)).length : ℝ) ∂D := by
  rw [measureReal_def, harvestLaw_apply, integral_eq_lintegral_of_nonneg_ae
    (ae_of_all _ fun _ => Nat.cast_nonneg _) (measurable_of_countable _).aestronglyMeasurable]
  simp only [ENNReal.ofReal_natCast]

theorem harvestLaw_real_univ (site : Level3Site) :
    (harvestLaw read D k H site).real Set.univ
      = ∫ x, ((siteHarvest read k H site x).length : ℝ) ∂D := by
  classical
  rw [harvestLaw_real read D k H]
  congr 1
  funext x
  rw [List.filter_eq_self.2 (by simp)]

variable [IsProbabilityMeasure D]

theorem integrable_siteCount (site : Level3Site) (P : FreeMonoid α → Bool) :
    Integrable (fun x => (((siteHarvest read k H site x).filter P).length : ℝ)) D :=
  Integrable.of_bound (measurable_of_countable _).aestronglyMeasurable 1 (ae_of_all _ fun x => by
    rw [Real.norm_eq_abs, abs_of_nonneg (Nat.cast_nonneg _)]
    have := (List.length_filter_le P _).trans (siteHarvest_length read k H site x)
    exact_mod_cast this)

/-- The probes that harvest a string at a good read-state at the site. -/
def goodAt {σ : Type*} (M : DFA α σ) (U : σ → ℝ) (θ : ℝ) (site : Level3Site) :
    Set (FreeMonoid α) :=
  {x | ∃ z ∈ siteHarvest read k H site x, ¬ BadAt U θ (M.eval z.toList)}

/-- A harvest that does not end well finds at most `y`, or twice what its sampler finds at good
read-states, at most `g` where the probes harvesting one are at most `g`. -/
theorem total_le_of_not_endsWell {σ : Type*} (M : DFA α σ) (U : σ → ℝ) (θ : ℝ) {g y : ℝ}
    (site : Level3Site) (hG : D.real (goodAt read k H M U θ site) ≤ g) (ε : ℝ)
    (dis : Set (FreeMonoid α))
    (h : ¬ EndsWell
      { target := M,
        badStates := BadAt U θ,
        probes := D,
        disagreeRate := ε,
        disagree := dis,
        walkStart := k,
        harvestLaw := harvestLaw read D k H,
        yield := y,
        ending := .harvest site }) :
    (harvestLaw read D k H site).real Set.univ ≤ max (2 * g) y := by
  classical
  set ν := harvestLaw read D k H site
  set B := {z : FreeMonoid α | BadAt U θ (M.eval z.toList)}
  by_contra hlt
  push Not at hlt
  refine h ⟨?_, harvestLaw_spread read D k H site, (le_max_right _ _).trans hlt.le⟩
  have hgood : ν.real Bᶜ ≤ g := by
    rw [harvestLaw_real read D k H]
    calc _ ≤ ∫ x, (goodAt read k H M U θ site).indicator (fun _ => (1 : ℝ)) x ∂D := by
          refine integral_mono (integrable_siteCount read D k H site _)
            ((integrable_const _).indicator (Set.to_countable _).measurableSet) fun x => ?_
          by_cases hp : x ∈ goodAt read k H M U θ site
          · rw [Set.indicator_of_mem hp]
            exact Nat.cast_le_one.2
              ((List.length_filter_le _ _).trans (siteHarvest_length read k H site x))
          · rw [Set.indicator_of_notMem hp]
            have : (siteHarvest read k H site x).filter (· ∈ Bᶜ) = [] :=
              List.filter_eq_nil_iff.2 fun z hz hzB => hp ⟨z, hz, by simpa [B] using hzB⟩
            rw [this]
            simp
      _ = D.real (goodAt read k H M U θ site) := by
          rw [integral_indicator_const _ (Set.to_countable _).measurableSet, smul_eq_mul, mul_one]
      _ ≤ g := hG
  have hsplit : ν.real Set.univ = ν.real B + ν.real Bᶜ := by
    have hfin : ν Set.univ ≠ ⊤ := by
      rw [harvestLaw_apply]
      refine ne_top_of_le_ne_top (b := 1) ENNReal.one_ne_top ?_
      calc _ ≤ ∫⁻ _, (1 : ℝ≥0∞) ∂D := lintegral_mono fun x => Nat.cast_le_one.2
            ((List.length_filter_le _ _).trans (siteHarvest_length read k H site x))
        _ = _ := by simp
    rw [← measureReal_union (μ := ν) disjoint_compl_right (Set.to_countable _).measurableSet
      (measure_ne_top_of_subset (Set.subset_univ _) hfin)
      (measure_ne_top_of_subset (Set.subset_univ _) hfin), Set.union_compl_self]
  have h2 := (le_max_left _ _).trans_lt hlt
  change ν.real Set.univ < 2 * ν.real B
  linarith

end Random

end OrthoDFA
