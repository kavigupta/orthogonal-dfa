import OrthoDFA.Proofs.RandomNoise
import OrthoDFA.Proofs.RoundEnd

/-!
# Level 3's sampler

Every string a probe harvests at a site is one it can read, undecided, so it starts as the probe
does for `k` letters, and a probe harvests no string twice: the sampler is spread from `k`. A
probe that cannot read a good read-state undecided finds none at a good one, and finds at most
`w` strings (one at the start or the middles, `L + 1` at an edge), so the sampler finds at most
`w G` at good read-states where those probes are at most `G`. A harvest that does not end well
finds at most twice that, or less than the yield.
-/

namespace OrthoDFA

namespace Random

open MeasureTheory
open scoped ENNReal
open OrthoDFA.Ideal (probeOrAgree)

variable {α : Type*} [Fintype α] [DecidableEq α] (read : FreeMonoid α → ARU)
  (D : Measure (FreeMonoid α)) (k : ℕ) (H : Hypothesis α)

theorem siteHarvest_pot (site : Level3Site α) (x : FreeMonoid α) :
    ∀ z ∈ siteHarvest read k H site x, z ∈ pot k x H.tree ∧ read z = .undecided := by
  cases site with
  | start => exact probeOrAgree_start read H.tree x H.edges k
  | edge p c =>
    intro z hz
    obtain ⟨e, he, hez⟩ := undecidedAtEdge_sub _ p c z hz
    exact positionReads_und read H.tree x H.edges k e he z hez
  | middles => exact probeOrAgree_pt read H.tree x H.edges k

theorem siteHarvest_nodup (site : Level3Site α) (x : FreeMonoid α) :
    (siteHarvest read k H site x).Nodup := by
  have h1 : ∀ l : List (FreeMonoid α), l.length ≤ 1 → l.Nodup := by
    intro l hl
    rcases l with _ | ⟨a, _ | ⟨b, l⟩⟩ <;> simp_all
  cases site with
  | start => exact h1 _ (startString_length _)
  | edge p c => exact undecidedAtEdge_nodup _ p c
  | middles => exact h1 _ (middleString_length _)

theorem siteHarvest_length (site : Level3Site α) {L : ℕ} {x : FreeMonoid α}
    (hx : x.toList.length ≤ L) : (siteHarvest read k H site x).length ≤ L + 1 := by
  cases site with
  | start => exact (startString_length _).trans (by omega)
  | edge p c =>
    have := (undecidedAtEdge_length _ p c).trans (positionReads_length read H.tree x H.edges k)
    exact this.trans (by omega)
  | middles => exact (middleString_length _).trans (by omega)

theorem harvestLaw_apply (site : Level3Site α) (A : Set (FreeMonoid α))
    [DecidablePred (· ∈ A)] :
    harvestLaw read D k H site A
      = ∫⁻ x, (((siteHarvest read k H site x).filter (· ∈ A)).length : ℝ≥0∞) ∂D :=
  bind_diracSum_apply D _ A

theorem harvestLaw_spread [IsFiniteMeasure D] (site : Level3Site α) :
    Spread D k (harvestLaw read D k H site) :=
  spread_of_fresh D k _ (siteHarvest_nodup read k H site) fun x z hz =>
    congrArg FreeMonoid.toList (pot_pre (siteHarvest_pot read k H site x z hz).1).2

variable [IsProbabilityMeasure D] {L : ℕ} (hL : ∀ᵐ x ∂D, x.toList.length ≤ L)
include hL

theorem integrable_siteCount (site : Level3Site α) (P : FreeMonoid α → Bool) :
    Integrable (fun x => (((siteHarvest read k H site x).filter P).length : ℝ)) D :=
  Integrable.of_bound (measurable_of_countable _).aestronglyMeasurable (L + 1) (by
    filter_upwards [hL] with x hx
    rw [Real.norm_eq_abs, abs_of_nonneg (Nat.cast_nonneg _)]
    have := (List.length_filter_le P _).trans (siteHarvest_length read k H site hx)
    exact_mod_cast this)

/-- The sampler's mass on `A` is the mean count of a probe's harvested strings in `A`. -/
theorem harvestLaw_real (site : Level3Site α) (A : Set (FreeMonoid α)) [DecidablePred (· ∈ A)] :
    (harvestLaw read D k H site).real A
      = ∫ x, (((siteHarvest read k H site x).filter (· ∈ A)).length : ℝ) ∂D := by
  rw [measureReal_def, harvestLaw_apply, integral_eq_lintegral_of_nonneg_ae
    (ae_of_all _ fun _ => Nat.cast_nonneg _) (measurable_of_countable _).aestronglyMeasurable]
  simp only [ENNReal.ofReal_natCast]

theorem harvestLaw_real_univ (site : Level3Site α) :
    (harvestLaw read D k H site).real Set.univ
      = ∫ x, ((siteHarvest read k H site x).length : ℝ) ∂D := by
  classical
  rw [harvestLaw_real read D k H hL]
  congr 1
  funext x
  rw [List.filter_eq_self.2 (by simp)]

/-- A harvest that does not end well finds at most `y`, or twice what its sampler can find at good
read-states: at most `w G`, where a probe finds at most `w` strings and the probes that can read a
good read-state undecided are at most `G`. -/
theorem total_le_of_not_endsWell {σ : Type*} (M : DFA α σ) (U : σ → ℝ) (θ : ℝ) {G y : ℝ}
    (hG : D.real (PotGood M U θ k read H.tree) ≤ G) (ε : ℝ) (dis : Set (FreeMonoid α))
    (site : Level3Site α) (w : ℕ) (hw : ∀ᵐ x ∂D, (siteHarvest read k H site x).length ≤ w)
    (h : ¬ EndsWell M (BadAt U θ) D ε dis k (harvestLaw read D k H) y (.harvest site)) :
    (harvestLaw read D k H site).real Set.univ ≤ max (2 * w * G) y := by
  classical
  set ν := harvestLaw read D k H site
  set B := {z : FreeMonoid α | BadAt U θ (M.eval z.toList)}
  by_contra hlt
  push Not at hlt
  refine h ⟨?_, harvestLaw_spread read D k H site, (le_max_right _ _).trans hlt.le⟩
  have hgood : ν.real Bᶜ ≤ w * G := by
    rw [harvestLaw_real read D k H hL]
    calc _ ≤ ∫ x, (PotGood M U θ k read H.tree).indicator (fun _ => (w : ℝ)) x ∂D := by
          refine integral_mono_ae (integrable_siteCount read D k H hL site _)
            ((integrable_const _).indicator (Set.to_countable _).measurableSet) ?_
          filter_upwards [hw] with x hx
          by_cases hp : x ∈ PotGood M U θ k read H.tree
          · rw [Set.indicator_of_mem hp]
            have := (List.length_filter_le (fun z => decide (z ∈ Bᶜ)) _).trans hx
            exact_mod_cast this
          · rw [Set.indicator_of_notMem hp]
            have : (siteHarvest read k H site x).filter (· ∈ Bᶜ) = [] := by
              refine List.filter_eq_nil_iff.2 fun z hz hzB => hp ?_
              have hz' := siteHarvest_pot read k H site x z hz
              exact ⟨z, hz'.1, by simpa [B] using hzB, hz'.2⟩
            rw [this]
            simp
      _ = w * D.real (PotGood M U θ k read H.tree) := by
          rw [integral_indicator_const _ (Set.to_countable _).measurableSet, smul_eq_mul,
            mul_comm]
      _ ≤ w * G := by gcongr
  have hsplit : ν.real Set.univ = ν.real B + ν.real Bᶜ := by
    have hfin : ν Set.univ ≠ ⊤ := by
      rw [harvestLaw_apply]
      refine ne_top_of_le_ne_top (b := ((L + 1 : ℕ) : ℝ≥0∞)) (by simp) ?_
      calc _ ≤ ∫⁻ _, ((L + 1 : ℕ) : ℝ≥0∞) ∂D := lintegral_mono_ae (by
            filter_upwards [hL] with x hx
            exact Nat.cast_le.2 ((List.length_filter_le _ _).trans
              (siteHarvest_length read k H site hx)))
        _ = _ := by simp
    rw [← measureReal_union (μ := ν) disjoint_compl_right (Set.to_countable _).measurableSet
      (measure_ne_top_of_subset (Set.subset_univ _) hfin)
      (measure_ne_top_of_subset (Set.subset_univ _) hfin), Set.union_compl_self]
  have h2 := (le_max_left _ _).trans_lt hlt
  change ν.real Set.univ < 2 * ν.real B
  linarith

end Random

end OrthoDFA
