import OrthoDFA.Proofs.RandomNoise
import OrthoDFA.Proofs.RoundEnd

/-!
# Level 3's sampler

Every string a probe harvests at a site is one it can read, undecided, so it starts as the probe
does for `k` letters, and a probe harvests at most one string: the sampler is spread from `k`, and
finds at the states read undecided at most `t` of the time at most the probes harvesting one there
(`goodAt`). A bound at each state's own rate bounds every such set (`sublevel_of_states`). Where
the probes harvesting there are few, a harvest that does not end well finds less than its yield,
or, at the middles, less than an eighth of the searches.
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

/-- A read-state is bad where it reads undecided more than `1.5θ` of the time. -/
def BadAt {σ : Type*} (U : σ → ℝ) (θ : ℝ) (q : σ) : Prop := 3 / 2 * θ < U q

/-- The probes that harvest a string at a good read-state at the site. -/
def goodAt {σ : Type*} (M : DFA α σ) (U : σ → ℝ) (θ : ℝ) (site : Level3Site) :
    Set (FreeMonoid α) :=
  {x | ∃ z ∈ siteHarvest read k H site x, ¬ BadAt U θ (M.eval z.toList)}

/-- The sampler finds at the states read undecided at most `1.5θ` of the time at most the probes
harvesting one there. -/
theorem harvestLaw_good_le {σ : Type*} (M : DFA α σ) (U : σ → ℝ) (θ : ℝ) (site : Level3Site) :
    (harvestLaw read D k H site).real {z | U (M.eval z.toList) ≤ 3 / 2 * θ}
      ≤ D.real (goodAt read k H M U θ site) := by
  classical
  rw [harvestLaw_real read D k H]
  calc _ ≤ ∫ x, (goodAt read k H M U θ site).indicator (fun _ => (1 : ℝ)) x ∂D := by
        refine integral_mono (integrable_siteCount read D k H site _)
          ((integrable_const _).indicator (Set.to_countable _).measurableSet) fun x => ?_
        by_cases hp : x ∈ goodAt read k H M U θ site
        · rw [Set.indicator_of_mem hp]
          exact Nat.cast_le_one.2
            ((List.length_filter_le _ _).trans (siteHarvest_length read k H site x))
        · rw [Set.indicator_of_notMem hp]
          have : (siteHarvest read k H site x).filter
              (· ∈ {z : FreeMonoid α | U (M.eval z.toList) ≤ 3 / 2 * θ}) = [] :=
            List.filter_eq_nil_iff.2 fun z hz hzB => hp ⟨z, hz, by simpa [BadAt] using hzB⟩
          rw [this]
          simp
    _ = D.real (goodAt read k H M U θ site) := by
        rw [integral_indicator_const _ (Set.to_countable _).measurableSet, smul_eq_mul, mul_one]

/-- A bound at each state's own rate bounds every set of the states read undecided at most `T` of
the time. -/
theorem sublevel_of_states {σ : Type*} [Fintype σ] (M : DFA α σ) (U : σ → ℝ)
    (ν : Measure (FreeMonoid α)) {A s : ℝ} (hA : 0 ≤ A) (hs : 0 ≤ s)
    (hU : ∀ z : FreeMonoid α, 0 ≤ U (M.eval z.toList))
    (h : ∀ q, 0 ≤ U q →
      ν.real {z | U (M.eval z.toList) ≤ U q} ≤ (A * U q + s) * ν.real Set.univ) :
    ∀ T, 0 ≤ T → ν.real {z | U (M.eval z.toList) ≤ T} ≤ (A * T + s) * ν.real Set.univ := by
  classical
  intro T hT
  have hν0 : 0 ≤ ν.real Set.univ := measureReal_nonneg
  by_cases hex : ∃ z : FreeMonoid α, U (M.eval z.toList) ≤ T
  · obtain ⟨z₀, hz₀⟩ := hex
    obtain ⟨q₀, hq₀, hmax⟩ := Finset.exists_max_image (Finset.univ.filter fun q => U q ≤ T) U
      ⟨M.eval z₀.toList, by simp [hz₀]⟩
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hq₀ hmax
    have hq0 : 0 ≤ U q₀ := (hU z₀).trans (hmax _ hz₀)
    have hset : {z : FreeMonoid α | U (M.eval z.toList) ≤ T}
        = {z | U (M.eval z.toList) ≤ U q₀} := by
      ext z
      exact ⟨fun hz => hmax _ hz, fun hz => hz.trans hq₀⟩
    rw [hset]
    refine (h q₀ hq0).trans (mul_le_mul_of_nonneg_right ?_ hν0)
    nlinarith
  · push Not at hex
    have : {z : FreeMonoid α | U (M.eval z.toList) ≤ T} = ∅ := by
      ext z
      simp [hex z]
    rw [this, measureReal_empty]
    have := mul_nonneg hA hT
    positivity

/-- Where at most `6(n + 1)t + η` of the probes harvest at the ends at the states read undecided
at most `t ≤ 1/(24(n + 1))` of the time, a harvest there yielding `y ≤ 1/4` puts at most `A·T + s`
of its law on the states read undecided at most `T` of the time, once `A ≥ 6(n + 1)/y` and
`s ≥ η/y`. -/
theorem ends_law {σ : Type*} [Fintype σ] (M : DFA α σ) (U : σ → ℝ) {n : ℕ} {η y A s : ℝ}
    (hy : 0 < y) (hy4 : y ≤ 1 / 4) (hη : 0 ≤ η) (hA : 6 * (n + 1) / y ≤ A) (hs : η / y ≤ s)
    (hU : ∀ z : FreeMonoid α, 0 ≤ U (M.eval z.toList))
    (hc : ∀ q, 0 ≤ U q → 24 * (n + 1) * U q ≤ 1 →
      D.real (goodAt read k H M U (2 / 3 * U q) .ends) ≤ 6 * (n + 1) * U q + η)
    (hyl : y ≤ (harvestLaw read D k H .ends).real Set.univ) :
    ∀ T, 0 ≤ T → (harvestLaw read D k H .ends).real {z | U (M.eval z.toList) ≤ T}
      ≤ (A * T + s) * (harvestLaw read D k H .ends).real Set.univ := by
  set ν := harvestLaw read D k H .ends
  set V := ν.real Set.univ
  have hn : (0 : ℝ) ≤ n := Nat.cast_nonneg n
  have hA0 : 0 ≤ A := (by positivity : (0 : ℝ) ≤ 6 * (n + 1) / y).trans hA
  have hs0 : 0 ≤ s := (div_nonneg hη hy.le).trans hs
  have htop : ν Set.univ ≠ ⊤ := by
    refine ne_top_of_le_ne_top ENNReal.one_ne_top ?_
    rw [harvestLaw_apply]
    calc _ ≤ ∫⁻ _, (1 : ℝ≥0∞) ∂D := lintegral_mono fun x => Nat.cast_le_one.2
          ((List.length_filter_le _ _).trans (siteHarvest_length read k H .ends x))
      _ = 1 := by simp
  refine sublevel_of_states M U ν hA0 hs0 hU fun q hq => ?_
  have hAq : 6 * (n + 1) * U q / y ≤ A * U q := by
    calc 6 * (n + 1) * U q / y = 6 * (n + 1) / y * U q := by ring
      _ ≤ A * U q := mul_le_mul_of_nonneg_right hA hq
  by_cases hsmall : 24 * (n + 1) * U q ≤ 1
  · have h1 := (harvestLaw_good_le read D k H M U (2 / 3 * U q) .ends).trans (hc q hq hsmall)
    rw [show 3 / 2 * (2 / 3 * U q) = U q by ring] at h1
    refine h1.trans ?_
    have e1 : 6 * (n + 1) * U q ≤ A * U q * y := by
      rw [div_le_iff₀ hy] at hAq; linarith
    have e2 : η ≤ s * y := by rw [div_le_iff₀ hy] at hs; linarith
    have e3 : (A * U q + s) * y ≤ (A * U q + s) * V :=
      mul_le_mul_of_nonneg_left hyl (by positivity)
    nlinarith
  · push Not at hsmall
    have h1 : 1 ≤ A * U q := by
      refine le_trans ?_ hAq
      rw [le_div_iff₀ hy]
      nlinarith
    calc ν.real {z | U (M.eval z.toList) ≤ U q} ≤ V := measureReal_mono (Set.subset_univ _) htop
      _ ≤ (A * U q + s) * V := by
          have : 0 ≤ V := measureReal_nonneg
          nlinarith

/-- Where at most `2(L + 1)(n + 1)t` of the searches plus `η` of the probes harvest at the middles
at the states read undecided at most `t ≤ 1/(8(L + 1)(n + 1))` of the time, a harvest there
yielding `y`, and at least an eighth of the searches, puts at most `A·T + s` of its law on the
states read undecided at most `T` of the time, once `A ≥ 16(L + 1)(n + 1)` and `s ≥ η/y`. -/
theorem middles_law {σ : Type*} [Fintype σ] (M : DFA α σ) (U : σ → ℝ) {L n : ℕ} {η y A s : ℝ}
    (hy : 0 < y) (hη : 0 ≤ η) (hA : 16 * ((L + 1) * (n + 1)) ≤ A) (hs : η / y ≤ s)
    (hU : ∀ z : FreeMonoid α, 0 ≤ U (M.eval z.toList))
    (hc : ∀ q, 0 ≤ U q → 8 * ((L + 1) * (n + 1) * U q) ≤ 1 →
      D.real (goodAt read k H M U (2 / 3 * U q) .middles)
        ≤ 2 * ((L + 1) * (n + 1) * U q)
          * D.real {x | searches (probeOrAgree read H.tree H.edges k x) = true} + η)
    (hyl : y ≤ (harvestLaw read D k H .middles).real Set.univ)
    (hZ : D.real {x | searches (probeOrAgree read H.tree H.edges k x) = true}
      ≤ 8 * (harvestLaw read D k H .middles).real Set.univ) :
    ∀ T, 0 ≤ T → (harvestLaw read D k H .middles).real {z | U (M.eval z.toList) ≤ T}
      ≤ (A * T + s) * (harvestLaw read D k H .middles).real Set.univ := by
  set ν := harvestLaw read D k H .middles
  set V := ν.real Set.univ
  set Z := D.real {x | searches (probeOrAgree read H.tree H.edges k x) = true}
  have hn : (0 : ℝ) ≤ n := Nat.cast_nonneg n
  have hL : (0 : ℝ) ≤ L := Nat.cast_nonneg L
  have hA0 : 0 ≤ A := (by positivity : (0 : ℝ) ≤ 16 * ((L + 1) * (n + 1))).trans hA
  have hs0 : 0 ≤ s := (div_nonneg hη hy.le).trans hs
  have hV0 : 0 ≤ V := measureReal_nonneg
  have htop : ν Set.univ ≠ ⊤ := by
    refine ne_top_of_le_ne_top ENNReal.one_ne_top ?_
    rw [harvestLaw_apply]
    calc _ ≤ ∫⁻ _, (1 : ℝ≥0∞) ∂D := lintegral_mono fun x => Nat.cast_le_one.2
          ((List.length_filter_le _ _).trans (siteHarvest_length read k H .middles x))
      _ = 1 := by simp
  refine sublevel_of_states M U ν hA0 hs0 hU fun q hq => ?_
  have hAq : 16 * ((L + 1) * (n + 1)) * U q ≤ A * U q := mul_le_mul_of_nonneg_right hA hq
  by_cases hsmall : 8 * ((L + 1) * (n + 1) * U q) ≤ 1
  · have h1 := (harvestLaw_good_le read D k H M U (2 / 3 * U q) .middles).trans
      (hc q hq hsmall)
    rw [show 3 / 2 * (2 / 3 * U q) = U q by ring] at h1
    refine h1.trans ?_
    have e2 : η ≤ s * y := by rw [div_le_iff₀ hy] at hs; linarith
    have e3 : s * y ≤ s * V := mul_le_mul_of_nonneg_left hyl hs0
    have hLnq : 0 ≤ (L + 1) * (n + 1) * U q := by positivity
    have e1 : 2 * ((L + 1) * (n + 1) * U q) * Z ≤ A * U q * V := by
      calc 2 * ((L + 1) * (n + 1) * U q) * Z
          ≤ 2 * ((L + 1) * (n + 1) * U q) * (8 * V) :=
            mul_le_mul_of_nonneg_left hZ (by positivity)
        _ = 16 * ((L + 1) * (n + 1)) * U q * V := by ring
        _ ≤ A * U q * V := mul_le_mul_of_nonneg_right hAq hV0
    nlinarith
  · push Not at hsmall
    have h1 : 1 ≤ A * U q := by nlinarith
    calc ν.real {z | U (M.eval z.toList) ≤ U q} ≤ V := measureReal_mono (Set.subset_univ _) htop
      _ ≤ (A * U q + s) * V := by nlinarith

/-- Where the probes harvesting at the ends at low-rate states are few (`ends_law`), a harvest
there that does not end well yields less than `y`. -/
theorem ends_total_of_not_endsWell {σ : Type*} [Fintype σ] (M : DFA α σ) (U : σ → ℝ) {n : ℕ}
    {η y A s : ℝ} (hy : 0 < y) (hy4 : y ≤ 1 / 4) (hη : 0 ≤ η) (hA : 6 * (n + 1) / y ≤ A)
    (hs : η / y ≤ s) (hU : ∀ z : FreeMonoid α, 0 ≤ U (M.eval z.toList))
    (hc : ∀ q, 0 ≤ U q → 24 * (n + 1) * U q ≤ 1 →
      D.real (goodAt read k H M U (2 / 3 * U q) .ends) ≤ 6 * (n + 1) * U q + η)
    (ε : ℝ) (dis : Set (FreeMonoid α))
    (h : ¬ EndsWell
      { target := M,
        undecided := U,
        harvestSlope := A,
        harvestSlack := s,
        probes := D,
        disagreeRate := ε,
        disagree := dis,
        walkStart := k,
        harvestLaw := harvestLaw read D k H,
        yield := y,
        ending := .harvest .ends }) :
    (harvestLaw read D k H .ends).real Set.univ ≤ max (2 * 0) y := by
  by_contra hlt
  push Not at hlt
  have hyl : y ≤ (harvestLaw read D k H .ends).real Set.univ := (le_max_right _ _).trans hlt.le
  exact h ⟨ends_law read D k H M U hy hy4 hη hA hs hU hc hyl, harvestLaw_spread read D k H .ends,
    hyl⟩

/-- Where the probes harvesting at the middles at low-rate states are few (`middles_law`), a
harvest there that does not end well yields less than `y`, or less than an eighth of the
searches. -/
theorem middles_total_of_not_endsWell {σ : Type*} [Fintype σ] (M : DFA α σ) (U : σ → ℝ)
    {L n : ℕ} {η y A s : ℝ} (hy : 0 < y) (hη : 0 ≤ η) (hA : 16 * ((L + 1) * (n + 1)) ≤ A)
    (hs : η / y ≤ s) (hU : ∀ z : FreeMonoid α, 0 ≤ U (M.eval z.toList))
    (hc : ∀ q, 0 ≤ U q → 8 * ((L + 1) * (n + 1) * U q) ≤ 1 →
      D.real (goodAt read k H M U (2 / 3 * U q) .middles)
        ≤ 2 * ((L + 1) * (n + 1) * U q)
          * D.real {x | searches (probeOrAgree read H.tree H.edges k x) = true} + η)
    (ε : ℝ) (dis : Set (FreeMonoid α))
    (h : ¬ EndsWell
      { target := M,
        undecided := U,
        harvestSlope := A,
        harvestSlack := s,
        probes := D,
        disagreeRate := ε,
        disagree := dis,
        walkStart := k,
        harvestLaw := harvestLaw read D k H,
        yield := y,
        ending := .harvest .middles }) :
    (harvestLaw read D k H .middles).real Set.univ
      ≤ max (2 * (1 / 16 * D.real {x | searches (probeOrAgree read H.tree H.edges k x) = true}
        + 0)) y := by
  by_contra hlt
  push Not at hlt
  have hyl : y ≤ (harvestLaw read D k H .middles).real Set.univ :=
    (le_max_right _ _).trans hlt.le
  have hZ : D.real {x | searches (probeOrAgree read H.tree H.edges k x) = true}
      ≤ 8 * (harvestLaw read D k H .middles).real Set.univ := by
    have := (le_max_left _ _).trans hlt.le
    linarith
  exact h ⟨middles_law read D k H M U hy hη hA hs hU hc hyl hZ,
    harvestLaw_spread read D k H .middles, hyl⟩

end Random

end OrthoDFA
