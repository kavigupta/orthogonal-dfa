import OrthoDFA.Proofs.RandomHarvest
import OrthoDFA.Proofs.BinomLaw
import OrthoDFA.Proofs.Hoeffding
import OrthoDFA.Proofs.BitSum
import OrthoDFA.Proofs.LoopLaw

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

open scoped Classical in
/-- Hoeffding's upper tail for the harvest: the sample finds at least `c` of its mass in `A` with
chance `exp(-2mγ²)` once `c ≥ m·(harvestLaw A + γ)`. The count is `Bin(m, harvestLaw A)`, and
`binomSfGe_symm` turns its upper tail into the complement's lower tail. This caps the numerator of
the sample's set-mass ratio. -/
theorem harvestSample_upper_exp [IsProbabilityMeasure D] (site : Level3Site)
    (A : Set (FreeMonoid α)) (m c : ℕ) {γ : ℝ} (hγ : 0 ≤ γ)
    (hc : (m : ℝ) * ((harvestLaw read D k H site).real A + γ) ≤ c) :
    (Measure.pi fun _ : Fin m => D).real
        {xs | (c : ℝ≥0∞) ≤ harvestSample read k H site xs A}
      ≤ Real.exp (-2 * m * γ ^ 2) := by
  have hp0 : 0 ≤ (harvestLaw read D k H site).real A := measureReal_nonneg
  have hp1 : (harvestLaw read D k H site).real A ≤ 1 := harvestLaw_real_le_one read D k H site A
  rcases le_or_gt c m with hcm | hcm
  · rw [harvestSample_upper_tail read D k H site A m c, binomSfGe_symm _ m c (by omega),
      show m + 1 - c = (m - c) + 1 by omega]
    have hb : ((m - c : ℕ) : ℝ) ≤ m * ((1 - (harvestLaw read D k H site).real A) - γ) := by
      rw [Nat.cast_sub hcm]
      have hrw : (m : ℝ) * ((1 - (harvestLaw read D k H site).real A) - γ)
          = m - m * ((harvestLaw read D k H site).real A + γ) := by ring
      rw [hrw]; linarith [hc]
    exact one_sub_binomSfGe_le_exp m (m - c) (by linarith) (by linarith) hγ hb
  · have hempty : {xs : Fin m → FreeMonoid α | (c : ℝ≥0∞) ≤ harvestSample read k H site xs A}
        = ∅ := by
      ext xs
      simp only [Set.mem_setOf_eq, Set.mem_empty_iff_false, iff_false, not_le]
      rw [harvestSample_apply_card read k H site A xs, Nat.cast_lt]
      have : (Finset.univ.filter fun i => siteHit read k H site A (xs i)).card ≤ m := by
        simpa using Finset.card_filter_le (Finset.univ : Finset (Fin m)) _
      exact lt_of_le_of_lt this hcm
    rw [hempty, measureReal_empty]
    positivity

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

open scoped Classical in
/-- Off the two tail events the sample's set-mass ratio on `S` is at most `(cn-1)/(bn+1)`, so the
chance `popLaw(harvestSample)` puts more than that on `S` is bounded by the numerator overshooting
(`Bin(m, harvestLaw S)` reaching `cn`) plus the denominator undershooting (`Bin(m, harvestLaw univ)`
at most `bn`). The empirical set-mass against the ideal, within slack. -/
theorem harvestSample_ratio_le [IsProbabilityMeasure D] (site : Level3Site)
    (S : Set (FreeMonoid α)) (m cn bn : ℕ) {γn γd : ℝ} (hγn : 0 ≤ γn) (hγd : 0 ≤ γd)
    (hcn : (m : ℝ) * ((harvestLaw read D k H site).real S + γn) ≤ cn)
    (hbn : (bn : ℝ) ≤ m * ((harvestLaw read D k H site).real Set.univ - γd)) :
    (Measure.pi fun _ : Fin m => D).real
        {xs | ((cn : ℝ) - 1) / ((bn : ℝ) + 1)
            < (popLaw (harvestSample read k H site xs)).real S}
      ≤ Real.exp (-2 * m * γn ^ 2) + Real.exp (-2 * m * γd ^ 2) := by
  set Enum := {xs : Fin m → FreeMonoid α | (cn : ℝ≥0∞) ≤ harvestSample read k H site xs S} with hEn
  set Eden := {xs : Fin m → FreeMonoid α | harvestSample read k H site xs Set.univ ≤ (bn : ℝ≥0∞)}
    with hEd
  have hsub : {xs : Fin m → FreeMonoid α | ((cn : ℝ) - 1) / ((bn : ℝ) + 1)
        < (popLaw (harvestSample read k H site xs)).real S} ⊆ Enum ∪ Eden := by
    intro xs hxs
    by_contra hcon
    simp only [hEn, hEd, Set.mem_union, Set.mem_setOf_eq, not_or, not_le] at hcon
    obtain ⟨hnum, hden⟩ := hcon
    have hnum' : (Finset.univ.filter fun i => siteHit read k H site S (xs i)).card < cn := by
      rw [harvestSample_apply_card read k H site S xs] at hnum; exact_mod_cast hnum
    have hden' : bn < (Finset.univ.filter fun i => siteHit read k H site Set.univ (xs i)).card := by
      rw [harvestSample_apply_card read k H site Set.univ xs] at hden; exact_mod_cast hden
    have hSr : (harvestSample read k H site xs).real S
        = ((Finset.univ.filter fun i => siteHit read k H site S (xs i)).card : ℝ) := by
      rw [measureReal_def, harvestSample_apply_card read k H site S xs, ENNReal.toReal_natCast]
    have hUr : (harvestSample read k H site xs).real Set.univ
        = ((Finset.univ.filter fun i => siteHit read k H site Set.univ (xs i)).card : ℝ) := by
      rw [measureReal_def, harvestSample_apply_card read k H site Set.univ xs,
        ENNReal.toReal_natCast]
    have h0 : harvestSample read k H site xs Set.univ ≠ 0 := by
      rw [harvestSample_apply_card read k H site Set.univ xs]
      simp only [ne_eq, Nat.cast_eq_zero]; omega
    have htop : harvestSample read k H site xs Set.univ ≠ ∞ := by
      rw [harvestSample_apply_card read k H site Set.univ xs]; exact ENNReal.natCast_ne_top _
    simp only [Set.mem_setOf_eq] at hxs
    rw [popLaw_real h0 htop, hSr, hUr] at hxs
    have hcS_le : ((Finset.univ.filter fun i => siteHit read k H site S (xs i)).card : ℝ)
        ≤ (cn : ℝ) - 1 := by
      have : (Finset.univ.filter fun i => siteHit read k H site S (xs i)).card + 1 ≤ cn := hnum'
      have := Nat.cast_le (α := ℝ).2 this; push_cast at this; linarith
    have hcU_ge : (bn : ℝ) + 1
        ≤ ((Finset.univ.filter fun i => siteHit read k H site Set.univ (xs i)).card : ℝ) := by
      have : bn + 1 ≤ (Finset.univ.filter fun i => siteHit read k H site Set.univ (xs i)).card :=
        hden'
      have := Nat.cast_le (α := ℝ).2 this; push_cast at this; linarith
    have hbn_pos : (0 : ℝ) < (bn : ℝ) + 1 := by positivity
    have hcU_pos : (0 : ℝ)
        < ((Finset.univ.filter fun i => siteHit read k H site Set.univ (xs i)).card : ℝ) :=
      lt_of_lt_of_le hbn_pos hcU_ge
    have hcn1 : (0 : ℝ) ≤ (cn : ℝ) - 1 := by
      have : (1 : ℝ) ≤ cn := by exact_mod_cast (show 1 ≤ cn by omega)
      linarith
    have hcontra : ((Finset.univ.filter fun i => siteHit read k H site S (xs i)).card : ℝ)
          / ((Finset.univ.filter fun i => siteHit read k H site Set.univ (xs i)).card : ℝ)
        ≤ ((cn : ℝ) - 1) / ((bn : ℝ) + 1) := by
      rw [div_le_div_iff₀ hcU_pos hbn_pos]
      nlinarith [mul_le_mul_of_nonneg_right hcS_le (le_of_lt hbn_pos),
        mul_le_mul_of_nonneg_left hcU_ge hcn1]
    linarith
  calc (Measure.pi fun _ : Fin m => D).real {xs | ((cn : ℝ) - 1) / ((bn : ℝ) + 1)
          < (popLaw (harvestSample read k H site xs)).real S}
      ≤ (Measure.pi fun _ : Fin m => D).real (Enum ∪ Eden) :=
        measureReal_mono hsub
    _ ≤ (Measure.pi fun _ : Fin m => D).real Enum
          + (Measure.pi fun _ : Fin m => D).real Eden := measureReal_union_le _ _
    _ ≤ Real.exp (-2 * m * γn ^ 2) + Real.exp (-2 * m * γd ^ 2) := by
        gcongr
        · exact harvestSample_upper_exp read D k H site S m cn hγn hcn
        · exact harvestSample_lower_exp read D k H site Set.univ m bn hγd hbn

/-- From an ideal set-mass bound `G` and yield `y` for `harvestLaw`, off a set of draws of measure
`2·exp(-2mγ²)` the empirical population puts at most `(G + γ)/(y − γ)` on `S`. -/
theorem harvestSample_le_of_ideal [IsProbabilityMeasure D] (H : HypothesisDFA α)
    (site : Level3Site) (S : Set (FreeMonoid α)) (m : ℕ) (G y γ : ℝ) (hm : 1 ≤ m) (hγ : 0 < γ)
    (hγy : γ < y) (hideal : (harvestLaw read D k H site).real S ≤ G)
    (hyield : y ≤ (harvestLaw read D k H site).real Set.univ) :
    (Measure.pi fun _ : Fin m => D).real
        {xs | (G + γ) / (y - γ) < (popLaw (harvestSample read k H site xs)).real S}
      ≤ 2 * Real.exp (-2 * m * γ ^ 2) := by
  haveI : IsProbabilityMeasure (Measure.pi fun _ : Fin m => D) := inferInstance
  set sS := (harvestLaw read D k H site).real S with hsS
  set sU := (harvestLaw read D k H site).real Set.univ with hsU
  have hsS0 : 0 ≤ sS := measureReal_nonneg
  have hmr : (0 : ℝ) < m := by exact_mod_cast hm
  set cn : ℕ := ⌈(m : ℝ) * (sS + γ)⌉₊ with hcnd
  set bn : ℕ := ⌊(m : ℝ) * (sU - γ)⌋₊ with hbnd
  have hcn : (m : ℝ) * (sS + γ) ≤ cn := Nat.le_ceil _
  have hbn : (bn : ℝ) ≤ (m : ℝ) * (sU - γ) := by
    rw [hbnd]; exact Nat.floor_le (by nlinarith [hyield, hγy, hmr])
  have hratio := harvestSample_ratio_le read D k H site S m cn bn hγ.le hγ.le
    (by rw [← hsS]; exact hcn) (by rw [← hsU]; exact hbn)
  have hfin : (Measure.pi fun _ : Fin m => D)
      {xs | ((cn : ℝ) - 1) / ((bn : ℝ) + 1)
        < (popLaw (harvestSample read k H site xs)).real S} ≠ ⊤ := measure_ne_top _ _
  refine le_trans (measureReal_mono ?_ hfin) (hratio.trans_eq (by ring))
  intro xs hxs
  simp only [Set.mem_setOf_eq] at hxs ⊢
  refine lt_of_le_of_lt ?_ hxs
  -- (cn - 1)/(bn + 1) ≤ (G + γ)/(y - γ)
  have hcn1 : (cn : ℝ) - 1 ≤ (m : ℝ) * (sS + γ) := by
    rw [hcnd]; have := Nat.ceil_lt_add_one (by positivity : (0:ℝ) ≤ (m:ℝ) * (sS + γ)); linarith
  have hbn1 : (m : ℝ) * (sU - γ) ≤ (bn : ℝ) + 1 := by
    rw [hbnd]; have := Nat.lt_floor_add_one ((m : ℝ) * (sU - γ)); linarith
  have hGγ : (m : ℝ) * (sS + γ) ≤ (m : ℝ) * (G + γ) := by nlinarith [hideal]
  have hyγ : (m : ℝ) * (y - γ) ≤ (m : ℝ) * (sU - γ) := by nlinarith [hyield]
  have hbnpos : (0 : ℝ) < (bn : ℝ) + 1 := by positivity
  have hyγ0 : (0 : ℝ) < y - γ := by linarith
  rw [div_le_div_iff₀ hbnpos hyγ0]
  nlinarith [hcn1, hbn1, hGγ, hyγ, mul_pos hmr hyγ0, hsS0, hγ]

/-- A finite union of `harvestSample_le_of_ideal`: over `cs` events, each with its own hypothesis,
set, ideal bound and yield, the draws where any one exceeds its threshold have measure at most
`cs.card · 2·exp(-2mγ²)`. -/
theorem harvestSample_biUnion_le [IsProbabilityMeasure D] {ι : Type*} (cs : Finset ι) (m : ℕ)
    (Hf : ι → HypothesisDFA α) (site : Level3Site) (Sf : ι → Set (FreeMonoid α))
    (G y : ι → ℝ) (γ : ℝ) (hm : 1 ≤ m) (hγ : 0 < γ) (hγy : ∀ i ∈ cs, γ < y i)
    (hideal : ∀ i ∈ cs, (harvestLaw read D k (Hf i) site).real (Sf i) ≤ G i)
    (hyield : ∀ i ∈ cs, y i ≤ (harvestLaw read D k (Hf i) site).real Set.univ) :
    (Measure.pi fun _ : Fin m => D).real
        {xs | ∃ i ∈ cs, (G i + γ) / (y i - γ)
          < (popLaw (harvestSample read k (Hf i) site xs)).real (Sf i)}
      ≤ cs.card * (2 * Real.exp (-2 * m * γ ^ 2)) := by
  classical
  have hset : {xs : Fin m → FreeMonoid α | ∃ i ∈ cs, (G i + γ) / (y i - γ)
        < (popLaw (harvestSample read k (Hf i) site xs)).real (Sf i)}
      = ⋃ i ∈ cs, {xs | (G i + γ) / (y i - γ)
        < (popLaw (harvestSample read k (Hf i) site xs)).real (Sf i)} := by
    ext xs; simp only [Set.mem_setOf_eq, Set.mem_iUnion, exists_prop]
  rw [hset]
  refine le_trans (measureReal_biUnion_finset_le _ _) ?_
  refine le_trans (Finset.sum_le_sum fun i hi =>
    harvestSample_le_of_ideal read D k (Hf i) site (Sf i) m (G i) (y i) (γ := γ) hm hγ
      (hγy i hi) (hideal i hi) (hyield i hi)) ?_
  rw [Finset.sum_const, nsmul_eq_mul]

/-- The `(ω, draws)` pairs where the empirical population (under the read `r ω`) exceeds its own
ideal-normalized threshold for some event in `cs` have measure at most `cs.card · 2·exp(-2mγ²)`,
uniformly in the noise `ω`. -/
theorem harvestSample_prod_biUnion_le {Ω' : Type*} [MeasurableSpace Ω'] (ρ : Measure Ω')
    [IsProbabilityMeasure ρ] [IsProbabilityMeasure D] {ι : Type*} (cs : Finset ι) (m : ℕ)
    (hm : 1 ≤ m) (γ : ℝ) (hγ : 0 < γ) (r : Ω' → FreeMonoid α → ARU) (Hf : ι → HypothesisDFA α)
    (site : Level3Site) (Sf : ι → Set (FreeMonoid α))
    (hmeas : MeasurableSet {p : Ω' × (Fin m → FreeMonoid α) | ∃ i ∈ cs,
      γ < (harvestLaw (r p.1) D k (Hf i) site).real Set.univ ∧
        ((harvestLaw (r p.1) D k (Hf i) site).real (Sf i) + γ)
          / ((harvestLaw (r p.1) D k (Hf i) site).real Set.univ - γ)
          < (popLaw (harvestSample (r p.1) k (Hf i) site p.2)).real (Sf i)}) :
    (ρ.prod (Measure.pi fun _ : Fin m => D)) {p : Ω' × (Fin m → FreeMonoid α) | ∃ i ∈ cs,
        γ < (harvestLaw (r p.1) D k (Hf i) site).real Set.univ ∧
          ((harvestLaw (r p.1) D k (Hf i) site).real (Sf i) + γ)
            / ((harvestLaw (r p.1) D k (Hf i) site).real Set.univ - γ)
            < (popLaw (harvestSample (r p.1) k (Hf i) site p.2)).real (Sf i)}
      ≤ ENNReal.ofReal (cs.card * (2 * Real.exp (-2 * m * γ ^ 2))) := by
  classical
  rw [Measure.prod_apply hmeas]
  have hbd : ∀ ω : Ω', (Measure.pi fun _ : Fin m => D)
      (Prod.mk ω ⁻¹' {p : Ω' × (Fin m → FreeMonoid α) | ∃ i ∈ cs,
        γ < (harvestLaw (r p.1) D k (Hf i) site).real Set.univ ∧
          ((harvestLaw (r p.1) D k (Hf i) site).real (Sf i) + γ)
            / ((harvestLaw (r p.1) D k (Hf i) site).real Set.univ - γ)
            < (popLaw (harvestSample (r p.1) k (Hf i) site p.2)).real (Sf i)})
      ≤ ENNReal.ofReal (cs.card * (2 * Real.exp (-2 * m * γ ^ 2))) := by
    intro ω
    set cs' := cs.filter fun i => γ < (harvestLaw (r ω) D k (Hf i) site).real Set.univ with hcs'
    have hsliceeq : (Prod.mk ω ⁻¹' {p : Ω' × (Fin m → FreeMonoid α) | ∃ i ∈ cs,
        γ < (harvestLaw (r p.1) D k (Hf i) site).real Set.univ ∧
          ((harvestLaw (r p.1) D k (Hf i) site).real (Sf i) + γ)
            / ((harvestLaw (r p.1) D k (Hf i) site).real Set.univ - γ)
            < (popLaw (harvestSample (r p.1) k (Hf i) site p.2)).real (Sf i)})
        = {xs : Fin m → FreeMonoid α | ∃ i ∈ cs',
          ((harvestLaw (r ω) D k (Hf i) site).real (Sf i) + γ)
            / ((harvestLaw (r ω) D k (Hf i) site).real Set.univ - γ)
            < (popLaw (harvestSample (r ω) k (Hf i) site xs)).real (Sf i)} := by
      ext xs
      simp only [Set.mem_preimage, Set.mem_setOf_eq, hcs', Finset.mem_filter]
      constructor
      · rintro ⟨i, hi, hγu, hlt⟩; exact ⟨i, ⟨hi, hγu⟩, hlt⟩
      · rintro ⟨i, ⟨hi, hγu⟩, hlt⟩; exact ⟨i, hi, hγu, hlt⟩
    rw [hsliceeq]
    have hfin : (Measure.pi fun _ : Fin m => D) {xs : Fin m → FreeMonoid α | ∃ i ∈ cs',
        ((harvestLaw (r ω) D k (Hf i) site).real (Sf i) + γ)
          / ((harvestLaw (r ω) D k (Hf i) site).real Set.univ - γ)
          < (popLaw (harvestSample (r ω) k (Hf i) site xs)).real (Sf i)} ≠ ⊤ := measure_ne_top _ _
    rw [← ENNReal.ofReal_toReal hfin, ← measureReal_def]
    refine ENNReal.ofReal_le_ofReal ((harvestSample_biUnion_le (read := r ω) D k cs' m Hf site Sf
        (fun i => (harvestLaw (r ω) D k (Hf i) site).real (Sf i))
        (fun i => (harvestLaw (r ω) D k (Hf i) site).real Set.univ) γ hm hγ
        (fun i hi => (Finset.mem_filter.1 hi).2) (fun _ _ => le_rfl) (fun _ _ => le_rfl)).trans ?_)
    exact mul_le_mul_of_nonneg_right (Nat.cast_le.2 (Finset.card_filter_le _ _)) (by positivity)
  refine le_trans (lintegral_mono hbd) ?_
  rw [lintegral_const, measure_univ, mul_one]

end Random

end OrthoDFA
