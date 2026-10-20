import OrthoDFA.Proofs.LoopRuns
import OrthoDFA.Proofs.LoopLaw

/-!
# From the avoiding probes back to all of them

The round draws its probes from `D` conditioned on avoiding a set `K₀` of first `k` letters, which
removes `m` of `D`. Two reads agreeing wherever `KeyOff` holds harvest the same at every avoiding
probe, so a claim about one read over the conditioned probes is a claim about the other over all
of `D` up to the removed mass: the disagreement grows by `m`, the sampler's yield shrinks by the
factor `1 − m`, and the removed probes, each harvesting at most one string, add at most `m` to
what it finds at any set of states (`looseEndsWell_of_endsWell`).
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory Random

variable {α : Type*} [Fintype α] [DecidableEq α]

/-- `EndsWell` with `m` more allowed of the sampler's law at every set of the states read
undecided at most `T` of the time. -/
def LooseEndsWell {σ H : Type*} (M : _root_.DFA α σ) (U : σ → ℝ) (A sl m : ℝ)
    (D : Measure (FreeMonoid α)) (ε : ℝ) (disagree : Set (FreeMonoid α)) (k : ℕ)
    (law : H → Measure (FreeMonoid α)) (y : ℝ) : RoundEnd H → Prop
  | .consistent => D.real disagree ≤ ε
  | .harvest h => (∀ T, 0 ≤ T → (law h).real {z | U (M.eval z.toList) ≤ T}
        ≤ (A * T + sl) * (law h).real Set.univ + m)
      ∧ Spread D k (law h) ∧ Yields (law h) y
  | .failed => False

/-- The probes the round may draw. -/
def avoiding (k : ℕ) (K₀ : Set (FreeMonoid α)) : Set (FreeMonoid α) := {x | Avoids k K₀ x}

open scoped Classical in
/-- The strings in `S` a probe walked under `H` harvests at the site, summed over `ν`. -/
def siteSum (ν : Measure (FreeMonoid α)) (read : FreeMonoid α → ARU) (k : ℕ) (H : HypothesisDFA α)
    (site : Level3Site) (S : Set (FreeMonoid α)) : ℝ :=
  ∫ x, (((siteHarvest read k H site x).filter (· ∈ S)).length : ℝ) ∂ν

variable {k : ℕ} {K₀ : Set (FreeMonoid α)}

lemma harvestLaw_real_siteSum (ν : Measure (FreeMonoid α)) (read : FreeMonoid α → ARU)
    (H : HypothesisDFA α) (site : Level3Site) (S : Set (FreeMonoid α)) :
    (harvestLaw read ν k H site).real S = siteSum ν read k H site S := by
  classical
  rw [harvestLaw_real]
  rfl

open scoped Classical in
lemma integrable_siteCount' (ν : Measure (FreeMonoid α)) [IsFiniteMeasure ν]
    (read : FreeMonoid α → ARU) (H : HypothesisDFA α) (site : Level3Site) (S : Set (FreeMonoid α)) :
    Integrable (fun x => (((siteHarvest read k H site x).filter (· ∈ S)).length : ℝ)) ν := by
  classical
  refine Integrable.of_bound (measurable_of_countable _).aestronglyMeasurable 1
    (ae_of_all _ fun x => ?_)
  rw [Real.norm_eq_abs, abs_of_nonneg (Nat.cast_nonneg _)]
  have := (List.length_filter_le (· ∈ S) _).trans (siteHarvest_length read k H site x)
  exact_mod_cast this

lemma siteSum_split (D : Measure (FreeMonoid α)) [IsFiniteMeasure D] (read : FreeMonoid α → ARU)
    (H : HypothesisDFA α) (site : Level3Site) (S A : Set (FreeMonoid α)) :
    siteSum D read k H site S
      = siteSum (D.restrict A) read k H site S + siteSum (D.restrict Aᶜ) read k H site S :=
  (integral_add_compl (measurableSet_of_countable A)
    (integrable_siteCount' (k := k) D read H site S)).symm

lemma siteSum_restrict_congr (D : Measure (FreeMonoid α)) {read read' : FreeMonoid α → ARU}
    (hc : ∀ z, KeyOff k K₀ z → read z = read' z) (H : HypothesisDFA α) (site : Level3Site)
    (S : Set (FreeMonoid α)) :
    siteSum (D.restrict (avoiding k K₀)) read k H site S
      = siteSum (D.restrict (avoiding k K₀)) read' k H site S := by
  classical
  refine setIntegral_congr_fun (measurableSet_of_countable _) fun x hx => ?_
  have : probeOrAgree read H.tree H.edges k x = probeOrAgree read' H.tree H.edges k x :=
    probeOrAgree_congr_zone H.edges k fun j hj hjx m _ => hc _ (keyOff_of_prefix hx hj hjx m)
  cases site <;> simp only [siteHarvest, this]

lemma siteSum_smul (ν : Measure (FreeMonoid α)) (c : ENNReal) (read : FreeMonoid α → ARU)
    (H : HypothesisDFA α) (site : Level3Site) (S : Set (FreeMonoid α)) :
    siteSum (c • ν) read k H site S = c.toReal * siteSum ν read k H site S := by
  unfold siteSum
  rw [integral_smul_measure, smul_eq_mul]

lemma siteSum_nonneg (ν : Measure (FreeMonoid α)) (read : FreeMonoid α → ARU)
    (H : HypothesisDFA α) (site : Level3Site) (S : Set (FreeMonoid α)) :
    0 ≤ siteSum ν read k H site S :=
  integral_nonneg fun _ => Nat.cast_nonneg _

lemma siteSum_le (ν : Measure (FreeMonoid α)) [IsFiniteMeasure ν] (read : FreeMonoid α → ARU)
    (H : HypothesisDFA α) (site : Level3Site) (S : Set (FreeMonoid α)) :
    siteSum ν read k H site S ≤ ν.real Set.univ := by
  classical
  have := integral_mono (integrable_siteCount' (k := k) ν read H site S) (integrable_const (1 : ℝ))
    fun x => Nat.cast_le_one.2 ((List.length_filter_le _ _).trans
      (siteHarvest_length read k H site x))
  rw [integral_const, smul_eq_mul, mul_one] at this
  exact this

/-- A claim at the avoiding probes, through a read agreeing off `K₀` and a set of probes `S'`
holding every avoiding probe of `S`, is a weaker claim at all of `D`. -/
theorem looseEndsWell_of_endsWell (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    {σ : Type*} (M : _root_.DFA α σ)
    (U : σ → ℝ) {a sl : ℝ} (ha : 0 ≤ a) (hsl : 0 ≤ sl) {read read' : FreeMonoid α → ARU}
    (hc : ∀ z, KeyOff k K₀ z → read z = read' z) (H : HypothesisDFA α)
    (S S' : Set (FreeMonoid α)) (hS : ∀ x ∈ avoiding k K₀, x ∈ S → x ∈ S')
    (ε y : ℝ) (hy : 0 ≤ y) (e : RoundEnd Level3Site)
    (hm : D.real (avoiding k K₀)ᶜ ≤ 1 / 2)
    (h : EndsWell
      { target := M,
        undecided := U,
        harvestSlope := a,
        harvestSlack := sl,
        probes := popLaw (D.restrict (avoiding k K₀)),
        disagreeRate := ε,
        disagree := S',
        walkStart := k,
        harvestLaw := harvestLaw read' (popLaw (D.restrict (avoiding k K₀))) k H,
        yield := y,
        ending := e }) :
    LooseEndsWell M U a sl (D.real (avoiding k K₀)ᶜ) D (ε + D.real (avoiding k K₀)ᶜ) S k
      (harvestLaw read D k H) ((1 - D.real (avoiding k K₀)ᶜ) * y) e := by
  classical
  set A := avoiding (α := α) k K₀
  set m := D.real Aᶜ with hmd
  have hA : D.real A = 1 - m := by
    rw [hmd, probReal_compl_eq_one_sub (measurableSet_of_countable A)]; ring
  have hApos : 0 < D.real A := by rw [hA]; linarith
  have hA0 : D.restrict A Set.univ ≠ 0 := by
    rw [Measure.restrict_apply_univ]
    intro h0
    rw [measureReal_def, h0, ENNReal.toReal_zero] at hApos
    exact lt_irrefl _ hApos
  have hAtop : D.restrict A Set.univ ≠ ⊤ := measure_ne_top _ _
  -- the conditioned probes
  set D' := popLaw (D.restrict A) with hD'
  have hD'eq : D' = (D.restrict A Set.univ)⁻¹ • D.restrict A := by
    rw [hD']; unfold popLaw; rw [if_neg (by tauto)]
  have hinv : ((D.restrict A Set.univ)⁻¹).toReal = (1 - m)⁻¹ := by
    rw [ENNReal.toReal_inv, Measure.restrict_apply_univ]
    rw [show (D A).toReal = D.real A from rfl, hA]
  -- the sums at `D` against those at `D'`
  have hsum : ∀ (site : Level3Site) (S : Set (FreeMonoid α)),
      siteSum D read k H site S
        = (1 - m) * siteSum D' read' k H site S + siteSum (D.restrict Aᶜ) read k H site S := by
    intro site S
    have hc' : siteSum (D.restrict A) read k H site S = siteSum (D.restrict A) read' k H site S :=
      siteSum_restrict_congr D hc H site S
    rw [siteSum_split D read H site S A, hc', hD'eq, siteSum_smul, hinv]
    field_simp [show (1 - m) ≠ 0 by linarith]
  cases e with
  | consistent =>
    show D.real S ≤ ε + m
    have h' : D'.real S' ≤ ε := h
    rw [hD'eq] at h'
    simp only [measureReal_def, Measure.smul_apply, smul_eq_mul, ENNReal.toReal_mul] at h'
    rw [hinv, Measure.restrict_apply (measurableSet_of_countable _), ← measureReal_def] at h'
    have hsplit := measureReal_inter_add_sdiff₀ (μ := D) (s := S)
      (measurableSet_of_countable A).nullMeasurableSet (measure_ne_top _ _)
    have hinter : D.real (S ∩ A) ≤ D.real (S' ∩ A) :=
      measureReal_mono (fun x ⟨h1, h2⟩ => ⟨hS x h2 h1, h2⟩) (measure_ne_top _ _)
    have hdiff : D.real (S \ A) ≤ m :=
      measureReal_mono (fun x hx => hx.2) (measure_ne_top _ _)
    have h1m : 0 < 1 - m := by linarith
    have hε : 0 ≤ ε := le_trans (mul_nonneg (inv_nonneg.2 h1m.le) measureReal_nonneg) h'
    have : D.real (S' ∩ A) ≤ ε * (1 - m) := by
      rw [inv_mul_le_iff₀ h1m] at h'
      linarith
    have hm0 : 0 ≤ m := measureReal_nonneg
    have hεm : ε * (1 - m) ≤ ε := by nlinarith
    linarith
  | harvest site =>
    obtain ⟨hlaw, -, hyield⟩ := h
    unfold Yields at hyield
    rw [harvestLaw_real_siteSum D' read' H site Set.univ] at hyield
    have h1m : 0 < 1 - m := by linarith
    have hRU := siteSum_le (k := k) (D.restrict Aᶜ) read H site Set.univ
    rw [measureReal_restrict_apply_univ] at hRU
    have hRU0 := siteSum_nonneg (k := k) (D.restrict Aᶜ) read H site Set.univ
    refine ⟨fun T hT => ?_, harvestLaw_spread read D k H site, ?_⟩
    · set B := {z : FreeMonoid α | U (M.eval z.toList) ≤ T}
      have hB := hlaw T hT
      rw [harvestLaw_real_siteSum D' read' H site Set.univ,
        harvestLaw_real_siteSum D' read' H site B] at hB
      rw [harvestLaw_real_siteSum D read H site Set.univ,
        harvestLaw_real_siteSum D read H site B, hsum, hsum]
      have hRB := siteSum_le (k := k) (D.restrict Aᶜ) read H site B
      rw [measureReal_restrict_apply_univ] at hRB
      have hRB' : siteSum (D.restrict Aᶜ) read k H site B ≤ m := hRB
      have hAT : 0 ≤ a * T + sl := by positivity
      have hU'0 := siteSum_nonneg (k := k) D' read' H site Set.univ
      have e1 : (1 - m) * siteSum D' read' k H site B
          ≤ (1 - m) * ((a * T + sl) * siteSum D' read' k H site Set.univ) :=
        mul_le_mul_of_nonneg_left hB h1m.le
      nlinarith [mul_nonneg hAT hRU0]
    · show (1 - m) * y ≤ (harvestLaw read D k H site).real Set.univ
      rw [harvestLaw_real_siteSum D read H site Set.univ, hsum]
      nlinarith only [mul_le_mul_of_nonneg_left hyield h1m.le, hRU0]
  | failed => exact h

end OrthoDFA
