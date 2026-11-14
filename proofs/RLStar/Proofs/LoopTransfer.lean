import RLStar.Proofs.LoopRuns
import RLStar.Proofs.LoopLaw
import RLStar.Proofs.LoopRounds

/-!
# A round's end, and the avoiding probes

`LooseEndsWell` is what the loop needs of a round's end. The round draws its probes from `D`
conditioned on avoiding a set `K₀` of first `k` letters, which removes `m` of `D`. Two reads
harvesting the same at every avoiding probe give a claim about one read over the conditioned
probes as a claim about the other over all of `D` up to the removed mass: the disagreement grows
by `m`, the harvest law's yield shrinks by the factor `1 − m`, and the removed probes, each
harvesting at most one string, add at most `m` to its mass at any set of states
(`looseEndsWell_of_avoiding`). Likewise a claim of at most half of a set at the avoiding probes is
one at all of `D` up to `m` (`real_inter_le_of_avoiding`).
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory Random

variable {α : Type*} [Fintype α] [DecidableEq α]

/-- A round's end is good: a consistent round has at most `ε` of `D` in `disagree`, and a
harvest's law at its site yields `y`, puts no more than `pc` on a string, and, past `m`, at most
`A·T + sl` of itself on the states read undecided at most `T` of the time. -/
def LooseEndsWell {σ : Type*} (M : _root_.DFA α σ) (U : σ → ℝ) (A sl m pc y : ℝ)
    (D : Measure (FreeMonoid α)) (ε : ℝ) (disagree : Set (FreeMonoid α))
    (law : Random.Level3Site → Measure (FreeMonoid α)) : LazyEnd α → Prop
  | .consistent => D.real disagree ≤ ε
  | .harvest s _ => (∀ T, 0 ≤ T → (law s).real {z | U (M.eval z.toList) ≤ T}
        ≤ (A * T + sl) * (law s).real Set.univ + m)
      ∧ (∀ t, (law s).real {t} ≤ pc) ∧ y ≤ (law s).real Set.univ
  | .failed => False

/-- The probes the round may draw. -/
def avoiding (k : ℕ) (K₀ : Set (FreeMonoid α)) : Set (FreeMonoid α) := {x | Avoids k K₀ x}

open scoped Classical in
/-- The strings in `S`, pads dropped, that a probe padded from `X` and walked under `H` harvests at
the site, summed over `ν`. -/
def lazySiteSum (ν : Measure (FreeMonoid α)) (read : FreeMonoid α → ARU) (Kw : ℕ)
    (X : Finset (FreeMonoid α)) (H : HypothesisDFA (Option α)) (site : Level3Site)
    (S : Set (FreeMonoid α)) : ℝ :=
  ∫ x, (((siteHarvest (padRead read) Kw H site (padIn Kw X x)).filter
    fun z => unpad z ∈ S).length : ℝ) ∂ν

variable {k : ℕ} {K₀ : Set (FreeMonoid α)}

lemma lazyHarvestLaw_real_siteSum (ν : Measure (FreeMonoid α)) (read : FreeMonoid α → ARU)
    (Kw : ℕ) (X : Finset (FreeMonoid α)) (H : HypothesisDFA (Option α)) (site : Level3Site)
    (S : Set (FreeMonoid α)) :
    (lazyHarvestLaw read ν Kw X H site).real S = lazySiteSum ν read Kw X H site S := by
  classical
  unfold lazyHarvestLaw lazySiteSum
  rw [map_measureReal_apply (measurable_of_countable _) (Set.to_countable _).measurableSet,
    harvestLaw_real, integral_map (measurable_of_countable _).aemeasurable
      (measurable_of_countable _).aestronglyMeasurable]
  rfl

open scoped Classical in
lemma integrable_lazySiteCount (ν : Measure (FreeMonoid α)) [IsFiniteMeasure ν]
    (read : FreeMonoid α → ARU) (Kw : ℕ) (X : Finset (FreeMonoid α)) (H : HypothesisDFA (Option α))
    (site : Level3Site) (S : Set (FreeMonoid α)) :
    Integrable (fun x => (((siteHarvest (padRead read) Kw H site (padIn Kw X x)).filter
      fun z => unpad z ∈ S).length : ℝ)) ν := by
  refine Integrable.of_bound (measurable_of_countable _).aestronglyMeasurable 1
    (ae_of_all _ fun x => ?_)
  rw [Real.norm_eq_abs, abs_of_nonneg (Nat.cast_nonneg _)]
  have := (List.length_filter_le (fun z => decide (unpad z ∈ S)) _).trans
    (siteHarvest_length (padRead read) Kw H site (padIn Kw X x))
  exact_mod_cast this

lemma lazySiteSum_split (D : Measure (FreeMonoid α)) [IsFiniteMeasure D]
    (read : FreeMonoid α → ARU) (Kw : ℕ) (X : Finset (FreeMonoid α)) (H : HypothesisDFA (Option α))
    (site : Level3Site) (S A : Set (FreeMonoid α)) :
    lazySiteSum D read Kw X H site S
      = lazySiteSum (D.restrict A) read Kw X H site S
        + lazySiteSum (D.restrict Aᶜ) read Kw X H site S :=
  (integral_add_compl (measurableSet_of_countable A)
    (integrable_lazySiteCount D read Kw X H site S)).symm

lemma lazySiteSum_smul (ν : Measure (FreeMonoid α)) (c : ENNReal) (read : FreeMonoid α → ARU)
    (Kw : ℕ) (X : Finset (FreeMonoid α)) (H : HypothesisDFA (Option α)) (site : Level3Site)
    (S : Set (FreeMonoid α)) :
    lazySiteSum (c • ν) read Kw X H site S = c.toReal * lazySiteSum ν read Kw X H site S := by
  unfold lazySiteSum
  rw [integral_smul_measure, smul_eq_mul]

lemma lazySiteSum_nonneg (ν : Measure (FreeMonoid α)) (read : FreeMonoid α → ARU) (Kw : ℕ)
    (X : Finset (FreeMonoid α)) (H : HypothesisDFA (Option α)) (site : Level3Site)
    (S : Set (FreeMonoid α)) : 0 ≤ lazySiteSum ν read Kw X H site S :=
  integral_nonneg fun _ => Nat.cast_nonneg _

lemma lazySiteSum_le (ν : Measure (FreeMonoid α)) [IsFiniteMeasure ν] (read : FreeMonoid α → ARU)
    (Kw : ℕ) (X : Finset (FreeMonoid α)) (H : HypothesisDFA (Option α)) (site : Level3Site)
    (S : Set (FreeMonoid α)) : lazySiteSum ν read Kw X H site S ≤ ν.real Set.univ := by
  classical
  have := integral_mono (integrable_lazySiteCount ν read Kw X H site S) (integrable_const (1 : ℝ))
    fun x => Nat.cast_le_one.2 ((List.length_filter_le _ _).trans
      (siteHarvest_length (padRead read) Kw H site (padIn Kw X x)))
  rw [integral_const, smul_eq_mul, mul_one] at this
  exact this

/-- The avoiding probes' sum, through a read harvesting the same at each of them, scaled back to
`D`, and the removed probes' sum. -/
lemma lazySiteSum_avoiding (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    {read read' : FreeMonoid α → ARU} (Kw : ℕ) (X : Finset (FreeMonoid α))
    (H : HypothesisDFA (Option α))
    (hc : ∀ x ∈ avoiding k K₀, ∀ site, siteHarvest (padRead read) Kw H site (padIn Kw X x)
      = siteHarvest (padRead read') Kw H site (padIn Kw X x))
    (hm : D.real (avoiding k K₀)ᶜ < 1) (site : Level3Site) (S : Set (FreeMonoid α)) :
    lazySiteSum D read Kw X H site S
      = (1 - D.real (avoiding k K₀)ᶜ)
          * lazySiteSum (popLaw (D.restrict (avoiding k K₀))) read' Kw X H site S
        + lazySiteSum (D.restrict (avoiding k K₀)ᶜ) read Kw X H site S := by
  classical
  set A' := avoiding (α := α) k K₀
  set m := D.real A'ᶜ with hmd
  have hA' : D.real A' = 1 - m := by
    rw [hmd, probReal_compl_eq_one_sub (measurableSet_of_countable A')]; ring
  have hApos : 0 < D.real A' := by rw [hA']; linarith
  have hA0 : D.restrict A' Set.univ ≠ 0 := by
    rw [Measure.restrict_apply_univ]
    intro h0
    rw [measureReal_def, h0, ENNReal.toReal_zero] at hApos
    exact lt_irrefl _ hApos
  have hAtop : D.restrict A' Set.univ ≠ ⊤ := measure_ne_top _ _
  have hD'eq : popLaw (D.restrict A') = (D.restrict A' Set.univ)⁻¹ • D.restrict A' := by
    unfold popLaw; rw [if_neg (by tauto)]
  have hinv : ((D.restrict A' Set.univ)⁻¹).toReal = (1 - m)⁻¹ := by
    rw [ENNReal.toReal_inv, Measure.restrict_apply_univ]
    rw [show (D A').toReal = D.real A' from rfl, hA']
  have hc' : lazySiteSum (D.restrict A') read Kw X H site S
      = lazySiteSum (D.restrict A') read' Kw X H site S := by
    refine setIntegral_congr_fun (measurableSet_of_countable _) fun x hx => ?_
    simp only [hc x hx site]
  rw [lazySiteSum_split D read Kw X H site S A', hc', hD'eq, lazySiteSum_smul, hinv]
  field_simp [show (1 - m) ≠ 0 by linarith]

omit [DecidableEq α] in
lemma real_le_of_avoiding (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    {S S' : Set (FreeMonoid α)} (hS : ∀ x ∈ avoiding k K₀, x ∈ S → x ∈ S') {ε : ℝ}
    (hm : D.real (avoiding k K₀)ᶜ < 1)
    (h : (popLaw (D.restrict (avoiding k K₀))).real S' ≤ ε) :
    D.real S ≤ ε + D.real (avoiding k K₀)ᶜ := by
  classical
  set A' := avoiding (α := α) k K₀
  set m := D.real A'ᶜ with hmd
  have hA' : D.real A' = 1 - m := by
    rw [hmd, probReal_compl_eq_one_sub (measurableSet_of_countable A')]; ring
  have hApos : 0 < D.real A' := by rw [hA']; linarith
  have hA0 : D.restrict A' Set.univ ≠ 0 := by
    rw [Measure.restrict_apply_univ]
    intro h0
    rw [measureReal_def, h0, ENNReal.toReal_zero] at hApos
    exact lt_irrefl _ hApos
  have hAtop : D.restrict A' Set.univ ≠ ⊤ := measure_ne_top _ _
  have hD'eq : popLaw (D.restrict A') = (D.restrict A' Set.univ)⁻¹ • D.restrict A' := by
    unfold popLaw; rw [if_neg (by tauto)]
  have hinv : ((D.restrict A' Set.univ)⁻¹).toReal = (1 - m)⁻¹ := by
    rw [ENNReal.toReal_inv, Measure.restrict_apply_univ]
    rw [show (D A').toReal = D.real A' from rfl, hA']
  rw [hD'eq] at h
  simp only [measureReal_def, Measure.smul_apply, smul_eq_mul, ENNReal.toReal_mul] at h
  rw [hinv, Measure.restrict_apply (measurableSet_of_countable _), ← measureReal_def] at h
  have hsplit := measureReal_inter_add_sdiff₀ (μ := D) (s := S)
    (measurableSet_of_countable A').nullMeasurableSet (measure_ne_top _ _)
  have hinter : D.real (S ∩ A') ≤ D.real (S' ∩ A') :=
    measureReal_mono (fun x ⟨h1, h2⟩ => ⟨hS x h2 h1, h2⟩) (measure_ne_top _ _)
  have hdiff : D.real (S \ A') ≤ m :=
    measureReal_mono (fun x hx => hx.2) (measure_ne_top _ _)
  have h1m : 0 < 1 - m := by linarith
  have hε : 0 ≤ ε := le_trans (mul_nonneg (inv_nonneg.2 h1m.le) measureReal_nonneg) h
  have : D.real (S' ∩ A') ≤ ε * (1 - m) := by
    rw [inv_mul_le_iff₀ h1m] at h
    linarith
  have hm0 : 0 ≤ m := measureReal_nonneg
  have hεm : ε * (1 - m) ≤ ε := by nlinarith
  linarith

omit [Fintype α] [DecidableEq α] in
private lemma transfer_slope_arith {A T sl m m₀ a b c d : ℝ} (hAT : 0 ≤ A * T + sl)
    (h1m : 0 ≤ 1 - m) (hm0 : 0 ≤ m) (hm₀ : 0 ≤ m₀) (hB : a ≤ (A * T + sl) * b + m₀)
    (hc : c ≤ m) (hd : 0 ≤ d) :
    (1 - m) * a + c ≤ (A * T + sl) * ((1 - m) * b + d) + (m₀ + m) := by
  have e1 := mul_le_mul_of_nonneg_left hB h1m
  have e2 := mul_nonneg hAT hd
  have e3 := mul_nonneg hm0 hm₀
  nlinarith

omit [Fintype α] [DecidableEq α] in
private lemma transfer_yield_arith {m y b d : ℝ} (h1m : 0 ≤ 1 - m) (hy : y ≤ b) (hd : 0 ≤ d) :
    (1 - m) * y ≤ (1 - m) * b + d := by
  nlinarith [mul_le_mul_of_nonneg_left hy h1m]

/-- A claim at the avoiding probes, through a read harvesting the same at every avoiding probe and
a set of probes `S'` holding every avoiding probe of `S`, is a weaker claim at all of `D`, given the
atom bound `pc'` there. -/
theorem looseEndsWell_of_avoiding (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    {σ : Type*} (M : _root_.DFA α σ) (U : σ → ℝ) {A sl m₀ pc pc' y ε : ℝ} (hA : 0 ≤ A)
    (hsl : 0 ≤ sl) (hm₀ : 0 ≤ m₀) {read read' : FreeMonoid α → ARU} (Kw : ℕ)
    (X : Finset (FreeMonoid α)) (H : HypothesisDFA (Option α))
    (hc : ∀ x ∈ avoiding k K₀, ∀ site, siteHarvest (padRead read) Kw H site (padIn Kw X x)
      = siteHarvest (padRead read') Kw H site (padIn Kw X x))
    (S S' : Set (FreeMonoid α)) (hS : ∀ x ∈ avoiding k K₀, x ∈ S → x ∈ S')
    (hm : D.real (avoiding k K₀)ᶜ ≤ 1 / 2)
    (hpc : ∀ site t, (lazyHarvestLaw read D Kw X H site).real {t} ≤ pc') (e : LazyEnd α)
    (h : LooseEndsWell M U A sl m₀ pc y (popLaw (D.restrict (avoiding k K₀))) ε S'
      (lazyHarvestLaw read' (popLaw (D.restrict (avoiding k K₀))) Kw X H) e) :
    LooseEndsWell M U A sl (m₀ + D.real (avoiding k K₀)ᶜ) pc'
      ((1 - D.real (avoiding k K₀)ᶜ) * y) D (ε + D.real (avoiding k K₀)ᶜ) S
      (lazyHarvestLaw read D Kw X H) e := by
  have hm1 : D.real (avoiding k K₀)ᶜ < 1 := by linarith
  have h1m : 0 ≤ 1 - D.real (avoiding k K₀)ᶜ := by linarith
  have hm0 : 0 ≤ D.real (avoiding k K₀)ᶜ := measureReal_nonneg
  cases e with
  | consistent => exact real_le_of_avoiding D hS hm1 h
  | harvest site ℓ =>
    obtain ⟨hlaw, -, hyield⟩ := h
    have hsum := lazySiteSum_avoiding D Kw X H hc hm1 site
    have hd := lazySiteSum_nonneg (D.restrict (avoiding k K₀)ᶜ) read Kw X H site Set.univ
    refine ⟨fun T hT => ?_, hpc site, ?_⟩
    · have hB := hlaw T hT
      rw [lazyHarvestLaw_real_siteSum, lazyHarvestLaw_real_siteSum] at hB ⊢
      rw [hsum, hsum]
      have hc' := lazySiteSum_le (D.restrict (avoiding k K₀)ᶜ) read Kw X H site
        {z | U (M.eval z.toList) ≤ T}
      rw [measureReal_restrict_apply_univ] at hc'
      exact transfer_slope_arith (by positivity) h1m hm0 hm₀ hB hc' hd
    · rw [lazyHarvestLaw_real_siteSum] at hyield ⊢
      rw [hsum]
      exact transfer_yield_arith h1m hyield hd
  | failed => exact (show False from h).elim

omit [DecidableEq α] in
/-- At most half of `E` and `u` more in `S'` at the avoiding probes, `S'` holding every avoiding
probe of `S`, is at most half of `E` and `u + m` more in `S` at all of `D`. -/
theorem real_inter_le_of_avoiding (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    {E S S' : Set (FreeMonoid α)} (hS : ∀ x ∈ avoiding k K₀, x ∈ S → x ∈ S') {u : ℝ} (hu : 0 ≤ u)
    (hm : D.real (avoiding k K₀)ᶜ ≤ 1 / 2)
    (h : (popLaw (D.restrict (avoiding k K₀))).real (E ∩ S')
      ≤ (popLaw (D.restrict (avoiding k K₀))).real E / 2 + u) :
    D.real (E ∩ S) ≤ D.real E / 2 + (u + D.real (avoiding k K₀)ᶜ) := by
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
  have hD' : ∀ X, (popLaw (D.restrict A)).real X = D.real (X ∩ A) / D.real A := by
    intro X
    unfold popLaw
    rw [if_neg (by tauto)]
    simp only [measureReal_def, Measure.smul_apply, smul_eq_mul, ENNReal.toReal_mul,
      ENNReal.toReal_inv, Measure.restrict_apply_univ,
      Measure.restrict_apply (measurableSet_of_countable X)]
    rw [div_eq_inv_mul]
  rw [hD', hD', div_le_iff₀ hApos, add_mul] at h
  have h1 : D.real (E ∩ S) ≤ D.real (E ∩ S' ∩ A) + m := by
    have hsplit := measureReal_inter_add_sdiff₀ (μ := D) (s := E ∩ S)
      (measurableSet_of_countable A).nullMeasurableSet (measure_ne_top _ _)
    have hinter : D.real (E ∩ S ∩ A) ≤ D.real (E ∩ S' ∩ A) :=
      measureReal_mono (fun x ⟨⟨h1, h2⟩, h3⟩ => ⟨⟨h1, hS x h3 h2⟩, h3⟩) (measure_ne_top _ _)
    have hdiff : D.real ((E ∩ S) \ A) ≤ m :=
      measureReal_mono (fun x hx => hx.2) (measure_ne_top _ _)
    linarith
  have h2 : D.real (E ∩ A) ≤ D.real E :=
    measureReal_mono Set.inter_subset_left (measure_ne_top _ _)
  have h3 : D.real A ≤ 1 := measureReal_le_one
  have h5 : D.real (E ∩ A) / D.real A / 2 * D.real A = D.real (E ∩ A) / 2 := by
    field_simp
  nlinarith [mul_le_mul_of_nonneg_left h3 hu]

end OrthoDFA
