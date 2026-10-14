import OrthoDFA.Proofs.IdealHarvest
import OrthoDFA.Proofs.SeqTest
import Mathlib.Probability.ProductMeasure
import Mathlib.Analysis.Complex.ExponentialBounds

/-!
# The idealized round: the proofs

Every step that does not end the round lowers `Θ`, and the start's `Θ` is below the budget, so the
round ends on its probes, and never fails.

The chance part (`anyStep_le`): a potential that pays for a step's event and, on average over the
fresh probe, for its own value after the step bounds the chance that some step of the run is such
an event. At most `Ψ` hypotheses follow, each starting its potential afresh. Two potentials, each
`a` times a sequential test's likelihood ratio, which a draw on the test's side of its rate
multiplies by at most 1 on average (`OrthoDFA.SeqRateSound`), so each test errs with chance at most
`a` a stretch, whatever its length:
* a consistent ending at a hypothesis that does not end clean on more than `disagreementRate` of
  the probes: the disagreements' test calls them below;
* a harvest at a hypothesis stopping at an undecided read on fewer than `harvestRate` of the
  probes: the undecided reads' test calls them above.
-/

namespace OrthoDFA

namespace Ideal

open MeasureTheory Finset
open scoped ENNReal

/-! ## Fresh draws -/

section Draws

variable {X : Type*} [MeasurableSpace X] [Countable X] [MeasurableSingletonClass X]

omit [Countable X] [MeasurableSingletonClass X] in
theorem piFinSuccAbove_zero {n : ℕ} (d : Fin (n + 1) → X) :
    MeasurableEquiv.piFinSuccAbove (fun _ : Fin (n + 1) => X) 0 d = (d 0, Fin.tail d) := by
  ext j
  · rfl
  · simp [MeasurableEquiv.piFinSuccAbove, Fin.tail]

/-- The chance of `E` over `T + 1` draws, as the first draw's average of its section. -/
theorem pi_succ_apply (D : Measure X) [IsProbabilityMeasure D] (T : ℕ)
    (E : Set (Fin (T + 1) → X)) :
    (Measure.pi fun _ : Fin (T + 1) => D) E
      = ∫⁻ x, (Measure.pi fun _ : Fin T => D) {xs | Fin.cons x xs ∈ E} ∂D := by
  have hmp := measurePreserving_piFinSuccAbove (fun _ : Fin (T + 1) => D) 0
  set e := MeasurableEquiv.piFinSuccAbove (fun _ : Fin (T + 1) => X) 0
  have hE : E = e ⁻¹' {p | Fin.cons p.1 p.2 ∈ E} := by
    ext d
    simp only [Set.mem_preimage, Set.mem_ofPred_eq, e, piFinSuccAbove_zero, Fin.cons_self_tail]
  rw [hE, hmp.measure_preimage (Set.to_countable _).measurableSet.nullMeasurableSet,
    Measure.prod_apply (Set.to_countable _).measurableSet]
  rfl

end Draws

variable {α : Type*} [DecidableEq α] (read : FreeMonoid α → ARU) (C : Level2Settings)

/-! ## The round ends -/

section Ends

variable [Fintype α] {σ : Type*} [Fintype σ] {M : DFA α σ} {side : σ → Bool}

omit [Fintype α] in
theorem inv_start (hn : 1 ≤ C.ends) : Inv read C (initialState : Level2State α) := by
  refine inv_fresh read C hn (fun _ _ _ _ _ h => by simp [initialState, newHypothesis] at h) ?_
  intro p hp
  simp only [DTree.leaves, List.map_cons, List.map_nil, List.cons_append, List.nil_append,
    List.mem_cons, List.not_mem_nil, or_false] at hp
  rcases hp with rfl | rfl
  · exact Or.inl rfl
  · exact Or.inr (Or.inl rfl)

/-- A step that goes on keeps `Inv` and lowers `Θ`. -/
theorem step_inv (hI : IdealReads read M side) (hC : TestsOK C)
    {s s' : Level2State α} (hs : Inv read C s) {x : FreeMonoid α}
    (h : applyProbe read C s x = .inl s') :
    Inv read C s' ∧ Θ C (Fintype.card σ) s' < Θ C (Fintype.card σ) s := by
  by_cases hc : Changes (probeOrAgree false read s.tree s.edges C.walkStart x)
  · obtain ⟨T, E, he, hinv, hΨ⟩ := step_change read C hI (one_le_ends hC) hs hc
    rw [he, Sum.inl.injEq] at h
    subst h
    exact ⟨hinv, Θ_lt_change C _ hs.probes_lt T E hΨ⟩
  · rcases step_same read C hC hs hc (stuckString_length read ⟨s.tree, s.edges⟩ C.walkStart x) with
      ⟨he, hinv, -⟩ | ⟨he, -⟩ | ⟨he, -, -⟩ <;> rw [he] at h
    · rw [Sum.inl.injEq] at h
      subst h
      exact ⟨hinv, Θ_lt_same C _ rfl rfl rfl hinv.probes_lt⟩
    · cases h
    · cases h

theorem applyProbe_ne_failed (s : Level2State α) (x : FreeMonoid α) :
    applyProbe read C s x ≠ .inr .failed := by
  classical
  have hkeep : ∀ n d, keep C s n d ≠ .inr .failed := by
    intro n d; unfold keep; dsimp only; split_ifs <;> simp
  unfold applyProbe
  split
  · split
    · simp
    · exact hkeep _ _
  · simp
  · exact hkeep _ _
  · exact hkeep _ _

end Ends

/-! ## The hit rate -/

section Rates

variable [Fintype α] (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]

/-- The chance a probe stops at an undecided read under the hypothesis of `s`. -/
noncomputable def hitRate (s : Level2State α) : ℝ :=
  D.real {x | stuckString read ⟨s.tree, s.edges⟩ C.walkStart x ≠ []}

/-- A probe that disagrees does not end clean: it changes the hypothesis, or its search stops at an
undecided midpoint. -/
theorem disagrees_le {s : Level2State α} :
    D.real {x | Disagrees read ⟨s.tree, s.edges⟩ C.walkStart x}
      ≤ D.real {x | (probeOrAgree false read s.tree s.edges C.walkStart x).clean = false} :=
  measureReal_mono fun x hx => probeOrAgree_disagrees false read s.tree s.edges C.walkStart hx

end Rates

/-! ## The chance of a bad step -/

section Run

variable [Fintype α] (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]

/-- Some step of the run from `s` is one of the steps `Ev`. -/
def AnyStep (Ev : Level2State α → FreeMonoid α → Prop) :
    Level2State α → List (FreeMonoid α) → Prop
  | _, [] => False
  | s, x :: xs => Ev s x ∨ ∃ s', applyProbe read C s x = .inl s' ∧ AnyStep Ev s' xs

/-- A potential that pays for a step's event and, on average over the probe, for its own value
after the step, bounds the chance that some step of the run is such an event. -/
theorem anyStep_le (Ev : Level2State α → FreeMonoid α → Prop) [∀ s x, Decidable (Ev s x)]
    (I : Level2State α → Prop)
    (F : Level2State α → ℝ≥0∞) (hI : ∀ s x s', I s → applyProbe read C s x = .inl s' → I s')
    (hF : ∀ s, I s → ∫⁻ x, ((if Ev s x then 1 else 0)
      + Sum.elim F (fun _ => 0) (applyProbe read C s x)) ∂D ≤ F s) :
    ∀ (N : ℕ) (s : Level2State α), I s →
      (Measure.pi fun _ : Fin N => D) {xs | AnyStep read C Ev s (List.ofFn xs)} ≤ F s := by
  intro N
  induction N with
  | zero => intro s _; simp [AnyStep]
  | succ N ih =>
    intro s hs
    rw [pi_succ_apply]
    refine le_trans (lintegral_mono fun x => ?_) (hF s hs)
    simp only [Set.mem_ofPred_eq, List.ofFn_succ, Fin.cons_zero, Fin.cons_succ, AnyStep]
    by_cases he : Ev s x
    · simp only [he, true_or, Set.ofPred_true, measure_univ, if_true]
      exact le_self_add
    · simp only [he, false_or, if_false, zero_add]
      rcases hst : applyProbe read C s x with s' | e
      · simp only [Sum.inl.injEq, exists_eq_left', Sum.elim_inl]
        exact ih s' (hI s x s' hs hst)
      · simp

/-- `a` times the ratio after one more draw averages at most `a` times the ratio now, on the far
side of `p₀` from `p₁`. -/
theorem lintegral_ratio_succ (P : FreeMonoid α → Prop) [DecidablePred P] {p₀ p₁ a : ℝ}
    (hp₀ : 0 < p₀) (hp₀1 : p₀ < 1) (hp₁ : 0 ≤ p₁) (hp₁1 : p₁ ≤ 1) (ha : 0 ≤ a)
    (hq : (D.real {x | P x} - p₀) * (p₁ - p₀) ≤ 0) {n h : ℕ} (hh : h ≤ n) :
    ∫⁻ x, ENNReal.ofReal (a * likelihoodRatio p₀ p₁ (n + 1)
      (h + if decide (P x) then 1 else 0)) ∂D ≤ ENNReal.ofReal (a * likelihoodRatio p₀ p₁ n h) := by
  have hl := likelihoodRatio_nonneg hp₀ hp₀1 hp₁ hp₁1 n h
  have hr : ∀ x, 0 ≤ ratioFactor p₀ p₁ (decide (P x)) := by
    intro x
    unfold ratioFactor
    split_ifs
    · positivity
    · exact div_nonneg (by linarith) (by linarith)
  calc _ = ∫⁻ x, ENNReal.ofReal (a * likelihoodRatio p₀ p₁ n h)
        * ENNReal.ofReal (ratioFactor p₀ p₁ (decide (P x))) ∂D := by
        refine lintegral_congr fun x => ?_
        rw [likelihoodRatio_succ hh, ← ENNReal.ofReal_mul (by positivity)]
        ring_nf
    _ = ENNReal.ofReal (a * likelihoodRatio p₀ p₁ n h)
        * ∫⁻ x, ENNReal.ofReal (ratioFactor p₀ p₁ (decide (P x))) ∂D :=
        lintegral_const_mul _ (measurable_of_countable _)
    _ ≤ ENNReal.ofReal (a * likelihoodRatio p₀ p₁ n h) * 1 := by
        gcongr; exact lintegral_ratioFactor_le D P hp₀ hp₀1 hp₁ hp₁1 hq
    _ = _ := mul_one _

variable {σ : Type*} [Fintype σ] {M : DFA α σ} {side : σ → Bool}

/-- The hypothesis of `s` does not end clean on more than `disagreementRate` of the probes. -/
def Bad (s : Level2State α) : Prop :=
  C.disagreementRate
    < D.real {x | (probeOrAgree false read s.tree s.edges C.walkStart x).clean = false}

open scoped Classical in
/-- What remains of a stretch's chance to fire the undecided reads' test at a rate below
`harvestRate`: `a` times its likelihood ratio. -/
noncomputable def harvestMass (s : Level2State α) : ℝ≥0∞ :=
  if hitRate read C D s < C.harvestRate then
    ENNReal.ofReal (C.testLevel
      * likelihoodRatio C.harvestRate C.alternativeRate s.probes s.undecided)
  else 0

open scoped Classical in
/-- What remains of a stretch's chance to fire the disagreements' test at a `Bad` hypothesis. -/
noncomputable def consistentMass (s : Level2State α) : ℝ≥0∞ :=
  if Bad read C D s then
    ENNReal.ofReal (C.testLevel
      * likelihoodRatio C.disagreementRate C.alternativeRate s.probes s.disagreements)
  else 0

theorem stuck_count (s : Level2State α) (x : FreeMonoid α) :
    (counted s (probeOrAgree false read s.tree s.edges C.walkStart x)).undecided
      = s.undecided + if decide (stuckString read ⟨s.tree, s.edges⟩ C.walkStart x ≠ []) then 1
        else 0 := by
  have := stuckString_length read ⟨s.tree, s.edges⟩ C.walkStart x
  show s.undecided + (stuckString read ⟨s.tree, s.edges⟩ C.walkStart x).length = _
  by_cases h : stuckString read ⟨s.tree, s.edges⟩ C.walkStart x = []
  · simp [h]
  · have h1 : (stuckString read ⟨s.tree, s.edges⟩ C.walkStart x).length = 1 := by
      have := List.length_pos_iff.2 h; omega
    simp [h, h1]

theorem dis_count (s : Level2State α) (x : FreeMonoid α) :
    (counted s (probeOrAgree false read s.tree s.edges C.walkStart x)).disagreements
      = s.disagreements + if decide ((probeOrAgree false read s.tree s.edges C.walkStart x).clean
        = false) then 1 else 0 := by
  show s.disagreements + (if (probeOrAgree false read s.tree s.edges C.walkStart x).clean then 0
    else 1) = _
  cases (probeOrAgree false read s.tree s.edges C.walkStart x).clean <;> simp

open scoped Classical in
/-- A harvest at a hypothesis stopping at an undecided read on fewer than `harvestRate` of the
probes: the undecided reads' sequential test fires at some look of the stretch, with chance at most
`a` a stretch. -/
theorem falseHarvest_le (hI : IdealReads read M side) (hC : TestsOK C) (N : ℕ) :
    (Measure.pi fun _ : Fin N => D) {xs | AnyStep read C
        (fun s x => applyProbe read C s x = .inr (.harvest ())
          ∧ hitRate read C D s < C.harvestRate) initialState (List.ofFn xs)}
      ≤ ENNReal.ofReal (stretches (Fintype.card σ) (Fintype.card α) * C.testLevel) := by
  set nσ := Fintype.card σ
  set y := C.harvestRate
  set m := C.alternativeRate
  set a := C.testLevel
  have ha := hC.testLevel_pos
  have hy := hC.harvestRate_pos
  have hym := hC.harvestRate_lt
  have hm1 : m < 1 := hC.alternativeRate_lt.trans hC.disagreementRate_lt
  have hfresh : ∀ T E, harvestMass read C D (newHypothesis T E) ≤ ENNReal.ofReal a := by
    intro T E
    unfold harvestMass
    split_ifs
    · simp only [newHypothesis, likelihoodRatio_zero, mul_one]
      exact le_rfl
    · exact bot_le
  let F : Level2State α → ℝ≥0∞ := fun s => ENNReal.ofReal (a * Ψ nσ s) + harvestMass read C D s
  have hF : ∀ s, Inv read C s → ∫⁻ x, ((if (applyProbe read C s x = .inr (.harvest ())
      ∧ hitRate read C D s < y) then 1 else 0) + Sum.elim F (fun _ => 0) (applyProbe read C s x))
      ∂D ≤ F s := by
    intro s hs
    let g : FreeMonoid α → ℝ≥0∞ := fun x => if hitRate read C D s < y then
      ENNReal.ofReal (a * likelihoodRatio y m (s.probes + 1) (s.undecided
        + if decide (stuckString read ⟨s.tree, s.edges⟩ C.walkStart x ≠ []) then 1 else 0))
      else 0
    have hpt : ∀ x, ((if (applyProbe read C s x = .inr (.harvest ()) ∧ hitRate read C D s < y)
        then 1 else 0) + Sum.elim F (fun _ => 0) (applyProbe read C s x))
        ≤ ENNReal.ofReal (a * Ψ nσ s) + g x := by
      intro x
      by_cases hc : Changes (probeOrAgree false read s.tree s.edges C.walkStart x)
      · obtain ⟨T, E, he, -, hΨ⟩ := step_change read C hI (one_le_ends hC) hs hc
        rw [he]
        simp only [reduceCtorEq, false_and, if_false, Sum.elim_inl, zero_add, F]
        refine le_trans ?_ le_self_add
        have h1 : ((Ψ nσ (newHypothesis T E) : ℕ) : ℝ) + 1 ≤ Ψ nσ s := by exact_mod_cast hΨ
        calc ENNReal.ofReal (a * Ψ nσ (newHypothesis T E))
              + harvestMass read C D (newHypothesis T E)
            ≤ ENNReal.ofReal (a * Ψ nσ (newHypothesis T E)) + ENNReal.ofReal a := by
              gcongr; exact hfresh T E
          _ ≤ ENNReal.ofReal (a * Ψ nσ s) := by
              rw [← ENNReal.ofReal_add (by positivity) ha.le]
              refine ENNReal.ofReal_le_ofReal ?_
              nlinarith
      · have hlen := stuckString_length read ⟨s.tree, s.edges⟩ C.walkStart x
        have hcnt := stuck_count read C s x
        rcases step_same read C hC hs hc hlen with ⟨he, -, -⟩ | ⟨he, hfire⟩ | ⟨he, -, -⟩ <;>
          rw [he]
        · simp only [reduceCtorEq, false_and, if_false, Sum.elim_inl, zero_add, F]
          refine add_le_add (le_of_eq rfl) (le_of_eq ?_)
          simp only [g, harvestMass]
          rw [hcnt]
          rfl
        · simp only [Sum.elim_inr, add_zero]
          split_ifs with hp
          · refine le_trans ?_ le_add_self
            simp only [g, if_pos hp.2]
            rw [← ENNReal.ofReal_one]
            refine ENNReal.ofReal_le_ofReal ?_
            have : 1 / a ≤ likelihoodRatio y m (s.probes + 1) (s.undecided
                + if decide (stuckString read ⟨s.tree, s.edges⟩ C.walkStart x ≠ []) then 1
                  else 0) := by
              rw [← hcnt]; exact hfire
            rw [div_le_iff₀ ha] at this
            linarith
          · exact bot_le
        · simp
    calc _ ≤ ∫⁻ x, ENNReal.ofReal (a * Ψ nσ s) + g x ∂D := lintegral_mono hpt
      _ = ENNReal.ofReal (a * Ψ nσ s) + ∫⁻ x, g x ∂D := by
          rw [lintegral_add_left measurable_const, lintegral_const, measure_univ, mul_one]
      _ ≤ F s := by
          simp only [F]
          gcongr
          simp only [g, harvestMass]
          split_ifs with hp
          · have hq : (D.real {x | stuckString read ⟨s.tree, s.edges⟩ C.walkStart x ≠ []} - y)
                * (m - y) ≤ 0 :=
              mul_nonpos_of_nonpos_of_nonneg (by simp only [hitRate] at hp; linarith)
                (by linarith)
            exact lintegral_ratio_succ D
              (fun x => stuckString read ⟨s.tree, s.edges⟩ C.walkStart x ≠ []) hy (by linarith)
              (by linarith) hm1.le ha.le hq hs.undecided_le_probes
          · simp
  refine (anyStep_le read C D _ (Inv read C) F (fun s x s' hs h => (step_inv read C hI hC hs h).1)
    hF N initialState (inv_start read C (one_le_ends hC))).trans ?_
  have hΨ : (Ψ nσ (initialState : Level2State α) : ℝ) + 1 ≤ stretches nσ (Fintype.card α) := by
    exact_mod_cast Ψ_start nσ (α := α)
  calc F initialState ≤ ENNReal.ofReal (a * Ψ nσ (initialState : Level2State α))
        + ENNReal.ofReal a := add_le_add le_rfl (hfresh _ _)
    _ ≤ _ := by
        rw [← ENNReal.ofReal_add (by positivity) ha.le]
        refine ENNReal.ofReal_le_ofReal ?_
        nlinarith

open scoped Classical in
/-- A consistent ending at a hypothesis that does not end clean on more than `disagreementRate` of
the probes: the disagreements' sequential test fires at some look of the stretch, with chance at
most `a` a stretch. -/
theorem falseConsistent_le (hI : IdealReads read M side) (hC : TestsOK C) (N : ℕ) :
    (Measure.pi fun _ : Fin N => D) {xs | AnyStep read C
        (fun s x => applyProbe read C s x = .inr .consistent ∧ Bad read C D s) initialState
          (List.ofFn xs)}
      ≤ ENNReal.ofReal (stretches (Fintype.card σ) (Fintype.card α) * C.testLevel) := by
  set nσ := Fintype.card σ
  set ε := C.disagreementRate
  set m := C.alternativeRate
  set a := C.testLevel
  have ha := hC.testLevel_pos
  have hmε := hC.alternativeRate_lt
  have hε1 := hC.disagreementRate_lt
  have hm0 : 0 < m := hC.harvestRate_pos.trans hC.harvestRate_lt
  have hfresh : ∀ T E, consistentMass read C D (newHypothesis T E) ≤ ENNReal.ofReal a := by
    intro T E
    unfold consistentMass
    split_ifs
    · simp only [newHypothesis, likelihoodRatio_zero, mul_one]
      exact le_rfl
    · exact bot_le
  let F : Level2State α → ℝ≥0∞ := fun s =>
    ENNReal.ofReal (a * Ψ nσ s) + consistentMass read C D s
  have hF : ∀ s, Inv read C s → ∫⁻ x, ((if (applyProbe read C s x = .inr .consistent
      ∧ Bad read C D s) then 1 else 0) + Sum.elim F (fun _ => 0) (applyProbe read C s x))
      ∂D ≤ F s := by
    intro s hs
    let g : FreeMonoid α → ℝ≥0∞ := fun x => if Bad read C D s then
      ENNReal.ofReal (a * likelihoodRatio ε m (s.probes + 1) (s.disagreements
        + if decide ((probeOrAgree false read s.tree s.edges C.walkStart x).clean = false) then 1
          else 0))
      else 0
    have hpt : ∀ x, ((if (applyProbe read C s x = .inr .consistent ∧ Bad read C D s)
        then 1 else 0) + Sum.elim F (fun _ => 0) (applyProbe read C s x))
        ≤ ENNReal.ofReal (a * Ψ nσ s) + g x := by
      intro x
      by_cases hc : Changes (probeOrAgree false read s.tree s.edges C.walkStart x)
      · obtain ⟨T, E, he, -, hΨ⟩ := step_change read C hI (one_le_ends hC) hs hc
        rw [he]
        simp only [reduceCtorEq, false_and, if_false, Sum.elim_inl, zero_add, F]
        refine le_trans ?_ le_self_add
        have h1 : ((Ψ nσ (newHypothesis T E) : ℕ) : ℝ) + 1 ≤ Ψ nσ s := by exact_mod_cast hΨ
        calc ENNReal.ofReal (a * Ψ nσ (newHypothesis T E))
              + consistentMass read C D (newHypothesis T E)
            ≤ ENNReal.ofReal (a * Ψ nσ (newHypothesis T E)) + ENNReal.ofReal a := by
              gcongr; exact hfresh T E
          _ ≤ ENNReal.ofReal (a * Ψ nσ s) := by
              rw [← ENNReal.ofReal_add (by positivity) ha.le]
              refine ENNReal.ofReal_le_ofReal ?_
              nlinarith
      · have hlen := stuckString_length read ⟨s.tree, s.edges⟩ C.walkStart x
        have hcnt := dis_count read C s x
        rcases step_same read C hC hs hc hlen with ⟨he, -, -⟩ | ⟨he, -⟩ | ⟨he, hbelow, -⟩ <;>
          rw [he]
        · simp only [reduceCtorEq, false_and, if_false, Sum.elim_inl, zero_add, F]
          refine add_le_add (le_of_eq rfl) (le_of_eq ?_)
          simp only [g, consistentMass]
          rw [hcnt]
          rfl
        · simp
        · simp only [Sum.elim_inr, add_zero]
          split_ifs with hp
          · refine le_trans ?_ le_add_self
            simp only [g, if_pos hp.2]
            rw [← ENNReal.ofReal_one]
            refine ENNReal.ofReal_le_ofReal ?_
            have : 1 / a ≤ likelihoodRatio ε m (s.probes + 1) (s.disagreements
                + if decide ((probeOrAgree false read s.tree s.edges C.walkStart x).clean = false)
                  then 1 else 0) := by
              rw [← hcnt]; exact hbelow
            rw [div_le_iff₀ ha] at this
            linarith
          · exact bot_le
    calc _ ≤ ∫⁻ x, ENNReal.ofReal (a * Ψ nσ s) + g x ∂D := lintegral_mono hpt
      _ = ENNReal.ofReal (a * Ψ nσ s) + ∫⁻ x, g x ∂D := by
          rw [lintegral_add_left measurable_const, lintegral_const, measure_univ, mul_one]
      _ ≤ F s := by
          simp only [F]
          gcongr
          simp only [g, consistentMass]
          split_ifs with hp
          · have hq : (D.real {x | (probeOrAgree false read s.tree s.edges C.walkStart x).clean
                = false} - ε) * (m - ε) ≤ 0 :=
              mul_nonpos_of_nonneg_of_nonpos (by simp only [Bad] at hp; linarith)
                (by linarith)
            exact lintegral_ratio_succ D
              (fun x => (probeOrAgree false read s.tree s.edges C.walkStart x).clean = false)
              (by linarith) hε1 hm0.le (by linarith) ha.le hq
              (hs.dis_le_undecided.trans hs.undecided_le_probes)
          · simp
  refine (anyStep_le read C D _ (Inv read C) F (fun s x s' hs h => (step_inv read C hI hC hs h).1)
    hF N initialState (inv_start read C (one_le_ends hC))).trans ?_
  have hΨ : (Ψ nσ (initialState : Level2State α) : ℝ) + 1 ≤ stretches nσ (Fintype.card α) := by
    exact_mod_cast Ψ_start nσ (α := α)
  calc F initialState ≤ ENNReal.ofReal (a * Ψ nσ (initialState : Level2State α))
        + ENNReal.ofReal a := add_le_add le_rfl (hfresh _ _)
    _ ≤ _ := by
        rw [← ENNReal.ofReal_add (by positivity) ha.le]
        refine ENNReal.ofReal_le_ofReal ?_
        nlinarith

end Run

/-! ## The theorem -/

section Theorem

variable [Fintype α] (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] {σ : Type*} [Fintype σ]
  {M : DFA α σ} {side : σ → Bool}

/-- A run that does not end well has a bad step: a consistent ending at a hypothesis that does not
end clean on more than `disagreementRate` of the probes, or a harvest at one stopping at an
undecided read on fewer than `harvestRate` of them. -/
theorem bad_any (hI : IdealReads read M side) (hC : TestsOK C) (hy : 0 < C.harvestRate) :
    ∀ (xs : List (FreeMonoid α)) (s : Level2State α), Inv read C s →
      Θ C (Fintype.card σ) s < xs.length →
      ¬ EndsWell M (AllUndecided read M) D C.disagreementRate
          {x | Disagrees read ⟨(runProbes read C s xs).1.tree, (runProbes read C s xs).1.edges⟩
            C.walkStart x} C.walkStart (fun _ => harvestLaw read D C.walkStart
              ⟨(runProbes read C s xs).1.tree, (runProbes read C s xs).1.edges⟩)
          C.harvestRate (runProbes read C s xs).2 →
      AnyStep read C (fun s x => applyProbe read C s x = .inr .consistent ∧ Bad read C D s) s xs
        ∨ AnyStep read C (fun s x => applyProbe read C s x = .inr (.harvest ())
          ∧ hitRate read C D s < C.harvestRate) s xs
  | [], s, _, h, _ => by simp at h
  | x :: xs, s, hs, hΘ, hbad => by
    rcases hst : applyProbe read C s x with s' | e
    · obtain ⟨hs', hlt⟩ := step_inv read C hI hC hs hst
      simp only [runProbes, hst] at hbad
      rcases bad_any hI hC hy xs s' hs' (by simp only [List.length_cons] at hΘ; omega) hbad
        with h | h
      · exact Or.inl (Or.inr ⟨s', hst, h⟩)
      · exact Or.inr (Or.inr ⟨s', hst, h⟩)
    · simp only [runProbes, hst] at hbad
      rcases e with _ | ⟨⟩ | _
      · by_contra hno
        simp only [AnyStep, hst, true_and, not_or] at hno
        apply hbad
        show D.real _ ≤ _
        have hB : ¬ Bad read C D s := hno.1.1
        unfold Bad at hB
        exact (disagrees_le read C D).trans (not_lt.1 hB)
      · by_contra hno
        simp only [AnyStep, hst, true_and, not_or] at hno
        have hle : C.harvestRate ≤ hitRate read C D s := not_lt.1 hno.2.1
        apply hbad
        refine endsWell_harvest read D ⟨s.tree, s.edges⟩ C.walkStart hI hs.edges _ _ hy
          (hle.trans ?_)
        have hfin : harvestLaw read D C.walkStart ⟨s.tree, s.edges⟩ Set.univ ≠ ⊤ :=
          ne_top_of_le_ne_top (by simp) (harvestLaw_univ_le read D ⟨s.tree, s.edges⟩ C.walkStart)
        exact ENNReal.toReal_mono hfin (stuck_le_harvestLaw read D ⟨s.tree, s.edges⟩ C.walkStart)
      · exact absurd hst (applyProbe_ne_failed read C s x)

end Theorem

theorem idealRoundCorrect_holds : IdealRoundCorrect := by
  intro α _ _ σ _ M read side C P hI hy hym hmε hε1 ha ha1 hP D _
  have hC : TestsOK C := ⟨hy, hym, hmε, hε1, ha, ha1⟩
  have hΘ : Θ C (Fintype.card σ) (initialState : Level2State α) < P :=
    (Θ_start C _ (one_le_ends hC)).trans_le hP
  have hsub : {draws : Fin P → FreeMonoid α | let result := indecisiveLStarLevel2 C read draws
        ¬ EndsWell M (AllUndecided read M) D C.disagreementRate
          {x | Disagrees read result.hypothesis C.walkStart x}
          C.walkStart (fun _ => harvestLaw read D C.walkStart result.hypothesis) C.harvestRate
          result.ending}
      ⊆ {xs | AnyStep read C (fun s x => applyProbe read C s x = .inr .consistent
            ∧ Bad read C D s) initialState (List.ofFn xs)}
        ∪ {xs | AnyStep read C (fun s x => applyProbe read C s x = .inr (.harvest ())
            ∧ hitRate read C D s < C.harvestRate) initialState (List.ofFn xs)} := by
    intro xs hxs
    exact bad_any read C D hI hC hy (List.ofFn xs) initialState
      (inv_start read C (one_le_ends hC)) (by simpa using hΘ) hxs
  have h1 := falseConsistent_le read C D hI hC P (σ := σ)
  have h2 := falseHarvest_le read C D hI hC P (σ := σ)
  have hS : (0 : ℝ) ≤ stretches (Fintype.card σ) (Fintype.card α) * C.testLevel := by
    have := ha.le; positivity
  refine (measureReal_mono hsub).trans (ENNReal.toReal_le_of_le_ofReal (by positivity)
    ((measure_union_le _ _).trans ?_))
  refine (add_le_add h1 h2).trans (le_of_eq ?_)
  rw [← ENNReal.ofReal_add hS hS]
  congr 1
  ring

end Ideal

end OrthoDFA
