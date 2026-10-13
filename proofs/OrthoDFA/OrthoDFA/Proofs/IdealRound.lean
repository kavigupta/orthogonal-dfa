import OrthoDFA.Proofs.IdealHarvest
import OrthoDFA.Proofs.RateTest
import Mathlib.Probability.ProductMeasure
import Mathlib.Analysis.Complex.ExponentialBounds

/-!
# The idealized round: the proofs

Every step that does not end the round lowers `Θ`, and the start's `Θ` is below the budget, so the
round ends on its probes, and never fails.

The chance part (`anyStep_le`): a potential that pays for a step's event and, on average over the
fresh probe, for its own value after the step bounds the chance that some step of the run is such
an event. A hypothesis is kept for at most `ends` probes, and at most `Ψ` hypotheses follow,
each starting its potential afresh. Two potentials, each the chance that one of the stretch's looks
to come errs, at most the test's level `a` a look (`OrthoDFA.RateTestSound`):
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
  · rcases step_same read C hC hs hc with ⟨he, hinv, -⟩ | ⟨he, -⟩ | ⟨he, -, -⟩ <;> rw [he] at h
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

open scoped Classical in
/-- How many of `ys` satisfy `P`. -/
noncomputable def countOf {X : Type*} (P : X → Prop) {j : ℕ} (ys : Fin j → X) : ℕ :=
  (univ.filter fun i => P (ys i)).card

/-- How many of the probes `ys` stop at an undecided read under the hypothesis of `s`. -/
noncomputable def hits (s : Level2State α) {j : ℕ} (ys : Fin j → FreeMonoid α) : ℕ :=
  countOf (fun x => stuckString read ⟨s.tree, s.edges⟩ C.walkStart x ≠ []) ys

theorem hits_cons (s : Level2State α) {j : ℕ} (x : FreeMonoid α) (ys : Fin j → FreeMonoid α) :
    hits read C s (Fin.cons x ys : Fin (j + 1) → FreeMonoid α)
      = (stuckString read ⟨s.tree, s.edges⟩ C.walkStart x).length + hits read C s ys := by
  classical
  unfold hits countOf
  rw [card_filter, card_filter, Fin.sum_univ_succ]
  simp only [Fin.cons_zero, Fin.cons_succ]
  have hx := stuckString_length read ⟨s.tree, s.edges⟩ C.walkStart x
  congr 1
  · split_ifs with h
    · have := List.length_pos_iff.2 h; omega
    · simp only [ne_eq, not_not] at h; simp [h]
  · exact sum_congr rfl fun i _ => by congr

theorem pi_hits_succ (s : Level2State α) (j : ℕ) (Q : ℕ → Prop) :
    (Measure.pi fun _ : Fin (j + 1) => D) {ys | Q (hits read C s ys)}
      = ∫⁻ x, (Measure.pi fun _ : Fin j => D)
          {ys | Q ((stuckString read ⟨s.tree, s.edges⟩ C.walkStart x).length
            + hits read C s ys)} ∂D := by
  rw [pi_succ_apply]
  congr 1
  funext x
  congr 1
  ext ys
  simp only [Set.mem_ofPred_eq, hits_cons]

theorem pi_hits_zero (s : Level2State α) (Q : ℕ → Prop) (hQ : Q 0) :
    (Measure.pi fun _ : Fin 0 => D) {ys | Q (hits read C s ys)} = 1 := by
  have : {ys : Fin 0 → FreeMonoid α | Q (hits read C s ys)} = Set.univ := by
    ext ys
    simp only [Set.mem_ofPred_eq, Set.mem_univ, iff_true]
    have : hits read C s ys = 0 := by simp [hits, countOf]
    rwa [this]
  rw [this, measure_univ]

variable {σ : Type*} [Fintype σ] {M : DFA α σ} {side : σ → Bool}

/-- How many of the probes `ys` do not end clean under the hypothesis of `s`. -/
noncomputable def dhits (s : Level2State α) {j : ℕ} (ys : Fin j → FreeMonoid α) : ℕ :=
  countOf (fun x => (probeOrAgree false read s.tree s.edges C.walkStart x).clean = false) ys

theorem dhits_cons (s : Level2State α) {j : ℕ} (x : FreeMonoid α) (ys : Fin j → FreeMonoid α) :
    dhits read C s (Fin.cons x ys : Fin (j + 1) → FreeMonoid α)
      = (if (probeOrAgree false read s.tree s.edges C.walkStart x).clean then 0 else 1)
        + dhits read C s ys := by
  classical
  unfold dhits countOf
  rw [card_filter, card_filter, Fin.sum_univ_succ]
  simp only [Fin.cons_zero, Fin.cons_succ]
  congr 1
  · cases (probeOrAgree false read s.tree s.edges C.walkStart x).clean <;> simp
  · exact sum_congr rfl fun i _ => by congr

theorem pi_dhits_succ (s : Level2State α) (j : ℕ) (Q : ℕ → Prop) :
    (Measure.pi fun _ : Fin (j + 1) => D) {ys | Q (dhits read C s ys)}
      = ∫⁻ x, (Measure.pi fun _ : Fin j => D)
          {ys | Q ((if (probeOrAgree false read s.tree s.edges C.walkStart x).clean then 0 else 1)
            + dhits read C s ys)} ∂D := by
  rw [pi_succ_apply]
  congr 1
  funext x
  congr 1
  ext ys
  simp only [Set.mem_ofPred_eq, dhits_cons]

theorem pi_dhits_zero (s : Level2State α) (Q : ℕ → Prop) (hQ : Q 0) :
    (Measure.pi fun _ : Fin 0 => D) {ys | Q (dhits read C s ys)} = 1 := by
  have : {ys : Fin 0 → FreeMonoid α | Q (dhits read C s ys)} = Set.univ := by
    ext ys
    simp only [Set.mem_ofPred_eq, Set.mem_univ, iff_true]
    have : dhits read C s ys = 0 := by simp [dhits, countOf]
    rwa [this]
  rw [this, measure_univ]

/-- The hypothesis of `s` does not end clean on more than `disagreementRate` of the probes. -/
def Bad (s : Level2State α) : Prop :=
  C.disagreementRate
    < D.real {x | (probeOrAgree false read s.tree s.edges C.walkStart x).clean = false}

open scoped Classical in
/-- Over a stretch's probes to come, the chance each look fires the rate test. -/
noncomputable def fireSum (s : Level2State α) : ℝ≥0∞ :=
  ∑ j ∈ range (C.ends - s.probes), (Measure.pi fun _ : Fin (j + 1) => D)
    {ys | binomSfGe (s.probes + (j + 1)) C.harvestRate (s.undecided + hits read C s ys)
      < C.testLevel}

open scoped Classical in
/-- What remains of a stretch's chance to fire the rate test at a rate below `y`. -/
noncomputable def fireMass (s : Level2State α) : ℝ≥0∞ :=
  if hitRate read C D s < C.harvestRate then fireSum read C D s else 0

open scoped Classical in
/-- A harvest at a hypothesis stopping at an undecided read on fewer than `y` of the probes. -/
theorem falseHarvest_le (hI : IdealReads read M side) (hC : TestsOK C)
    (hy1 : C.harvestRate ≤ 1) (ha : 0 ≤ C.testLevel) (N : ℕ) :
    (Measure.pi fun _ : Fin N => D) {xs | AnyStep read C
        (fun s x => applyProbe read C s x = .inr (.harvest ())
          ∧ hitRate read C D s < C.harvestRate) initialState (List.ofFn xs)}
      ≤ ENNReal.ofReal (stretches (Fintype.card σ) (Fintype.card α) * C.ends
          * C.testLevel) := by
  set nσ := Fintype.card σ
  set y := C.harvestRate
  set a := C.testLevel
  set n0 := C.ends
  have hfresh : ∀ T E, fireMass read C D (newHypothesis T E) ≤ ENNReal.ofReal (n0 * a) := by
    intro T E
    unfold fireMass fireSum
    split_ifs with hp
    · simp only [newHypothesis, Nat.sub_zero, zero_add]
      refine (sum_le_card_nsmul _ _ (ENNReal.ofReal a) fun j _ => ?_).trans (le_of_eq ?_)
      · have := rateSide_above_le D (fun x => stuckString read ⟨T, E⟩ C.walkStart x ≠ []) y a 0
          (j + 1) hy1 ha hp.le
        rw [← ENNReal.ofReal_toReal (measure_ne_top _ _)]
        refine ENNReal.ofReal_le_ofReal (le_trans (le_of_eq ?_) this)
        rw [← measureReal_def]
        congr 1
        ext ys
        simp only [Set.mem_ofPred_eq]
        exact (rateSide_zero _ _ _ _).symm
      · rw [card_range, nsmul_eq_mul, ENNReal.ofReal_mul (Nat.cast_nonneg _),
          ENNReal.ofReal_natCast]
    · exact bot_le
  let F : Level2State α → ℝ≥0∞ := fun s =>
    ENNReal.ofReal (n0 * a * Ψ nσ s) + fireMass read C D s
  have hF : ∀ s, Inv read C s → ∫⁻ x, ((if (applyProbe read C s x = .inr (.harvest ())
      ∧ hitRate read C D s < y) then 1 else 0) + Sum.elim F (fun _ => 0) (applyProbe read C s x))
      ∂D ≤ F s := by
    intro s hs
    have hlt := hs.probes_lt
    let h : FreeMonoid α → ℝ≥0∞ := fun x =>
      ∑ j ∈ range (n0 - s.probes), (Measure.pi fun _ : Fin j => D)
        {ys | binomSfGe (s.probes + 1 + j) y (s.undecided
          + (stuckString read ⟨s.tree, s.edges⟩ C.walkStart x).length + hits read C s ys) < a}
    let g : FreeMonoid α → ℝ≥0∞ := fun x => if hitRate read C D s < y then h x else 0
    have hsplit : ∀ x, h x = (Measure.pi fun _ : Fin 0 => D) {ys | binomSfGe (s.probes + 1 + 0) y
          (s.undecided + (stuckString read ⟨s.tree, s.edges⟩ C.walkStart x).length
            + hits read C s ys) < a}
        + fireSum read C D (counted s (probeOrAgree false read s.tree s.edges C.walkStart x)) := by
      intro x
      simp only [h, fireSum]
      rw [show n0 - s.probes = n0 - (s.probes + 1) + 1 by omega, sum_range_succ', add_comm]
      rfl
    have hpt : ∀ x, ((if (applyProbe read C s x = .inr (.harvest ()) ∧ hitRate read C D s < y)
        then 1 else 0) + Sum.elim F (fun _ => 0) (applyProbe read C s x))
        ≤ ENNReal.ofReal (n0 * a * Ψ nσ s) + g x := by
      intro x
      by_cases hc : Changes (probeOrAgree false read s.tree s.edges C.walkStart x)
      · obtain ⟨T, E, he, -, hΨ⟩ := step_change read C hI (one_le_ends hC) hs hc
        rw [he]
        simp only [reduceCtorEq, false_and, if_false, Sum.elim_inl, zero_add, F]
        refine le_trans ?_ le_self_add
        have h1 : ((Ψ nσ (newHypothesis T E) : ℕ) : ℝ) + 1 ≤ Ψ nσ s := by exact_mod_cast hΨ
        calc ENNReal.ofReal (n0 * a * Ψ nσ (newHypothesis T E))
              + fireMass read C D (newHypothesis T E)
            ≤ ENNReal.ofReal (n0 * a * Ψ nσ (newHypothesis T E)) + ENNReal.ofReal (n0 * a) := by
              gcongr; exact hfresh T E
          _ ≤ ENNReal.ofReal (n0 * a * Ψ nσ s) := by
              rw [← ENNReal.ofReal_add (by positivity) (by positivity)]
              refine ENNReal.ofReal_le_ofReal ?_
              have : (0 : ℝ) ≤ n0 * a := by positivity
              nlinarith
      · by_cases hp : hitRate read C D s < y
        · have hp' : hitRate read C D
              (counted s (probeOrAgree false read s.tree s.edges C.walkStart x)) < y := hp
          simp only [g, if_pos hp, hsplit x]
          rcases step_same read C hC hs hc
              with ⟨he, -, -⟩ | ⟨he, hfire⟩ | ⟨he, -, -⟩ <;> rw [he]
          · simp only [reduceCtorEq, false_and, if_false, Sum.elim_inl, zero_add, F]
            rw [show fireMass read C D (counted s (probeOrAgree false read s.tree s.edges
              C.walkStart x)) = fireSum read C D (counted s (probeOrAgree false read s.tree s.edges
              C.walkStart x)) from if_pos hp']
            exact add_le_add (le_of_eq rfl) le_add_self
          · simp only [hp, and_true, if_true, Sum.elim_inr, add_zero]
            refine le_trans (le_of_eq ?_) (le_trans le_self_add le_add_self)
            rw [eq_comm]
            refine pi_hits_zero read C D s (fun m => binomSfGe (s.probes + 1 + 0) y (s.undecided
              + (stuckString read ⟨s.tree, s.edges⟩ C.walkStart x).length + m) < a) ?_
            simpa [Fires, counted, stuckString] using hfire
          · simp
        · simp only [g, if_neg hp]
          rcases step_same read C hC hs hc
              with ⟨he, -, -⟩ | ⟨he, -⟩ | ⟨he, -, -⟩ <;> rw [he]
          · have hp' : ¬ hitRate read C D
                (counted s (probeOrAgree false read s.tree s.edges C.walkStart x)) < y := hp
            simp only [reduceCtorEq, false_and, if_false, Sum.elim_inl, zero_add, F]
            rw [show fireMass read C D (counted s (probeOrAgree false read s.tree s.edges
              C.walkStart x)) = 0 from if_neg hp', add_zero, add_zero]
            exact le_rfl
          · simp [hp]
          · simp
    calc _ ≤ ∫⁻ x, ENNReal.ofReal (n0 * a * Ψ nσ s) + g x ∂D := lintegral_mono hpt
      _ = ENNReal.ofReal (n0 * a * Ψ nσ s) + ∫⁻ x, g x ∂D := by
          rw [lintegral_add_left measurable_const, lintegral_const, measure_univ, mul_one]
      _ ≤ F s := by
          simp only [F]
          gcongr
          simp only [g, fireMass, fireSum, h]
          split_ifs with hp
          · rw [lintegral_finset_sum _ fun _ _ => measurable_of_countable _]
            refine le_of_eq (sum_congr rfl fun j _ => ?_)
            rw [pi_hits_succ read C D s j (fun m => binomSfGe (s.probes + (j + 1)) C.harvestRate
              (s.undecided + m) < C.testLevel)]
            refine lintegral_congr fun x => ?_
            congr 1
            ext ys
            simp only [Set.mem_ofPred_eq]
            constructor <;> intro hh <;> convert hh using 2 <;> ring
          · simp
  refine (anyStep_le read C D _ (Inv read C) F (fun s x s' hs h => (step_inv read C hI hC hs h).1)
    hF N initialState (inv_start read C (one_le_ends hC))).trans ?_
  have hΨ : (Ψ nσ (initialState : Level2State α) : ℝ) + 1 ≤ stretches nσ (Fintype.card α) := by
    exact_mod_cast Ψ_start nσ (α := α)
  calc F initialState ≤ ENNReal.ofReal (n0 * a * Ψ nσ (initialState : Level2State α))
        + ENNReal.ofReal (n0 * a) := add_le_add le_rfl (hfresh _ _)
    _ ≤ _ := by
        rw [← ENNReal.ofReal_add (by positivity) (by positivity)]
        refine ENNReal.ofReal_le_ofReal ?_
        have : (0 : ℝ) ≤ n0 * a := by positivity
        nlinarith

open scoped Classical in
/-- Over a stretch's probes to come, the chance each look calls the disagreements below. -/
noncomputable def belowSum (s : Level2State α) : ℝ≥0∞ :=
  ∑ j ∈ range (C.ends - s.probes), (Measure.pi fun _ : Fin (j + 1) => D)
    {ys | rateSide C.disagreementRate C.testLevel 0 (s.probes + (j + 1))
      (s.disagreements + dhits read C s ys) = some false}

open scoped Classical in
/-- What remains of a stretch's chance to call the disagreements below at a `Bad` hypothesis. -/
noncomputable def belowMass (s : Level2State α) : ℝ≥0∞ :=
  if Bad read C D s then belowSum read C D s else 0

open scoped Classical in
/-- A consistent ending at a hypothesis that does not end clean on more than `disagreementRate` of
the probes. -/
theorem falseConsistent_le (hI : IdealReads read M side) (hC : TestsOK C) (N : ℕ) :
    (Measure.pi fun _ : Fin N => D) {xs | AnyStep read C
        (fun s x => applyProbe read C s x = .inr .consistent ∧ Bad read C D s) initialState
          (List.ofFn xs)}
      ≤ ENNReal.ofReal (stretches (Fintype.card σ) (Fintype.card α) * C.ends
          * C.testLevel) := by
  set nσ := Fintype.card σ
  set ε := C.disagreementRate
  set a := C.testLevel
  set n0 := C.ends
  have ha : 0 ≤ a := hC.testLevel_pos.le
  have hε0 : 0 ≤ ε := by linarith [hC.harvestRate_nonneg, hC.harvestRate_lt]
  have hfresh : ∀ T E, belowMass read C D (newHypothesis T E) ≤ ENNReal.ofReal (n0 * a) := by
    intro T E
    unfold belowMass belowSum
    split_ifs with hp
    · simp only [newHypothesis, Nat.sub_zero, zero_add]
      refine (sum_le_card_nsmul _ _ (ENNReal.ofReal a) fun j _ => ?_).trans (le_of_eq ?_)
      · have := rateSide_below_le D
          (fun x => (probeOrAgree false read T E C.walkStart x).clean = false) ε a 0
          (j + 1) hε0 ha hp.le
        rw [← ENNReal.ofReal_toReal (measure_ne_top _ _)]
        exact ENNReal.ofReal_le_ofReal (le_trans (le_of_eq rfl) this)
      · rw [card_range, nsmul_eq_mul, ENNReal.ofReal_mul (Nat.cast_nonneg _),
          ENNReal.ofReal_natCast]
    · exact bot_le
  let F : Level2State α → ℝ≥0∞ := fun s =>
    ENNReal.ofReal (n0 * a * Ψ nσ s) + belowMass read C D s
  have hF : ∀ s, Inv read C s → ∫⁻ x, ((if (applyProbe read C s x = .inr .consistent
      ∧ Bad read C D s) then 1 else 0) + Sum.elim F (fun _ => 0) (applyProbe read C s x))
      ∂D ≤ F s := by
    intro s hs
    have hlt := hs.probes_lt
    let cl : FreeMonoid α → ℕ := fun x =>
      if (probeOrAgree false read s.tree s.edges C.walkStart x).clean then 0 else 1
    let h : FreeMonoid α → ℝ≥0∞ := fun x =>
      ∑ j ∈ range (n0 - s.probes), (Measure.pi fun _ : Fin j => D)
        {ys | rateSide ε a 0 (s.probes + 1 + j) (s.disagreements + cl x + dhits read C s ys)
          = some false}
    let g : FreeMonoid α → ℝ≥0∞ := fun x => if Bad read C D s then h x else 0
    have hsplit : ∀ x, h x = (Measure.pi fun _ : Fin 0 => D) {ys | rateSide ε a 0
          (s.probes + 1 + 0) (s.disagreements + cl x + dhits read C s ys) = some false}
        + belowSum read C D (counted s (probeOrAgree false read s.tree s.edges C.walkStart x)) := by
      intro x
      simp only [h, belowSum]
      rw [show n0 - s.probes = n0 - (s.probes + 1) + 1 by omega, sum_range_succ', add_comm]
      rfl
    have hpt : ∀ x, ((if (applyProbe read C s x = .inr .consistent ∧ Bad read C D s)
        then 1 else 0) + Sum.elim F (fun _ => 0) (applyProbe read C s x))
        ≤ ENNReal.ofReal (n0 * a * Ψ nσ s) + g x := by
      intro x
      by_cases hc : Changes (probeOrAgree false read s.tree s.edges C.walkStart x)
      · obtain ⟨T, E, he, -, hΨ⟩ := step_change read C hI (one_le_ends hC) hs hc
        rw [he]
        simp only [reduceCtorEq, false_and, if_false, Sum.elim_inl, zero_add, F]
        refine le_trans ?_ le_self_add
        have h1 : ((Ψ nσ (newHypothesis T E) : ℕ) : ℝ) + 1 ≤ Ψ nσ s := by exact_mod_cast hΨ
        calc ENNReal.ofReal (n0 * a * Ψ nσ (newHypothesis T E))
              + belowMass read C D (newHypothesis T E)
            ≤ ENNReal.ofReal (n0 * a * Ψ nσ (newHypothesis T E)) + ENNReal.ofReal (n0 * a) := by
              gcongr; exact hfresh T E
          _ ≤ ENNReal.ofReal (n0 * a * Ψ nσ s) := by
              rw [← ENNReal.ofReal_add (by positivity) (by positivity)]
              refine ENNReal.ofReal_le_ofReal ?_
              have : (0 : ℝ) ≤ n0 * a := by positivity
              nlinarith
      · by_cases hp : Bad read C D s
        · have hp' : Bad read C D
              (counted s (probeOrAgree false read s.tree s.edges C.walkStart x)) := hp
          simp only [g, if_pos hp, hsplit x]
          rcases step_same read C hC hs hc
              with ⟨he, -, -⟩ | ⟨he, -⟩ | ⟨he, hbelow, -⟩ <;> rw [he]
          · simp only [reduceCtorEq, false_and, if_false, Sum.elim_inl, zero_add, F]
            rw [show belowMass read C D (counted s (probeOrAgree false read s.tree s.edges
              C.walkStart x)) = belowSum read C D (counted s (probeOrAgree false read s.tree
              s.edges C.walkStart x)) from if_pos hp']
            exact add_le_add (le_of_eq rfl) le_add_self
          · simp
          · simp only [hp, and_true, if_true, Sum.elim_inr, add_zero]
            refine le_trans (le_of_eq ?_) (le_trans le_self_add le_add_self)
            rw [eq_comm]
            refine pi_dhits_zero read C D s (fun m => rateSide ε a 0 (s.probes + 1 + 0)
              (s.disagreements + cl x + m) = some false) ?_
            exact hbelow
        · simp only [g, if_neg hp]
          rcases step_same read C hC hs hc
              with ⟨he, -, -⟩ | ⟨he, -⟩ | ⟨he, -, -⟩ <;> rw [he]
          · have hp' : ¬ Bad read C D
                (counted s (probeOrAgree false read s.tree s.edges C.walkStart x)) := hp
            simp only [reduceCtorEq, false_and, if_false, Sum.elim_inl, zero_add, F]
            rw [show belowMass read C D (counted s (probeOrAgree false read s.tree s.edges
              C.walkStart x)) = 0 from if_neg hp', add_zero, add_zero]
            exact le_rfl
          · simp
          · simp [hp]
    calc _ ≤ ∫⁻ x, ENNReal.ofReal (n0 * a * Ψ nσ s) + g x ∂D := lintegral_mono hpt
      _ = ENNReal.ofReal (n0 * a * Ψ nσ s) + ∫⁻ x, g x ∂D := by
          rw [lintegral_add_left measurable_const, lintegral_const, measure_univ, mul_one]
      _ ≤ F s := by
          simp only [F]
          gcongr
          simp only [g, belowMass, belowSum, h]
          split_ifs with hp
          · rw [lintegral_finset_sum _ fun _ _ => measurable_of_countable _]
            refine le_of_eq (sum_congr rfl fun j _ => ?_)
            rw [pi_dhits_succ read C D s j (fun m => rateSide ε a 0 (s.probes + (j + 1))
              (s.disagreements + m) = some false)]
            refine lintegral_congr fun x => ?_
            congr 1
            ext ys
            simp only [Set.mem_ofPred_eq, cl]
            constructor <;> intro hh <;> convert hh using 2 <;> ring
          · simp
  refine (anyStep_le read C D _ (Inv read C) F (fun s x s' hs h => (step_inv read C hI hC hs h).1)
    hF N initialState (inv_start read C (one_le_ends hC))).trans ?_
  have hΨ : (Ψ nσ (initialState : Level2State α) : ℝ) + 1 ≤ stretches nσ (Fintype.card α) := by
    exact_mod_cast Ψ_start nσ (α := α)
  calc F initialState ≤ ENNReal.ofReal (n0 * a * Ψ nσ (initialState : Level2State α))
        + ENNReal.ofReal (n0 * a) := add_le_add le_rfl (hfresh _ _)
    _ ≤ _ := by
        rw [← ENNReal.ofReal_add (by positivity) (by positivity)]
        refine ENNReal.ofReal_le_ofReal ?_
        have : (0 : ℝ) ≤ n0 * a := by positivity
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
  intro α _ _ σ _ M read side C P hI hy hyε hε1 ha ha1 hP D _
  have hC : TestsOK C := ⟨hy.le, hyε, hε1, ha, ha1⟩
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
  have h2 := falseHarvest_le read C D hI hC (by linarith) ha.le P (σ := σ)
  have hS : (0 : ℝ) ≤ stretches (Fintype.card σ) (Fintype.card α) * C.ends * C.testLevel := by
    have := ha.le; positivity
  refine (measureReal_mono hsub).trans (ENNReal.toReal_le_of_le_ofReal (by positivity)
    ((measure_union_le _ _).trans ?_))
  refine (add_le_add h1 h2).trans (le_of_eq ?_)
  rw [← ENNReal.ofReal_add hS hS]
  congr 1
  simp only [Level2Settings.ends]
  ring

end Ideal

end OrthoDFA
