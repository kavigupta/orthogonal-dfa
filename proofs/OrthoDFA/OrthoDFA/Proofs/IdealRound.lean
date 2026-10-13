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
an event. A hypothesis is kept for at most `probesToEnd` probes, and at most `Ψ` hypotheses follow,
each starting its potential afresh. Three potentials:
* a consistent ending at a hypothesis that changes on more than `ε` of the probes needs
  `probesToEnd` probes in a row that keep it, chance `(1 − ε)^n`;
* a harvest at a hypothesis stopping at an undecided read on fewer than `y` of the probes: the
  chance that each of the stretch's looks to come fires the rate test, at most `a` a look;
* a consistent ending at one stopping at an undecided read on at least `r` of them: the chance
  that the stretch's count at its last look leaves the test silent, `P(Bin(n, r) < K)`.
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
theorem inv_start (hn : 1 ≤ C.probesToEnd) : Inv read C (initialState : Level2State α) := by
  refine inv_fresh read C hn (fun _ _ _ _ _ h => by simp [initialState, newHypothesis] at h) ?_
  intro p hp
  simp only [DTree.leaves, List.map_cons, List.map_nil, List.cons_append, List.nil_append,
    List.mem_cons, List.not_mem_nil, or_false] at hp
  rcases hp with rfl | rfl
  · exact Or.inl rfl
  · exact Or.inr (Or.inl rfl)

/-- A step that goes on keeps `Inv` and lowers `Θ`. -/
theorem step_inv (hI : IdealReads read M side) (hn : 1 ≤ C.probesToEnd)
    {s s' : Level2State α} (hs : Inv read C s) {x : FreeMonoid α}
    (h : applyProbe read C s x = .inl s') :
    Inv read C s' ∧ Θ C (Fintype.card σ) s' < Θ C (Fintype.card σ) s := by
  by_cases hc : Changes (probeOrAgree false read s.tree s.edges C.walkStart x)
  · obtain ⟨T, E, he, hinv, hΨ⟩ := step_change read C hI hn hs hc
    rw [he, Sum.inl.injEq] at h
    subst h
    exact ⟨hinv, Θ_lt_change C _ hs.probes_lt T E hΨ⟩
  · rcases step_same read C hc with ⟨he, hlt, hnf⟩ | ⟨he, -⟩ | ⟨he, -, -⟩ <;> rw [he] at h
    · rw [Sum.inl.injEq] at h
      subst h
      exact ⟨inv_counted read C hs _ hlt, Θ_lt_same C _ rfl rfl rfl hlt⟩
    · cases h
    · cases h

theorem applyProbe_ne_failed (s : Level2State α) (x : FreeMonoid α) :
    applyProbe read C s x ≠ .inr .failed := by
  classical
  unfold applyProbe
  split
  · unfold keep; dsimp only; split_ifs <;> simp
  · unfold keep; dsimp only; split_ifs <;> simp
  · split <;> simp
  · simp

end Ends

/-! ## The hit rate -/

section Rates

variable [Fintype α] (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]

/-- The chance a probe stops at an undecided read under the hypothesis of `s`. -/
noncomputable def hitRate (s : Level2State α) : ℝ :=
  D.real {x | stuckString read ⟨s.tree, s.edges⟩ C.walkStart x ≠ []}

/-- A probe that disagrees changes the hypothesis, or stops at an undecided read. -/
theorem disagrees_le {s : Level2State α} (hs : Inv read C s) :
    D.real {x | Disagrees read ⟨s.tree, s.edges⟩ C.walkStart x}
      ≤ D.real {x | Changes (probeOrAgree false read s.tree s.edges C.walkStart x)}
        + hitRate read C D s := by
  classical
  have hsub : {x | Disagrees read ⟨s.tree, s.edges⟩ C.walkStart x}
      ⊆ {x | Changes (probeOrAgree false read s.tree s.edges C.walkStart x)}
        ∪ {x | stuckString read ⟨s.tree, s.edges⟩ C.walkStart x ≠ []} := by
    intro x hx
    have hcl := probeOrAgree_disagrees false read s.tree s.edges C.walkStart hx
    have hOK := probeOrAgree_ok false read s.tree s.edges C.walkStart hs.edges x
    show Changes (probeOrAgree false read s.tree s.edges C.walkStart x)
      ∨ (probeOrAgree false read s.tree s.edges C.walkStart x).undecidedStrings ≠ []
    revert hcl hOK
    generalize probeOrAgree false read s.tree s.edges C.walkStart x = o
    intro hcl hOK
    cases o with
    | agree => simp [Outcome.clean] at hcl
    | wrongEdge => exact Or.inl trivial
    | unlearnedEdge => exact Or.inl trivial
    | undecided k zs cl =>
      simp only [Outcome.clean] at hcl
      subst hcl
      exact Or.inr (hOK.2.1 rfl)
  calc _ ≤ _ := measureReal_mono hsub
    _ ≤ _ := measureReal_union_le _ _

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

open scoped Classical in
/-- Over `n` fresh probes, `K` or more stop at an undecided read with chance `P(Bin(n, p) ≥ K)`. -/
theorem pi_hits_ge (s : Level2State α) (n K : ℕ) :
    (Measure.pi fun _ : Fin n => D).real {ys | K ≤ hits read C s ys}
      = binomSfGe n (hitRate read C D s) K := by
  have := pi_count_ge D (fun x => stuckString read ⟨s.tree, s.edges⟩ C.walkStart x ≠ [])
    (univ : Finset (Fin n)) K
  rw [card_univ, Fintype.card_fin] at this
  exact this

variable {σ : Type*} [Fintype σ] {M : DFA α σ} {side : σ → Bool}

/-- The hypothesis of `s` changes on more than `ε` of the probes. -/
def Bad (ε : ℝ) (s : Level2State α) : Prop :=
  ε < D.real {x | Changes (probeOrAgree false read s.tree s.edges C.walkStart x)}

open scoped Classical in
/-- A consistent ending at a hypothesis that changes on more than `ε` of the probes. -/
theorem falseConsistent_le (hI : IdealReads read M side) (hn : 1 ≤ C.probesToEnd) {ε : ℝ}
    (hε : ε ≤ 1) (N : ℕ) :
    (Measure.pi fun _ : Fin N => D) {xs | AnyStep read C
        (fun s x => applyProbe read C s x = .inr .consistent ∧ Bad read C D ε s)
        initialState (List.ofFn xs)}
      ≤ ENNReal.ofReal (stretches (Fintype.card σ) (Fintype.card α) * (1 - ε) ^ C.probesToEnd) := by
  set q := 1 - ε
  have hq0 : 0 ≤ q := by linarith
  set nσ := Fintype.card σ
  let F : Level2State α → ℝ≥0∞ := fun s => ENNReal.ofReal
    (Ψ nσ s * q ^ C.probesToEnd + if Bad read C D ε s then q ^ (C.probesToEnd - s.probes) else 0)
  have hF : ∀ s, Inv read C s → ∫⁻ x, ((if (applyProbe read C s x = .inr .consistent
      ∧ Bad read C D ε s) then 1 else 0) + Sum.elim F (fun _ => 0) (applyProbe read C s x)) ∂D
      ≤ F s := by
    intro s hs
    set S := {x | ¬ Changes (probeOrAgree false read s.tree s.edges C.walkStart x)}
    have hS : MeasurableSet S := (Set.to_countable _).measurableSet
    set m := C.probesToEnd - s.probes
    have hm : 1 ≤ m := by have := hs.probes_lt; omega
    set c : ℝ := if Bad read C D ε s then q ^ (m - 1) else 0
    have hc0 : 0 ≤ c := by positivity
    have hpt : ∀ x, ((if (applyProbe read C s x = .inr .consistent ∧ Bad read C D ε s) then 1
        else 0) + Sum.elim F (fun _ => 0) (applyProbe read C s x))
        ≤ ENNReal.ofReal (Ψ nσ s * q ^ C.probesToEnd)
          + S.indicator (fun _ => ENNReal.ofReal c) x := by
      intro x
      by_cases hc : Changes (probeOrAgree false read s.tree s.edges C.walkStart x)
      · obtain ⟨T, E, he, -, hΨ⟩ := step_change read C hI hn hs hc
        have hxS : x ∉ S := fun h => h hc
        rw [he, Set.indicator_of_notMem hxS, add_zero]
        simp only [reduceCtorEq, false_and, if_false, Sum.elim_inl, zero_add, F]
        refine ENNReal.ofReal_le_ofReal ?_
        have : ((Ψ nσ (newHypothesis T E) : ℕ) : ℝ) + 1 ≤ Ψ nσ s := by exact_mod_cast hΨ
        have hb : (if Bad read C D ε (newHypothesis T E) then q ^ (C.probesToEnd -
            (newHypothesis T E).probes) else 0) ≤ q ^ C.probesToEnd := by
          split_ifs
          · simp [newHypothesis]
          · positivity
        have := pow_nonneg hq0 C.probesToEnd
        nlinarith
      · have hxS : x ∈ S := hc
        rw [Set.indicator_of_mem hxS]
        rcases step_same read C hc with ⟨he, -, -⟩ | ⟨he, -⟩ | ⟨he, hN, -⟩ <;> rw [he]
        · simp only [reduceCtorEq, false_and, if_false, Sum.elim_inl, zero_add, F]
          rw [← ENNReal.ofReal_add (by positivity) hc0]
          refine ENNReal.ofReal_le_ofReal ?_
          have hp : C.probesToEnd
              - (counted s (probeOrAgree false read s.tree s.edges C.walkStart x)).probes
              = m - 1 := by simp [counted, m]; omega
          change Ψ nσ s * q ^ C.probesToEnd + (if Bad read C D ε s then q ^ (C.probesToEnd
              - (counted s (probeOrAgree false read s.tree s.edges C.walkStart x)).probes)
              else 0) ≤ _
          rw [hp]
        · simp
        · simp only [true_and, Sum.elim_inr, add_zero]
          split_ifs with hb
          · have : m = 1 := by simp [counted] at hN; omega
            simp only [c, if_pos hb, this, Nat.sub_self, pow_zero, ENNReal.ofReal_one]
            exact le_add_self
          · exact bot_le
    calc _ ≤ ∫⁻ x, ENNReal.ofReal (Ψ nσ s * q ^ C.probesToEnd)
          + S.indicator (fun _ => ENNReal.ofReal c) x ∂D := lintegral_mono hpt
      _ = ENNReal.ofReal (Ψ nσ s * q ^ C.probesToEnd) + ENNReal.ofReal c * D S := by
          rw [lintegral_add_left measurable_const, lintegral_const, measure_univ, mul_one,
            lintegral_indicator_const hS]
      _ ≤ F s := by
          simp only [F]
          rw [ENNReal.ofReal_add (by positivity) (by positivity)]
          gcongr
          simp only [c]
          split_ifs with hb
          · have hDS : D S ≤ ENNReal.ofReal q := by
              rw [← ENNReal.ofReal_toReal (measure_ne_top D S)]
              refine ENNReal.ofReal_le_ofReal ?_
              have h1 : D.real S = 1 - D.real Sᶜ := by
                rw [measureReal_compl hS, probReal_univ]; ring
              have h2 : Sᶜ = {x | Changes (probeOrAgree false read s.tree s.edges C.walkStart x)} :=
                by ext x; simp [S]
              change D.real S ≤ q
              rw [h1, h2]
              unfold Bad at hb
              simp only [q]
              linarith
            calc ENNReal.ofReal (q ^ (m - 1)) * D S
                ≤ ENNReal.ofReal (q ^ (m - 1)) * ENNReal.ofReal q := by gcongr
              _ = ENNReal.ofReal (q ^ m) := by
                  rw [← ENNReal.ofReal_mul (by positivity), ← pow_succ]
                  congr 2
                  omega
          · simp
  refine (anyStep_le read C D _ (Inv read C) F (fun s x s' hs h => (step_inv read C hI hn hs h).1)
    hF N initialState (inv_start read C hn)).trans (ENNReal.ofReal_le_ofReal ?_)
  have hΨ : (Ψ nσ (initialState : Level2State α) : ℝ) + 1 ≤ stretches nσ (Fintype.card α) := by
    exact_mod_cast Ψ_start nσ (α := α)
  have hb : (if Bad read C D ε initialState then q ^ (C.probesToEnd -
      (initialState : Level2State α).probes) else 0) ≤ q ^ C.probesToEnd := by
    split_ifs
    · simp [initialState, newHypothesis]
    · positivity
  have := pow_nonneg hq0 C.probesToEnd
  nlinarith

open scoped Classical in
/-- Over a stretch's probes to come, the chance each look fires the rate test. -/
noncomputable def fireSum (s : Level2State α) : ℝ≥0∞ :=
  ∑ j ∈ range (C.probesToEnd - s.probes), (Measure.pi fun _ : Fin (j + 1) => D)
    {ys | binomSfGe (s.probes + (j + 1)) C.harvestRate (s.undecided + hits read C s ys)
      < C.harvestLevel}

open scoped Classical in
/-- What remains of a stretch's chance to fire the rate test at a rate below `y`. -/
noncomputable def fireMass (s : Level2State α) : ℝ≥0∞ :=
  if hitRate read C D s < C.harvestRate then fireSum read C D s else 0

open scoped Classical in
/-- The chance that a stretch's count at its last look leaves the rate test silent. -/
noncomputable def missSum (s : Level2State α) : ℝ≥0∞ :=
  (Measure.pi fun _ : Fin (C.probesToEnd - s.probes) => D)
    {ys | ¬ binomSfGe C.probesToEnd C.harvestRate (s.undecided + hits read C s ys)
      < C.harvestLevel}

open scoped Classical in
/-- What remains of a stretch's chance to end with the rate test silent at a rate of at least
`r`. -/
noncomputable def missMass (r : ℝ) (s : Level2State α) : ℝ≥0∞ :=
  if r ≤ hitRate read C D s then missSum read C D s else 0

open scoped Classical in
/-- A harvest at a hypothesis stopping at an undecided read on fewer than `y` of the probes. -/
theorem falseHarvest_le (hI : IdealReads read M side) (hn : 1 ≤ C.probesToEnd)
    (hy1 : C.harvestRate ≤ 1) (ha : 0 ≤ C.harvestLevel) (N : ℕ) :
    (Measure.pi fun _ : Fin N => D) {xs | AnyStep read C
        (fun s x => applyProbe read C s x = .inr (.harvest ())
          ∧ hitRate read C D s < C.harvestRate) initialState (List.ofFn xs)}
      ≤ ENNReal.ofReal (stretches (Fintype.card σ) (Fintype.card α) * C.probesToEnd
          * C.harvestLevel) := by
  set nσ := Fintype.card σ
  set y := C.harvestRate
  set a := C.harvestLevel
  set n0 := C.probesToEnd
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
      · obtain ⟨T, E, he, -, hΨ⟩ := step_change read C hI hn hs hc
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
          rcases step_same read C hc with ⟨he, -, -⟩ | ⟨he, hfire⟩ | ⟨he, -, -⟩ <;> rw [he]
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
          rcases step_same read C hc with ⟨he, -, -⟩ | ⟨he, -⟩ | ⟨he, -, -⟩ <;> rw [he]
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
              (s.undecided + m) < C.harvestLevel)]
            refine lintegral_congr fun x => ?_
            congr 1
            ext ys
            simp only [Set.mem_ofPred_eq]
            constructor <;> intro hh <;> convert hh using 2 <;> ring
          · simp
  refine (anyStep_le read C D _ (Inv read C) F (fun s x s' hs h => (step_inv read C hI hn hs h).1)
    hF N initialState (inv_start read C hn)).trans ?_
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
/-- A consistent ending at a hypothesis stopping at an undecided read on at least `r` of the
probes. -/
theorem missedHarvest_le (hI : IdealReads read M side) (hn : 1 ≤ C.probesToEnd)
    (hy0 : 0 ≤ C.harvestRate) (hy1 : C.harvestRate ≤ 1) (ha : 0 < C.harvestLevel) {r : ℝ}
    (hr0 : 0 ≤ r) (hr1 : r ≤ 1) (N : ℕ) :
    (Measure.pi fun _ : Fin N => D) {xs | AnyStep read C
        (fun s x => applyProbe read C s x = .inr .consistent ∧ r ≤ hitRate read C D s)
        initialState (List.ofFn xs)}
      ≤ ENNReal.ofReal (stretches (Fintype.card σ) (Fintype.card α) * (1 - binomSfGe C.probesToEnd
          r (fireCount C.harvestRate C.harvestLevel C.probesToEnd))) := by
  set nσ := Fintype.card σ
  set K := fireCount C.harvestRate C.harvestLevel C.probesToEnd
  set b := 1 - binomSfGe C.probesToEnd r K
  have hK : binomSfGe C.probesToEnd C.harvestRate K < C.harvestLevel := by
    have hne : {k | binomSfGe C.probesToEnd C.harvestRate k < C.harvestLevel}.Nonempty :=
      ⟨C.probesToEnd + 1, by
        simp only [Set.mem_ofPred_eq]
        unfold binomSfGe
        rw [Finset.Icc_eq_empty (by omega)]
        simpa using ha⟩
    exact Nat.sInf_mem hne
  have hb : 0 ≤ b := by
    have := binomSfGe_antitone' (n := C.probesToEnd) hr0 hr1 (Nat.zero_le K)
    rw [binomSfGe_zero_right] at this
    simp only [b]; linarith
  have hfresh : ∀ T E, missMass read C D r (newHypothesis T E) ≤ ENNReal.ofReal b := by
    intro T E
    unfold missMass missSum
    split_ifs with hp
    · set H := (newHypothesis T E : Level2State α)
      have hsub : {ys : Fin (C.probesToEnd - H.probes) → FreeMonoid α | ¬ binomSfGe
          C.probesToEnd C.harvestRate (H.undecided + hits read C H ys) < C.harvestLevel}
          ⊆ {ys | K ≤ hits read C H ys}ᶜ := by
        intro ys hys hK'
        refine hys (lt_of_le_of_lt (binomSfGe_antitone' hy0 hy1 ?_) hK)
        simp only [Set.mem_ofPred_eq] at hK'
        simpa [H, newHypothesis] using hK'
      have hc : (Measure.pi fun _ : Fin (C.probesToEnd - H.probes) => D).real
          {ys | K ≤ hits read C H ys}ᶜ = 1 - binomSfGe C.probesToEnd (hitRate read C D H) K := by
        rw [measureReal_compl (Set.to_countable _).measurableSet, probReal_univ,
          pi_hits_ge read C D H _ K]
        simp [H, newHypothesis]
      have hmono : binomSfGe C.probesToEnd r K ≤ binomSfGe C.probesToEnd (hitRate read C D H) K :=
        binomSfGe_mono hr0 measureReal_le_one hp _ _
      refine (ENNReal.le_ofReal_iff_toReal_le (measure_ne_top _ _) hb).2 ?_
      rw [← measureReal_def]
      refine (measureReal_mono hsub).trans ?_
      rw [hc]
      change 1 - binomSfGe C.probesToEnd (hitRate read C D H) K ≤ 1 - binomSfGe C.probesToEnd r K
      linarith
    · exact bot_le
  let F : Level2State α → ℝ≥0∞ := fun s => ENNReal.ofReal (b * Ψ nσ s) + missMass read C D r s
  have hF : ∀ s, Inv read C s → ∫⁻ x, ((if (applyProbe read C s x = .inr .consistent
      ∧ r ≤ hitRate read C D s) then 1 else 0) + Sum.elim F (fun _ => 0) (applyProbe read C s x))
      ∂D ≤ F s := by
    intro s hs
    have hlt := hs.probes_lt
    let h : FreeMonoid α → ℝ≥0∞ := fun x =>
      (Measure.pi fun _ : Fin (C.probesToEnd - s.probes - 1) => D)
        {ys | ¬ binomSfGe C.probesToEnd C.harvestRate (s.undecided
          + (stuckString read ⟨s.tree, s.edges⟩ C.walkStart x).length + hits read C s ys)
            < C.harvestLevel}
    let g : FreeMonoid α → ℝ≥0∞ := fun x => if r ≤ hitRate read C D s then h x else 0
    have hpt : ∀ x, ((if (applyProbe read C s x = .inr .consistent ∧ r ≤ hitRate read C D s)
        then 1 else 0) + Sum.elim F (fun _ => 0) (applyProbe read C s x))
        ≤ ENNReal.ofReal (b * Ψ nσ s) + g x := by
      intro x
      by_cases hc : Changes (probeOrAgree false read s.tree s.edges C.walkStart x)
      · obtain ⟨T, E, he, -, hΨ⟩ := step_change read C hI hn hs hc
        rw [he]
        simp only [reduceCtorEq, false_and, if_false, Sum.elim_inl, zero_add, F]
        refine le_trans ?_ le_self_add
        have h1 : ((Ψ nσ (newHypothesis T E) : ℕ) : ℝ) + 1 ≤ Ψ nσ s := by exact_mod_cast hΨ
        calc ENNReal.ofReal (b * Ψ nσ (newHypothesis T E)) + missMass read C D r (newHypothesis T E)
            ≤ ENNReal.ofReal (b * Ψ nσ (newHypothesis T E)) + ENNReal.ofReal b := by
              gcongr; exact hfresh T E
          _ ≤ ENNReal.ofReal (b * Ψ nσ s) := by
              rw [← ENNReal.ofReal_add (by positivity) hb]
              refine ENNReal.ofReal_le_ofReal ?_
              nlinarith
      · by_cases hp : r ≤ hitRate read C D s
        · have hp' : r ≤ hitRate read C D
              (counted s (probeOrAgree false read s.tree s.edges C.walkStart x)) := hp
          simp only [g, if_pos hp]
          rcases step_same read C hc with ⟨he, -, -⟩ | ⟨he, -⟩ | ⟨he, hN, hnf⟩ <;> rw [he]
          · simp only [reduceCtorEq, false_and, if_false, Sum.elim_inl, zero_add, F]
            rw [show missMass read C D r (counted s (probeOrAgree false read s.tree s.edges
              C.walkStart x)) = missSum read C D (counted s (probeOrAgree false read s.tree s.edges
              C.walkStart x)) from if_pos hp']
            refine add_le_add (le_of_eq rfl) (le_of_eq ?_)
            show _ = (Measure.pi fun _ : Fin (C.probesToEnd - s.probes - 1) => D) _
            rw [Nat.sub_sub]
            rfl
          · simp
          · simp only [hp, and_self, if_true, Sum.elim_inr, add_zero]
            refine le_trans (le_of_eq ?_) le_add_self
            have h0 : C.probesToEnd - s.probes - 1 = 0 := by simp [counted] at hN; omega
            have hp1 : (counted s (probeOrAgree false read s.tree s.edges C.walkStart x)).probes
                = C.probesToEnd := by simp [counted] at hN ⊢; omega
            show _ = (Measure.pi fun _ : Fin (C.probesToEnd - s.probes - 1) => D) _
            rw [h0, eq_comm]
            refine pi_hits_zero read C D s (fun m => ¬ binomSfGe C.probesToEnd C.harvestRate
              (s.undecided + (stuckString read ⟨s.tree, s.edges⟩ C.walkStart x).length + m)
                < C.harvestLevel) ?_
            have := hnf
            unfold Fires at this
            rw [hp1] at this
            simpa [counted, stuckString] using this
        · simp only [g, if_neg hp]
          rcases step_same read C hc with ⟨he, -, -⟩ | ⟨he, -⟩ | ⟨he, -, -⟩ <;> rw [he]
          · have hp' : ¬ r ≤ hitRate read C D
                (counted s (probeOrAgree false read s.tree s.edges C.walkStart x)) := hp
            simp only [reduceCtorEq, false_and, if_false, Sum.elim_inl, zero_add, F]
            rw [show missMass read C D r (counted s (probeOrAgree false read s.tree s.edges
              C.walkStart x)) = 0 from if_neg hp', add_zero, add_zero]
            exact le_rfl
          · simp
          · simp [hp]
    calc _ ≤ ∫⁻ x, ENNReal.ofReal (b * Ψ nσ s) + g x ∂D := lintegral_mono hpt
      _ = ENNReal.ofReal (b * Ψ nσ s) + ∫⁻ x, g x ∂D := by
          rw [lintegral_add_left measurable_const, lintegral_const, measure_univ, mul_one]
      _ ≤ F s := by
          simp only [F]
          gcongr
          simp only [g, missMass, missSum, h]
          split_ifs with hp
          · rw [show C.probesToEnd - s.probes = C.probesToEnd - s.probes - 1 + 1 by omega,
              pi_hits_succ read C D s _
              (fun m => ¬ binomSfGe C.probesToEnd C.harvestRate (s.undecided + m) < C.harvestLevel)]
            refine le_of_eq (lintegral_congr fun x => ?_)
            simp only [add_assoc]
            rfl
          · simp
  refine (anyStep_le read C D _ (Inv read C) F (fun s x s' hs h => (step_inv read C hI hn hs h).1)
    hF N initialState (inv_start read C hn)).trans ?_
  have hΨ : (Ψ nσ (initialState : Level2State α) : ℝ) + 1 ≤ stretches nσ (Fintype.card α) := by
    exact_mod_cast Ψ_start nσ (α := α)
  calc F initialState ≤ ENNReal.ofReal (b * Ψ nσ (initialState : Level2State α))
        + ENNReal.ofReal b := add_le_add le_rfl (hfresh _ _)
    _ ≤ _ := by
        rw [← ENNReal.ofReal_add (by positivity) hb]
        refine ENNReal.ofReal_le_ofReal ?_
        nlinarith

end Run

/-! ## The theorem -/

section Theorem

variable [Fintype α] (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] {σ : Type*} [Fintype σ]
  {M : DFA α σ} {side : σ → Bool}

/-- A run that does not end well has a bad step: a consistent ending at a hypothesis that changes
on more than `ε` of the probes or stops at an undecided read on at least `r` of them, or a harvest
at one stopping at an undecided read on fewer than `y` of them. -/
theorem bad_any (hI : IdealReads read M side) (hn : 1 ≤ C.probesToEnd) (hy : 0 < C.harvestRate)
    (ε r : ℝ) : ∀ (xs : List (FreeMonoid α)) (s : Level2State α), Inv read C s →
      Θ C (Fintype.card σ) s < xs.length →
      ¬ EndsWell M (AllUndecided read M) D (ε + r)
          {x | Disagrees read ⟨(runProbes read C s xs).1.tree, (runProbes read C s xs).1.edges⟩
            C.walkStart x} C.walkStart (fun _ => harvestLaw read D C.walkStart
              ⟨(runProbes read C s xs).1.tree, (runProbes read C s xs).1.edges⟩)
          C.harvestRate (runProbes read C s xs).2 →
      AnyStep read C (fun s x => applyProbe read C s x = .inr .consistent ∧ Bad read C D ε s)
          s xs
        ∨ AnyStep read C (fun s x => applyProbe read C s x = .inr (.harvest ())
          ∧ hitRate read C D s < C.harvestRate) s xs
        ∨ AnyStep read C (fun s x => applyProbe read C s x = .inr .consistent
          ∧ r ≤ hitRate read C D s) s xs
  | [], s, _, h, _ => by simp at h
  | x :: xs, s, hs, hΘ, hbad => by
    rcases hst : applyProbe read C s x with s' | e
    · obtain ⟨hs', hlt⟩ := step_inv read C hI hn hs hst
      simp only [runProbes, hst] at hbad
      rcases bad_any hI hn hy ε r xs s' hs' (by simp only [List.length_cons] at hΘ; omega) hbad
        with h | h | h
      · exact Or.inl (Or.inr ⟨s', hst, h⟩)
      · exact Or.inr (Or.inl (Or.inr ⟨s', hst, h⟩))
      · exact Or.inr (Or.inr (Or.inr ⟨s', hst, h⟩))
    · simp only [runProbes, hst] at hbad
      rcases e with _ | ⟨⟩ | _
      · by_contra hno
        simp only [AnyStep, hst, true_and, not_or] at hno
        apply hbad
        show D.real _ ≤ _
        have hB : ¬ Bad read C D ε s := hno.1.1
        have hr : hitRate read C D s < r := not_le.1 hno.2.2.1
        unfold Bad at hB
        have := disagrees_le read C D hs
        linarith
      · by_contra hno
        simp only [AnyStep, hst, true_and, not_or] at hno
        have hle : C.harvestRate ≤ hitRate read C D s := not_lt.1 hno.2.1.1
        apply hbad
        refine endsWell_harvest read D ⟨s.tree, s.edges⟩ C.walkStart hI hs.edges _ _ hy
          (hle.trans ?_)
        have hfin : harvestLaw read D C.walkStart ⟨s.tree, s.edges⟩ Set.univ ≠ ⊤ :=
          ne_top_of_le_ne_top (by simp) (harvestLaw_univ_le read D ⟨s.tree, s.edges⟩ C.walkStart)
        exact ENNReal.toReal_mono hfin (stuck_le_harvestLaw read D ⟨s.tree, s.edges⟩ C.walkStart)
      · exact absurd hst (applyProbe_ne_failed read C s x)

end Theorem

theorem idealRoundCorrect_holds : IdealRoundCorrect := by
  intro α _ _ σ _ M read side C P hI hn hy hy1 ha hP D _ ε r hε1 hr0 hr1
  have hΘ : Θ C (Fintype.card σ) (initialState : Level2State α) < P :=
    (Θ_start C _ hn).trans_le hP
  have hsub : {draws : Fin P → FreeMonoid α | let result := indecisiveLStarLevel2 C read draws
        ¬ EndsWell M (AllUndecided read M) D (ε + r)
          {x | Disagrees read result.hypothesis C.walkStart x}
          C.walkStart (fun _ => harvestLaw read D C.walkStart result.hypothesis) C.harvestRate
          result.ending}
      ⊆ {xs | AnyStep read C (fun s x => applyProbe read C s x = .inr .consistent
            ∧ Bad read C D ε s) initialState (List.ofFn xs)}
        ∪ {xs | AnyStep read C (fun s x => applyProbe read C s x = .inr (.harvest ())
            ∧ hitRate read C D s < C.harvestRate) initialState (List.ofFn xs)}
        ∪ {xs | AnyStep read C (fun s x => applyProbe read C s x = .inr .consistent
            ∧ r ≤ hitRate read C D s) initialState (List.ofFn xs)} := by
    intro xs hxs
    rcases bad_any read C D hI hn hy ε r (List.ofFn xs) initialState (inv_start read C hn)
      (by simpa using hΘ) hxs with h | h | h
    · exact Or.inl (Or.inl h)
    · exact Or.inl (Or.inr h)
    · exact Or.inr h
  have h1 := falseConsistent_le read C D hI hn hε1 P (σ := σ)
  have h2 := falseHarvest_le read C D hI hn hy1 ha.le P (σ := σ)
  have h3 := missedHarvest_le read C D hI hn hy.le hy1 ha hr0 hr1 P (σ := σ)
  set S : ℝ := (stretches (Fintype.card σ) (Fintype.card α) : ℝ)
  have hS : 0 ≤ S := Nat.cast_nonneg _
  have hSP : S * C.probesToEnd * C.harvestLevel ≤ P * C.harvestLevel :=
    mul_le_mul_of_nonneg_right (by simp only [S]; exact_mod_cast hP) ha.le
  have hq : 0 ≤ (1 - ε) ^ C.probesToEnd := pow_nonneg (by linarith) _
  have hb : 0 ≤ 1 - binomSfGe C.probesToEnd r (fireCount C.harvestRate C.harvestLevel
      C.probesToEnd) := by
    have := binomSfGe_antitone' (n := C.probesToEnd) hr0 hr1
      (Nat.zero_le (fireCount C.harvestRate C.harvestLevel C.probesToEnd))
    rw [binomSfGe_zero_right] at this
    linarith
  refine (measureReal_mono hsub).trans (ENNReal.toReal_le_of_le_ofReal (by positivity)
    (((measure_union_le _ _).trans (add_le_add (measure_union_le _ _) le_rfl)).trans ?_))
  calc _ ≤ ENNReal.ofReal (S * (1 - ε) ^ C.probesToEnd)
        + ENNReal.ofReal (S * C.probesToEnd * C.harvestLevel)
        + ENNReal.ofReal (S * (1 - binomSfGe C.probesToEnd r
          (fireCount C.harvestRate C.harvestLevel C.probesToEnd))) :=
        add_le_add (add_le_add h1 h2) h3
    _ ≤ _ := by
        rw [← ENNReal.ofReal_add (by positivity) (by positivity),
          ← ENNReal.ofReal_add (by positivity) (by positivity)]
        refine ENNReal.ofReal_le_ofReal ?_
        nlinarith

end Ideal

end OrthoDFA
