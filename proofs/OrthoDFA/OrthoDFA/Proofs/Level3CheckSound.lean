import OrthoDFA.Level3CheckSound
import OrthoDFA.Proofs.Level3Sound

/-!
# A passing check certifies its rate: the proof

A step is bad where its probe fires the success test and its block's check passes a hypothesis
failing more than the rate. Within a stretch the hypothesis is fixed and a firing ends it, so the
stretch's chance of a bad step is at most its check's level, the block being fresh
(`seg_check_le`). A change opens a fresh stretch at the next index, as at level 3
(`anyB_le_tail`).
-/

namespace OrthoDFA

namespace Random

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α] (read : FreeMonoid α → ARU)
  (D Dc : Measure (FreeMonoid α)) (C : Level3Settings) (K : CheckSettings)

/-- The hypothesis fails more than the check's rate from `0`. -/
def CheckBad (s : Level3State α) : Prop :=
  K.rate < Dc.real {x | startFails read s.tree s.edges x = true}

/-- The probe fires the success test and the block's check passes a bad hypothesis. -/
def FalseCheck (s : Level3State α) (d : CheckDraw α K) : Prop :=
  applyProbe read C s d.1 = .inr .consistent
    ∧ checkPasses read K s.hypotheses s.tree s.edges (List.ofFn d.2) ∧ CheckBad read Dc K s

omit [Fintype α] in
theorem fixOf_hyp {s s' : Level3State α} {x : FreeMonoid α} (h : fixOf read s x = some s') :
    s' = newHypothesis s'.tree s'.edges (s.hypotheses + 1) := by
  unfold fixOf at h
  split at h
  · simp only [Option.some.injEq] at h
    rw [← h]; rfl
  · split at h
    · simp only [Option.some.injEq] at h
      rw [← h]; rfl
    · simp at h
  · simp at h

omit [Fintype α] in
theorem firstFix_hyp {s s' : Level3State α} :
    ∀ {xs : List (FreeMonoid α)}, firstFix read s xs = some s' →
      s' = newHypothesis s'.tree s'.edges (s.hypotheses + 1)
  | [], h => by simp [firstFix] at h
  | x :: xs, h => by
    simp only [firstFix] at h
    rcases hf : fixOf read s x with _ | t
    · rw [hf] at h
      exact firstFix_hyp h
    · rw [hf] at h
      simp only [Option.elim, Option.some.injEq] at h
      subst h
      exact fixOf_hyp read hf

omit [Fintype α] in
theorem checkStep_inl {s s' : Level3State α} {b : Fin K.size → FreeMonoid α}
    (h : checkStep read K s b = .inl s') :
    s' = newHypothesis s'.tree s'.edges (s.hypotheses + 1) := by
  classical
  unfold checkStep at h
  split_ifs at h
  rcases hf : firstFix read s (List.ofFn b) with _ | t <;> rw [hf] at h
  · simp at h
  · simp only [Sum.inl.injEq] at h
    subst h
    exact firstFix_hyp read hf

omit [Fintype α] in
theorem checkStep_inr_consistent {s : Level3State α} {b : Fin K.size → FreeMonoid α}
    (h : checkStep read K s b = .inr .consistent) :
    checkPasses read K s.hypotheses s.tree s.edges (List.ofFn b) := by
  classical
  unfold checkStep at h
  split_ifs at h with hp
  · exact hp
  · rcases hf : firstFix read s (List.ofFn b) with _ | t <;> rw [hf] at h <;> simp at h

omit [Fintype α] in
/-- A step that keeps the hypothesis is level 3's counted probe, at a probe that does not fire;
any other keeps nothing but opens the next hypothesis. -/
theorem applyCheck_inl {s s' : Level3State α} {d : CheckDraw α K}
    (h : applyCheck read C K s d = .inl s') :
    (applyProbe read C s d.1 = .inl s' ∧ (s' = countProbe read C s d.1
      ∨ s' = newHypothesis s'.tree s'.edges (s.hypotheses + 1)))
    ∨ (applyProbe read C s d.1 = .inr .consistent
      ∧ s' = newHypothesis s'.tree s'.edges (s.hypotheses + 1)) := by
  unfold applyCheck at h
  rcases ha : applyProbe read C s d.1 with t | e
  · rw [ha] at h
    simp only [Sum.inl.injEq] at h
    subst h
    refine Or.inl ⟨rfl, ?_⟩
    rcases applyProbe_inl read C ha with ⟨h1, -⟩ | h1
    · exact Or.inl h1
    · exact Or.inr h1
  · rw [ha] at h
    rcases e with _ | e | _
    · exact Or.inr ⟨rfl, checkStep_inl read K h⟩
    · simp at h
    · simp at h

omit [Fintype α] in
theorem applyCheck_sameHyp {s s' : Level3State α} {d : CheckDraw α K}
    (h : applyCheck read C K s d = .inl s') (hsm : SameHyp s s') :
    applyProbe read C s d.1 = .inl s' ∧ s'.tree = s.tree ∧ s'.edges = s.edges := by
  have hne : ¬ s' = newHypothesis s'.tree s'.edges (s.hypotheses + 1) := fun hn => by
    have := congrArg Level3State.hypotheses hn
    simp only [newHypothesis, SameHyp] at this hsm
    omega
  rcases applyCheck_inl read C K h with ⟨ha, h1 | h1⟩ | ⟨-, h1⟩
  · exact ⟨ha, by rw [h1]; rfl, by rw [h1]; rfl⟩
  · exact absurd h1 hne
  · exact absurd h1 hne

variable [IsProbabilityMeasure D] [IsProbabilityMeasure Dc]

omit [DecidableEq α] [IsProbabilityMeasure D] in
/-- A check that passes a bad hypothesis is its level's risk. -/
theorem check_pass_le (hr0 : 0 < K.rate) (hr1 : K.rate < 1) (hK : 0 < K.δ) (j : ℕ)
    (s : Level3State α) (hs : CheckBad read Dc K s) :
    (Measure.pi fun _ : Fin K.size => Dc)
        {b | checkPasses read K j s.tree s.edges (List.ofFn b)}
      ≤ ENNReal.ofReal (K.level j) := by
  classical
  have h := seqRateSound Dc (fun x => startFails read s.tree s.edges x = true) K.rate (K.rate / 2)
    (K.level j) K.size hr0 hr1 (by linarith) (by linarith)
    (by unfold CheckSettings.level; positivity)
    (by
      have : 0 ≤ Dc.real {x | startFails read s.tree s.edges x = true} - K.rate := by
        unfold CheckBad at hs; linarith
      nlinarith)
  rw [← ENNReal.ofReal_toReal (measure_ne_top _ _), ← measureReal_def]
  refine ENNReal.ofReal_le_ofReal ((measureReal_mono fun b hb => ?_).trans h)
  obtain ⟨n, hn, hf⟩ := hb
  rw [List.length_ofFn] at hn
  exact ⟨n, hn, by simpa using hf⟩

/-- Within a stretch, a bad step has chance at most the check's level. -/
theorem seg_check_le (hr0 : 0 < K.rate) (hr1 : K.rate < 1) (hK : 0 < K.δ) (N : ℕ) :
    ∀ s : Level3State α,
      (Measure.pi fun _ : Fin N => D.prod (Measure.pi fun _ : Fin K.size => Dc))
          {ds | SegB (applyCheck read C K) (FalseCheck read Dc C K) SameHyp s (List.ofFn ds)}
        ≤ ENNReal.ofReal (K.level s.hypotheses) := by
  classical
  induction N with
  | zero =>
    intro s
    have : {ds : Fin 0 → CheckDraw α K | SegB (applyCheck read C K) (FalseCheck read Dc C K)
        SameHyp s (List.ofFn ds)} = ∅ := by ext ds; simp [SegB]
    rw [this, measure_empty]
    exact bot_le
  | succ N ih =>
    intro s
    rw [pi_cons_apply]
    set lv := ENNReal.ofReal (K.level s.hypotheses)
    set fire : FreeMonoid α → Prop := fun x => applyProbe read C s x = .inr .consistent
    have hpt : ∀ d : CheckDraw α K,
        (Measure.pi fun _ : Fin N => D.prod (Measure.pi fun _ : Fin K.size => Dc))
          {ds | (Fin.cons d ds : Fin (N + 1) → CheckDraw α K) ∈ {ds : Fin (N + 1) → CheckDraw α K |
            SegB (applyCheck read C K) (FalseCheck read Dc C K) SameHyp s (List.ofFn ds)}}
        ≤ Set.indicator {d | FalseCheck read Dc C K s d} 1 d
          + Set.indicator {d | ¬ fire d.1} (fun _ => lv) d := by
      intro d
      simp only [Set.mem_ofPred_eq, List.ofFn_succ, Fin.cons_zero, Fin.cons_succ, SegB]
      by_cases hB : FalseCheck read Dc C K s d
      · simp only [hB, true_or, Set.ofPred_true, measure_univ]
        rw [Set.indicator_of_mem (show d ∈ {d | FalseCheck read Dc C K s d} from hB)]
        exact le_add_right le_rfl
      simp only [hB, false_or]
      rw [Set.indicator_of_notMem (show d ∉ {d | FalseCheck read Dc C K s d} from hB), zero_add]
      rcases hst : applyCheck read C K s d with s' | e
      · simp only [Sum.inl.injEq, exists_eq_left']
        by_cases hsm : SameHyp s s'
        · simp only [hsm, true_and]
          obtain ⟨ha, -, -⟩ := applyCheck_sameHyp read C K hst hsm
          have hnf : ¬ fire d.1 := fun hf => by
            simp only [fire] at hf
            rw [hf] at ha
            cases ha
          rw [Set.indicator_of_mem (show d ∈ {d | ¬ fire d.1} from hnf)]
          have := ih s'
          rwa [show s'.hypotheses = s.hypotheses from hsm] at this
        · simp only [hsm, false_and, Set.ofPred_false, measure_empty]
          exact bot_le
      · simp only [reduceCtorEq, false_and, exists_false, Set.ofPred_false, measure_empty]
        exact bot_le
    calc ∫⁻ d, _ ∂(D.prod (Measure.pi fun _ : Fin K.size => Dc))
        ≤ ∫⁻ d, (Set.indicator {d | FalseCheck read Dc C K s d} 1 d
            + Set.indicator {d | ¬ fire d.1} (fun _ => lv) d)
            ∂(D.prod (Measure.pi fun _ : Fin K.size => Dc)) := lintegral_mono hpt
      _ = (D.prod (Measure.pi fun _ : Fin K.size => Dc)) {d | FalseCheck read Dc C K s d}
            + lv * (D.prod (Measure.pi fun _ : Fin K.size => Dc)) {d | ¬ fire d.1} := by
          rw [lintegral_add_left (measurable_one.indicator (Set.to_countable _).measurableSet),
            lintegral_indicator_one (Set.to_countable _).measurableSet,
            lintegral_indicator_const (Set.to_countable _).measurableSet]
      _ ≤ lv * D {x | fire x} + lv * D {x | ¬ fire x} := by
          gcongr
          · rw [Measure.prod_apply (Set.to_countable _).measurableSet]
            calc ∫⁻ x, (Measure.pi fun _ : Fin K.size => Dc)
                  (Prod.mk x ⁻¹' {d | FalseCheck read Dc C K s d}) ∂D
                ≤ ∫⁻ x, Set.indicator {x | fire x} (fun _ => lv) x ∂D := lintegral_mono fun x => by
                  by_cases hf : fire x
                  · rw [Set.indicator_of_mem (show x ∈ {x | fire x} from hf)]
                    by_cases hb : CheckBad read Dc K s
                    · exact (measure_mono fun b hb' => hb'.2.1).trans
                        (check_pass_le read Dc K hr0 hr1 hK _ s hb)
                    · have : Prod.mk x ⁻¹' {d | FalseCheck read Dc C K s d} = ∅ := by
                        ext b
                        simp only [Set.mem_preimage, Set.mem_ofPred_eq, Set.mem_empty_iff_false,
                          iff_false]
                        exact fun h => hb h.2.2
                      rw [this, measure_empty]
                      exact bot_le
                  · rw [Set.indicator_of_notMem (show x ∉ {x | fire x} from hf)]
                    have : Prod.mk x ⁻¹' {d | FalseCheck read Dc C K s d} = ∅ := by
                      ext b
                      simp only [Set.mem_preimage, Set.mem_ofPred_eq, Set.mem_empty_iff_false,
                        iff_false]
                      exact fun h => hf h.1
                    rw [this, measure_empty]
              _ = lv * D {x | fire x} := by
                  rw [lintegral_indicator_const (Set.to_countable _).measurableSet]
          · have : {d : CheckDraw α K | ¬ fire d.1} = {x | ¬ fire x} ×ˢ Set.univ := by
              ext d; simp
            rw [this, Measure.prod_prod, measure_univ, mul_one]
      _ = lv := by
          rw [← mul_add, show {x | ¬ fire x} = {x | fire x}ᶜ from rfl,
            measure_add_measure_compl (Set.to_countable _).measurableSet, measure_univ, mul_one]

omit [IsProbabilityMeasure D] [IsProbabilityMeasure Dc] in
/-- A run ending consistent at a bad hypothesis has a bad step. -/
theorem anyB_of_check :
    ∀ (ds : List (CheckDraw α K)) (s : Level3State α),
      (runChecked read C K s ds).2 = .consistent → CheckBad read Dc K (runChecked read C K s ds).1 →
      AnyB (applyCheck read C K) (FalseCheck read Dc C K) s ds
  | [], s, h, _ => by simp [runChecked] at h
  | d :: ds, s, h, hb => by
    simp only [AnyB]
    rcases hst : applyCheck read C K s d with s' | e
    · simp only [runChecked, hst] at h hb
      exact Or.inr ⟨s', rfl, anyB_of_check ds s' h hb⟩
    · simp only [runChecked, hst] at h hb
      subst h
      refine Or.inl ?_
      unfold applyCheck at hst
      rcases ha : applyProbe read C s d.1 with t | e
      · rw [ha] at hst; cases hst
      · rw [ha] at hst
        rcases e with _ | e | _
        · exact ⟨ha, checkStep_inr_consistent read K hst, hb⟩
        · cases hst
        · cases hst

end Random

open MeasureTheory Random

theorem level3CheckSound_holds : Level3CheckSound := by
  intro α _ _ read D Dc _ _ C K hK hr0 hr1
  classical
  set T : ℕ → ENNReal := fun m => ENNReal.ofReal (K.δ / (8 * (m + 1))) with hT
  have htel : ∀ m : ℕ, ENNReal.ofReal (K.level m) + T (m + 1) ≤ T m := by
    intro m
    simp only [hT, CheckSettings.level]
    rw [← ENNReal.ofReal_add (by positivity) (by positivity)]
    apply ENNReal.ofReal_le_ofReal
    push_cast
    rw [div_add_div _ _ (by positivity) (by positivity), div_le_div_iff₀ (by positivity)
      (by positivity)]
    ring_nf
    nlinarith [sq_nonneg (m : ℝ), hK]
  have hmain := anyB_le_tail (applyCheck read C K) (FalseCheck read Dc C K) SameHyp
    (D.prod (Measure.pi fun _ : Fin K.size => Dc)) (fun _ => True) Level3State.hypotheses
    (fun m => ENNReal.ofReal (K.level m)) T htel (fun s x s' hs hsm => hsm)
    (fun s x s' hs hsm => by
      refine ⟨trivial, ?_⟩
      rcases applyCheck_inl read C K hs with ⟨-, h1 | h1⟩ | ⟨-, h1⟩
      · exact absurd (by rw [h1]; rfl) hsm
      · have := congrArg Level3State.hypotheses h1
        simpa [newHypothesis] using this
      · have := congrArg Level3State.hypotheses h1
        simpa [newHypothesis] using this)
    (fun s _ N => seg_check_le read D Dc C K hr0 hr1 hK N s)
    C.probeBudget initialState
  have hseg0 := seg_check_le read D Dc C K hr0 hr1 hK C.probeBudget initialState
  have hsub : {draws : Fin C.probeBudget → CheckDraw α K |
      (indecisiveLStarCheck C K read draws).ending = .consistent
        ∧ K.rate < Dc.real {x | startFails read (indecisiveLStarCheck C K read draws).hypothesis.tree
          (indecisiveLStarCheck C K read draws).hypothesis.edges x = true}}
      ⊆ {ds | AnyB (applyCheck read C K) (FalseCheck read Dc C K) initialState
          (List.ofFn ds)} := fun draws h =>
    anyB_of_check read Dc C K (List.ofFn draws) initialState h.1 h.2
  have hfin : (Measure.pi fun _ : Fin C.probeBudget => D.prod (Measure.pi fun _ : Fin K.size => Dc))
      {ds | AnyB (applyCheck read C K) (FalseCheck read Dc C K) initialState (List.ofFn ds)}
      ≤ ENNReal.ofReal (K.δ / 8) := by
    refine hmain.trans ?_
    have h0 : (initialState : Level3State α).hypotheses = 0 := rfl
    rw [h0] at hseg0 ⊢
    refine (add_le_add hseg0 le_rfl).trans ((htel 0).trans (le_of_eq ?_))
    simp [hT]
  calc _ ≤ (Measure.pi fun _ : Fin C.probeBudget => D.prod (Measure.pi fun _ : Fin K.size => Dc)).real
        {ds | AnyB (applyCheck read C K) (FalseCheck read Dc C K) initialState (List.ofFn ds)} :=
        measureReal_mono hsub
    _ ≤ K.δ / 8 := by
        rw [measureReal_def]
        exact ENNReal.toReal_le_of_le_ofReal (by positivity) hfin

end OrthoDFA
