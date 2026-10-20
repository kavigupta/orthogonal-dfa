import OrthoDFA.Level3Sound
import OrthoDFA.Proofs.RandomStretch

/-!
# A consistent ending certifies its rate: the proof

A step is bad where it ends the round consistent at a hypothesis disagreeing on more than `ε`.
Within a stretch the hypothesis is fixed and every probe counted is no change, so the success test
counts the disagreements among fresh draws (`seg_false_consistent`), and fires falsely with chance
at most its level (`seqRateSound`). A change opens a fresh stretch at the next index, so the chance
of a bad step from a stretch at index `m` is at most the levels from `m` on, `δ/(8(m + 1))`
(`anyB_le_tail`).
-/

namespace OrthoDFA

namespace Random

open MeasureTheory

section Compose

variable {S X E : Type*} [MeasurableSpace X] [Countable X] [MeasurableSingletonClass X]
  (step : S → X → S ⊕ E) (B : S → X → Prop) (same : S → S → Prop)
  (D : Measure X) [IsProbabilityMeasure D] (F : S → Prop)

/-- The chance of a bad step is at most the current segment's, plus `T` past its index, where a
fresh segment at index `m` risks `b m` and `b m + T (m + 1) ≤ T m`. -/
theorem anyB_le_tail (idx : S → ℕ) (b T : ℕ → ENNReal) (htel : ∀ m, b m + T (m + 1) ≤ T m)
    (hsame : ∀ s x s', step s x = .inl s' → same s s' → idx s' = idx s)
    (hchg : ∀ s x s', step s x = .inl s' → ¬ same s s' → F s' ∧ idx s' = idx s + 1)
    (hseg : ∀ s, F s → ∀ N, (Measure.pi fun _ : Fin N => D)
      {xs | SegB step B same s (List.ofFn xs)} ≤ b (idx s)) :
    ∀ N s, (Measure.pi fun _ : Fin N => D) {xs | AnyB step B s (List.ofFn xs)}
      ≤ (Measure.pi fun _ : Fin N => D) {xs | SegB step B same s (List.ofFn xs)}
        + T (idx s + 1) := by
  intro N
  induction N with
  | zero =>
    intro s
    have : {xs : Fin 0 → X | AnyB step B s (List.ofFn xs)} = ∅ := by ext xs; simp [AnyB]
    rw [this, measure_empty]
    exact bot_le
  | succ N ih =>
    intro s
    rw [pi_cons_apply, pi_cons_apply]
    calc ∫⁻ x, (Measure.pi fun _ : Fin N => D)
          {xs | Fin.cons x xs ∈ {xs : Fin (N + 1) → X | AnyB step B s (List.ofFn xs)}} ∂D
        ≤ ∫⁻ x, ((Measure.pi fun _ : Fin N => D)
          {xs | Fin.cons x xs ∈ {xs : Fin (N + 1) → X | SegB step B same s (List.ofFn xs)}}
            + T (idx s + 1)) ∂D :=
          lintegral_mono fun x => ?_
      _ = _ := by
          rw [lintegral_add_right _ measurable_const, lintegral_const, measure_univ, mul_one]
    simp only [Set.mem_ofPred_eq, List.ofFn_succ, Fin.cons_zero, Fin.cons_succ, AnyB, SegB]
    by_cases hB : B s x
    · simp only [hB, true_or, Set.ofPred_true, measure_univ]
      exact le_add_right le_rfl
    · simp only [hB, false_or]
      rcases hst : step s x with s' | e
      · simp only [Sum.inl.injEq, exists_eq_left']
        by_cases hsm : same s s'
        · have hi := hsame s x s' hst hsm
          simp only [hsm, true_and]
          rw [← hi]
          exact ih s'
        · obtain ⟨hF, hi⟩ := hchg s x s' hst hsm
          simp only [hsm, false_and, Set.ofPred_false, measure_empty, zero_add]
          calc _ ≤ _ := ih s'
            _ ≤ b (idx s' ) + T (idx s' + 1) := add_le_add (hseg s' hF N) le_rfl
            _ ≤ T (idx s') := htel _
            _ = T (idx s + 1) := by rw [hi]
      · simp only [reduceCtorEq, false_and, exists_false, Set.ofPred_false, measure_empty]
        exact bot_le

end Compose

variable {α : Type*} [Fintype α] [DecidableEq α] (read : FreeMonoid α → ARU) (C : Level3Settings)
  (D : Measure (FreeMonoid α))

/-- The step ends the round consistent at a hypothesis disagreeing on more than the rate. -/
def FalseConsistent (s : Level3State α) (x : FreeMonoid α) : Prop :=
  applyProbe read C s x = .inr .consistent
    ∧ C.disagreementRate < D.real (searchSet read C.walkStart ⟨s.tree, s.edges⟩
      ∪ endsSet read C.walkStart ⟨s.tree, s.edges⟩)

def SameHyp (s s' : Level3State α) : Prop := s'.hypotheses = s.hypotheses

omit [Fintype α] in
theorem applyProbe_inl {s s' : Level3State α} {x : FreeMonoid α}
    (h : applyProbe read C s x = .inl s') :
    (s' = countProbe read C s x
      ∧ (∀ p c u t, probeOrAgree read s.tree s.edges C.walkStart x ≠ .unlearnedEdge p c u t))
    ∨ s' = newHypothesis s'.tree s'.edges (s.hypotheses + 1) := by
  unfold applyProbe at h
  split at h
  · simp only [Sum.inl.injEq] at h
    exact Or.inr (by rw [← h]; rfl)
  · rename_i p c u t hw
    split at h
    · simp only [Sum.inl.injEq] at h
      exact Or.inr (by rw [← h]; rfl)
    · unfold applyTests at h
      split at h
      · simp at h
      · simp only [Sum.inl.injEq] at h
        exact Or.inl ⟨h.symm, fun p' c' u' t' h' => by rw [hw] at h'; cases h'⟩
  · rename_i hn1 hn2
    unfold applyTests at h
    split at h
    · simp at h
    · simp only [Sum.inl.injEq] at h
      refine Or.inl ⟨h.symm, fun p c u t h' => ?_⟩
      exact hn1 p c u t h'

theorem applyProbe_inr {s : Level3State α} {x : FreeMonoid α} {e : RoundEnd Level3Site}
    (h : applyProbe read C s x = .inr e) :
    runTests C (countProbe read C s x) = some e
      ∧ ∀ p c u t, probeOrAgree read s.tree s.edges C.walkStart x ≠ .unlearnedEdge p c u t := by
  refine ⟨step_inr read C h, fun p c u t hu => ?_⟩
  unfold applyProbe at h
  rw [hu] at h
  simp at h

omit [Fintype α] [DecidableEq α] in
/-- Off an unlearned edge, a probe disagrees exactly where success counts it. -/
theorem fSE_or_fW {T : DTree α} {E : Edges α} {x : FreeMonoid α}
    (h : ∀ p c u t, probeOrAgree read T E C.walkStart x ≠ .unlearnedEdge p c u t) :
    (fSE read C T E x || fW read C T E x) = fSE read C T E x := by
  simp only [fSE, fS, fW]
  cases hg : probeOrAgree read T E C.walkStart x with
  | unlearnedEdge p c u t => exact absurd hg (h p c u t)
  | _ => simp [searches, changes]

/-- A false success within a stretch shows in its counts. -/
theorem seg_false_consistent (T : DTree α) (E : Edges α) (j : ℕ) :
    ∀ (xs ys : List (FreeMonoid α)) (s : Level3State α),
      StInv read C T E j ys s →
      (∀ y ∈ ys, (fSE read C T E y || fW read C T E y) = fSE read C T E y) →
      SegB (applyProbe read C) (FalseConsistent read C D) SameHyp s xs →
      C.disagreementRate < D.real {x | fSE read C T E x || fW read C T E x}
        ∧ ∃ n, 1 ≤ n ∧ n ≤ xs.length ∧ seqRate C.disagreementRate C.disagreementAlternative
          (C.level j) (ys.length + n)
          ((ys ++ xs.take n).countP fun x => fSE read C T E x || fW read C T E x)
  | [], _, _, _, _, h => by simp [SegB] at h
  | x :: xs, ys, s, hst, hys, h => by
    simp only [SegB] at h
    rcases h with ⟨hinr, hrate⟩ | ⟨s', hs', hsm, hseg⟩
    · obtain ⟨htests, hnu⟩ := applyProbe_inr read C hinr
      have hst' := stInv_countProbe read C T E j hst x
      rcases look_some C htests with ⟨-, h⟩ | ⟨-, -, h⟩ | ⟨hfire, -⟩
      · cases h
      · cases h
      refine ⟨?_, 1, le_rfl, by simp, ?_⟩
      · rw [← disagreeSet_eq, ← hst.tree, ← hst.edges]
        exact hrate
      · have hc : (ys ++ [x]).countP (fun x => fSE read C T E x || fW read C T E x)
            = (ys ++ [x]).countP (fSE read C T E) := by
          refine List.countP_congr fun y hy => ?_
          rcases List.mem_append.1 hy with hy | hy
          · simp [hys y hy]
          · rw [List.mem_singleton.1 hy, ← hst.tree, ← hst.edges] at *
            simp [fSE_or_fW read C hnu]
        simp only [List.take_succ_cons, List.take_zero]
        rw [hst'.hypotheses, hst'.probeCount, hst'.searchCount, hst'.endsUndecided,
          ← countP_fSE, ← hc] at hfire
        simpa using hfire
    · rcases applyProbe_inl read C hs' with ⟨rfl, hnu⟩ | hnew
      · have hst' := stInv_countProbe read C T E j hst x
        have hys' : ∀ y ∈ ys ++ [x], (fSE read C T E y || fW read C T E y) = fSE read C T E y := by
          intro y hy
          rcases List.mem_append.1 hy with hy | hy
          · exact hys y hy
          · rw [List.mem_singleton.1 hy]
            rw [← hst.tree, ← hst.edges]
            exact fSE_or_fW read C hnu
        obtain ⟨hrate, n, hn1, hnl, hfire⟩ :=
          seg_false_consistent T E j xs (ys ++ [x]) _ hst' hys' hseg
        refine ⟨hrate, n + 1, by omega, by simp; omega, ?_⟩
        simp only [List.length_append, List.length_singleton, List.take_succ_cons,
          List.append_assoc, List.singleton_append] at hfire ⊢
        rwa [show ys.length + (n + 1) = ys.length + 1 + n by omega]
      · exfalso
        have := congrArg Level3State.hypotheses hnew
        simp only [SameHyp] at hsm
        simp only [newHypothesis] at this
        omega

theorem seg_le_level [IsProbabilityMeasure D] (hδ : 0 < C.δ) (hε0 : 0 < C.disagreementRate)
    (hε1 : C.disagreementRate < 1) (T : DTree α) (E : Edges α) (j N : ℕ) :
    (Measure.pi fun _ : Fin N => D)
        {xs | SegB (applyProbe read C) (FalseConsistent read C D) SameHyp (newHypothesis T E j)
          (List.ofFn xs)}
      ≤ ENNReal.ofReal (C.level j) := by
  classical
  by_cases hr : C.disagreementRate < D.real {x | fSE read C T E x || fW read C T E x}
  · have hsub : {xs : Fin N → FreeMonoid α | SegB (applyProbe read C) (FalseConsistent read C D)
          SameHyp (newHypothesis T E j) (List.ofFn xs)}
        ⊆ {xs | ∃ n ≤ N, seqRate C.disagreementRate C.disagreementAlternative (C.level j) n
          (((List.ofFn xs).take n).countP fun x =>
            @decide ((fSE read C T E x || fW read C T E x) = true)
              (Classical.propDecidable _))} := by
      intro xs hxs
      obtain ⟨-, n, -, hn, hfire⟩ := seg_false_consistent read C D T E j (List.ofFn xs) []
        (newHypothesis T E j) (stInv_fresh read C T E j) (by simp) hxs
      simp only [List.length_ofFn, List.length_nil, zero_add, List.nil_append] at hn hfire
      exact ⟨n, hn, by simpa using hfire⟩
    have hs := seqRateSound D (fun x => (fSE read C T E x || fW read C T E x) = true)
      C.disagreementRate C.disagreementAlternative (C.level j) N hε0 hε1
      (by unfold Level3Settings.disagreementAlternative; linarith)
      (by unfold Level3Settings.disagreementAlternative; linarith)
      (by unfold Level3Settings.level; positivity)
      (by
        unfold Level3Settings.disagreementAlternative
        have : 0 ≤ D.real {x | (fSE read C T E x || fW read C T E x) = true}
            - C.disagreementRate := by linarith
        nlinarith)
    rw [← ENNReal.ofReal_toReal (measure_ne_top _ _), ← measureReal_def]
    exact ENNReal.ofReal_le_ofReal ((measureReal_mono hsub).trans hs)
  · have : {xs : Fin N → FreeMonoid α | SegB (applyProbe read C) (FalseConsistent read C D)
          SameHyp (newHypothesis T E j) (List.ofFn xs)} = ∅ := by
      ext xs
      simp only [Set.mem_ofPred_eq, Set.mem_empty_iff_false, iff_false]
      intro hxs
      exact hr (seg_false_consistent read C D T E j (List.ofFn xs) []
        (newHypothesis T E j) (stInv_fresh read C T E j) (by simp) hxs).1
    rw [this, measure_empty]
    exact bot_le

/-- A run that ends consistent at a hypothesis disagreeing on more than the rate has a bad
step. -/
theorem anyB_of_consistent :
    ∀ (xs : List (FreeMonoid α)) (s : Level3State α),
      (runProbes read C s xs).2 = .consistent →
      C.disagreementRate < D.real (searchSet read C.walkStart
          ⟨(runProbes read C s xs).1.tree, (runProbes read C s xs).1.edges⟩
        ∪ endsSet read C.walkStart
          ⟨(runProbes read C s xs).1.tree, (runProbes read C s xs).1.edges⟩) →
      AnyB (applyProbe read C) (FalseConsistent read C D) s xs
  | [], s, h, _ => by simp [runProbes] at h
  | x :: xs, s, h, hr => by
    simp only [AnyB]
    rcases hst : applyProbe read C s x with s' | e
    · simp only [runProbes, hst] at h hr
      exact Or.inr ⟨s', rfl, anyB_of_consistent xs s' h hr⟩
    · simp only [runProbes, hst] at h hr
      subst h
      exact Or.inl ⟨hst, hr⟩

end Random

open MeasureTheory Random

theorem level3ConsistentSound_holds : Level3ConsistentSound := by
  intro α _ _ read D _ C hδ hε0 hε1
  classical
  set T : ℕ → ENNReal := fun m => ENNReal.ofReal (C.δ / (8 * (m + 1))) with hT
  have hlev : ∀ m : ℕ, C.level m = C.δ / (8 * (m + 1) * (m + 2)) := fun _ => rfl
  have htel : ∀ m : ℕ, ENNReal.ofReal (C.level m) + T (m + 1) ≤ T m := by
    intro m
    simp only [hT, hlev]
    rw [← ENNReal.ofReal_add (by positivity) (by positivity)]
    apply ENNReal.ofReal_le_ofReal
    push_cast
    rw [div_add_div _ _ (by positivity) (by positivity), div_le_div_iff₀ (by positivity)
      (by positivity)]
    ring_nf
    nlinarith [sq_nonneg (m : ℝ), hδ]
  have hmain := anyB_le_tail (applyProbe read C) (FalseConsistent read C D) SameHyp D
    (fun s => s = newHypothesis s.tree s.edges s.hypotheses) Level3State.hypotheses
    (fun m => ENNReal.ofReal (C.level m)) T htel
    (fun s x s' hs hsm => hsm)
    (fun s x s' hs hsm => by
      rcases applyProbe_inl read C hs with ⟨rfl, -⟩ | hnew
      · exact absurd rfl hsm
      · refine ⟨?_, ?_⟩
        · conv_lhs => rw [hnew]
          have := congrArg Level3State.hypotheses hnew
          simp only [newHypothesis] at this
          rw [this]
        · have := congrArg Level3State.hypotheses hnew
          simpa [newHypothesis] using this)
    (fun s hs N => by
      rw [hs]
      exact seg_le_level read C D hδ hε0 hε1 _ _ _ N)
    C.probeBudget initialState
  have hseg0 := seg_le_level read C D hδ hε0 hε1 (.node 1 .leaf .leaf) (fun _ => none) 0
    C.probeBudget
  have hsub : {draws : Fin C.probeBudget → FreeMonoid α |
      (indecisiveLStarLevel3 C read draws).ending = .consistent
        ∧ C.disagreementRate < D.real (searchSet read C.walkStart
            (indecisiveLStarLevel3 C read draws).hypothesis
          ∪ endsSet read C.walkStart (indecisiveLStarLevel3 C read draws).hypothesis)}
      ⊆ {xs | AnyB (applyProbe read C) (FalseConsistent read C D) initialState
          (List.ofFn xs)} := fun draws h =>
    anyB_of_consistent read C D (List.ofFn draws) initialState h.1 h.2
  have hfin : (Measure.pi fun _ : Fin C.probeBudget => D)
      {xs | AnyB (applyProbe read C) (FalseConsistent read C D) initialState (List.ofFn xs)}
      ≤ ENNReal.ofReal (C.δ / 8) := by
    refine hmain.trans ?_
    have h0 : (initialState : Level3State α) = newHypothesis (.node 1 .leaf .leaf)
        (fun _ => none) 0 := rfl
    rw [h0]
    refine (add_le_add hseg0 le_rfl).trans ?_
    refine (htel 0).trans (le_of_eq ?_)
    simp [hT]
  calc _ ≤ (Measure.pi fun _ : Fin C.probeBudget => D).real
        {xs | AnyB (applyProbe read C) (FalseConsistent read C D) initialState (List.ofFn xs)} :=
        measureReal_mono hsub
    _ ≤ C.δ / 8 := by
        rw [measureReal_def]
        exact ENNReal.toReal_le_of_le_ofReal (by positivity) hfin

end OrthoDFA
