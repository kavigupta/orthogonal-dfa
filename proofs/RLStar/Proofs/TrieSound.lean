import RLStar.Proofs.Trie
import RLStar.Proofs.RandomStretch
import RLStar.Proofs.SeqTest

/-!
# Local walk starts: a consistent ending is sound

Over a stretch the hypothesis and the expanded prefixes are fixed, so the success test counts,
over fresh probes, those in one set, a skipped probe being in it. Its false success at any look
has chance at most the stretch's level (`SeqRateSound`), and the stretches' levels sum to `δ / 8`
(`anyB_le_tail`).
-/

noncomputable section

namespace OrthoDFA

namespace Random

open MeasureTheory Query

section Compose

variable {S X E : Type*} [MeasurableSpace X] [Countable X] [MeasurableSingletonClass X]
  (step : S → X → S ⊕ E) (B : S → X → Prop) (same : S → S → Prop)
  (D : Measure X) [IsProbabilityMeasure D]

/-- `∑_{i > k} b i`. -/
def stretchTail (b : ℕ → ENNReal) (k : ℕ) : ENNReal := ∑' i, if k < i then b i else 0

theorem stretchTail_succ (b : ℕ → ENNReal) (k : ℕ) :
    stretchTail b k = b (k + 1) + stretchTail b (k + 1) := by
  unfold stretchTail
  rw [ENNReal.tsum_eq_add_tsum_ite (k + 1)]
  simp only [show k < k + 1 by omega, if_true]
  congr 1
  refine tsum_congr fun i => ?_
  by_cases h : i = k + 1
  · subst h; simp
  · by_cases h' : k + 1 < i
    · simp [h, h', show k < i by omega]
    · simp [h, h', show ¬ k < i by omega]

theorem tsum_eq_stretchTail (b : ℕ → ENNReal) : ∑' i, b i = b 0 + stretchTail b 0 := by
  unfold stretchTail
  rw [ENNReal.tsum_eq_add_tsum_ite 0]
  congr 1
  refine tsum_congr fun i => ?_
  by_cases h : i = 0
  · subst h; simp
  · simp [h, Nat.pos_of_ne_zero h]

/-- The chance of a bad step is at most the current segment's, plus a fresh one's at each later
index. -/
theorem anyB_le_tail (idx : S → ℕ) (b : ℕ → ENNReal) (F : S → Prop)
    (hsame : ∀ s x s', step s x = .inl s' → same s s' → idx s' = idx s)
    (hchg : ∀ s x s', step s x = .inl s' → ¬ same s s' → F s' ∧ idx s' = idx s + 1)
    (hseg : ∀ s, F s → ∀ N, (Measure.pi fun _ : Fin N => D)
      {xs | SegB step B same s (List.ofFn xs)} ≤ b (idx s)) :
    ∀ N s, (Measure.pi fun _ : Fin N => D) {xs | AnyB step B s (List.ofFn xs)}
      ≤ (Measure.pi fun _ : Fin N => D) {xs | SegB step B same s (List.ofFn xs)}
        + stretchTail b (idx s) := by
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
            + stretchTail b (idx s)) ∂D :=
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
            _ ≤ b (idx s') + stretchTail b (idx s') := add_le_add (hseg s' hF N) le_rfl
            _ = stretchTail b (idx s) := by rw [hi, ← stretchTail_succ]
      · simp only [reduceCtorEq, false_and, exists_false, Set.ofPred_false, measure_empty]
        exact bot_le

end Compose

theorem tsum_level_le (C : Level3Settings) (hδ : 0 ≤ C.δ) :
    ∑' i, ENNReal.ofReal (C.level i) ≤ ENNReal.ofReal (C.δ / 8) := by
  apply ENNReal.tsum_le_of_sum_range_le
  intro n
  rw [← ENNReal.ofReal_sum_of_nonneg (fun i _ => by unfold Level3Settings.level; positivity)]
  apply ENNReal.ofReal_le_ofReal
  have key : ∀ n : ℕ, ∑ i ∈ Finset.range n, C.level i = C.δ / 8 * (n / (n + 1)) := by
    intro n
    induction n with
    | zero => simp
    | succ n ih =>
      rw [Finset.sum_range_succ, ih]
      unfold Level3Settings.level
      push_cast
      field_simp
      ring
  rw [key]
  have h1 : (n : ℝ) / (n + 1) ≤ 1 := by rw [div_le_one (by positivity)]; linarith
  have h0 : 0 ≤ C.δ / 8 := by positivity
  nlinarith

variable {α : Type*} [Fintype α] [DecidableEq α]
  (C : Level3Settings) (cap : ℕ) (read : FreeMonoid α → ARU) (D : Measure (FreeMonoid α))

def trieStep (s : TrieState α) (x : FreeMonoid α) : TrieState α ⊕ RoundEnd TrieSite :=
  (applyTrieProbeQ C cap s x).eval read

def trieDis (s : TrieState α) : Set (FreeMonoid α) :=
  trieDisagreeSet read s.expanded ⟨s.tree, s.edges⟩

/-- The step ends the round consistent while more than `ε` of the probes disagree. -/
def FalseConsistent (s : TrieState α) (x : FreeMonoid α) : Prop :=
  trieStep C cap read s x = .inr .consistent ∧ C.disagreementRate < D.real (trieDis read s)

def SameStretch (s s' : TrieState α) : Prop := s'.stretches = s.stretches

omit [Fintype α] [DecidableEq α] in
theorem successDisagrees_iff (H : HypothesisDFA α) (X : Finset (FreeMonoid α)) (x : FreeMonoid α)
    (ho : ∀ p c u t,
      probeOrAgree read H.tree H.edges (walkStartIn X x) x ≠ .unlearnedEdge p c u t) :
    successDisagrees (probeOrAgree read H.tree H.edges (walkStartIn X x) x) = true
      ↔ x ∈ trieDisagreeSet read X H := by
  simp only [trieDisagreeSet, searchSet, endsSet, Set.mem_setOf_eq, Set.mem_union]
  generalize probeOrAgree read H.tree H.edges (walkStartIn X x) x = o at ho ⊢
  cases o with
  | undecided zs cl => cases cl <;> cases zs <;> simp [successDisagrees, searches, endsString]
  | unlearnedEdge p c u t => exact absurd rfl (ho p c u t)
  | _ => simp [successDisagrees, searches, endsString]

omit [Fintype α] [DecidableEq α] in
theorem mem_of_harvestString (H : HypothesisDFA α) (X : Finset (FreeMonoid α)) (x : FreeMonoid α)
    {z : FreeMonoid α}
    (hz : z ∈ harvestString (probeOrAgree read H.tree H.edges (walkStartIn X x) x)) :
    x ∈ trieDisagreeSet read X H := by
  simp only [trieDisagreeSet, searchSet, endsSet, Set.mem_setOf_eq, Set.mem_union]
  generalize probeOrAgree read H.tree H.edges (walkStartIn X x) x = o at hz ⊢
  cases o with
  | undecided zs cl =>
    cases cl <;> cases zs <;> simp_all [harvestString, endsString, middleString]
  | _ => simp_all [harvestString, endsString, middleString]

omit [Fintype α] [DecidableEq α] in
theorem applyTrieTests_inr {t : TrieState α} {e : RoundEnd TrieSite}
    (h : applyTrieTests C t = .inr e) : runTrieTests C t = some e := by
  unfold applyTrieTests at h
  split at h
  · rename_i e' he'; simp only [Sum.inr.injEq] at h; rw [he', h]
  · simp at h

omit [Fintype α] [DecidableEq α] in
theorem runTrieTests_consistent {t : TrieState α} (h : runTrieTests C t = some .consistent) :
    seqRate C.disagreementRate C.disagreementAlternative (C.level t.stretches) t.successProbes
      t.successDisagree := by
  unfold runTrieTests at h
  split_ifs at h with h1 h2 h3 h4 <;> simp_all

omit [Fintype α] [DecidableEq α] in
open scoped Classical in
/-- A counted probe keeps the stretch and counts in the success test as its disagreement. -/
theorem counted_step {s : TrieState α} {x : FreeMonoid α} {o : Outcome α}
    (ho : probeOrAgree read s.tree s.edges (walkStartIn s.expanded x) x = o)
    (hne : ∀ p c u t, o ≠ .unlearnedEdge p c u t) :
    let t := countTrieOutcome s o (prefixOf x (walkStartIn s.expanded x))
    t.stretches = s.stretches ∧ t.tree = s.tree ∧ t.edges = s.edges ∧ t.expanded = s.expanded
      ∧ t.successProbes = s.successProbes + 1
      ∧ t.successDisagree = s.successDisagree + if x ∈ trieDis read s then 1 else 0 := by
  refine ⟨rfl, rfl, rfl, rfl, rfl, ?_⟩
  have hiff := successDisagrees_iff read ⟨s.tree, s.edges⟩ s.expanded x (by rw [ho]; exact hne)
  rw [ho] at hiff
  show s.successDisagree + (if successDisagrees o then 1 else 0) = _
  by_cases hx : x ∈ trieDis read s
  · rw [if_pos (hiff.2 hx), if_pos hx]
  · rw [if_neg (fun h => hx (hiff.1 h)), if_neg hx]

open scoped Classical in
theorem trieStep_inl {s s' : TrieState α} {x : FreeMonoid α}
    (h : trieStep C cap read s x = .inl s') :
    (s'.stretches = s.stretches ∧ s'.tree = s.tree ∧ s'.edges = s.edges
        ∧ s'.expanded = s.expanded ∧ s'.successProbes = s.successProbes + 1
        ∧ s'.successDisagree = s.successDisagree + if x ∈ trieDis read s then 1 else 0)
      ∨ (s'.stretches = s.stretches + 1 ∧ s'.successProbes = 0 ∧ s'.successDisagree = 0) := by
  unfold trieStep applyTrieProbeQ at h
  rw [eval_bind] at h
  generalize ho' : (probeOrAgreeQ s.tree s.edges (walkStartIn s.expanded x) x).eval read = o at h
  have ho : probeOrAgree read s.tree s.edges (walkStartIn s.expanded x) x = o := ho'
  by_cases hr : ∃ z ∈ harvestString o, z ∈ s.seen.map Prod.fst
  · rw [if_pos hr, eval_pure, Sum.inl.injEq] at h
    subst h
    obtain ⟨z, hz, -⟩ := hr
    rw [← ho] at hz
    have hx : x ∈ trieDis read s := mem_of_harvestString read ⟨s.tree, s.edges⟩ s.expanded x hz
    by_cases hX : s.skipExpands cap o (prefixOf x (walkStartIn s.expanded x)) = s.expanded
    · left
      simp [TrieState.skipRepeat, hX, hx]
    · right
      simp [TrieState.skipRepeat, hX]
  rw [if_neg hr] at h
  have hc : ∀ (_ : ∀ p c u t, o ≠ .unlearnedEdge p c u t),
      applyTrieTests C (countTrieOutcome s o (prefixOf x (walkStartIn s.expanded x))) = .inl s' →
      (s'.stretches = s.stretches ∧ s'.tree = s.tree ∧ s'.edges = s.edges
        ∧ s'.expanded = s.expanded ∧ s'.successProbes = s.successProbes + 1
        ∧ s'.successDisagree = s.successDisagree + if x ∈ trieDis read s then 1 else 0)
      ∨ (s'.stretches = s.stretches + 1 ∧ s'.successProbes = 0 ∧ s'.successDisagree = 0) :=
    fun hne h' => by
    rw [applyTrieTests_inl C h']
    exact Or.inl (counted_step read ho hne)
  cases o with
  | unlearnedEdge p c u t =>
    simp only [eval_pure, Sum.inl.injEq] at h; subst h
    exact Or.inr ⟨rfl, rfl, rfl⟩
  | wrongEdge p c u t =>
    simp only at h
    split at h
    · rw [eval_map, splitTrieLeafQ, eval_map, Sum.inl.injEq] at h
      subst h
      exact Or.inr ⟨rfl, rfl, rfl⟩
    · rw [eval_pure] at h
      exact hc (fun _ _ _ _ h => by cases h) h
  | agree => rw [eval_pure] at h; exact hc (fun _ _ _ _ h => by cases h) h
  | undecidedStart z => rw [eval_pure] at h; exact hc (fun _ _ _ _ h => by cases h) h
  | undecided zs cl => rw [eval_pure] at h; exact hc (fun _ _ _ _ h => by cases h) h

open scoped Classical in
theorem trieStep_consistent {s : TrieState α} {x : FreeMonoid α}
    (h : trieStep C cap read s x = .inr .consistent) :
    seqRate C.disagreementRate C.disagreementAlternative (C.level s.stretches)
      (s.successProbes + 1) (s.successDisagree + if x ∈ trieDis read s then 1 else 0) := by
  unfold trieStep applyTrieProbeQ at h
  rw [eval_bind] at h
  generalize ho' : (probeOrAgreeQ s.tree s.edges (walkStartIn s.expanded x) x).eval read = o at h
  have ho : probeOrAgree read s.tree s.edges (walkStartIn s.expanded x) x = o := ho'
  by_cases hr : ∃ z ∈ harvestString o, z ∈ s.seen.map Prod.fst
  · rw [if_pos hr, eval_pure] at h
    cases h
  rw [if_neg hr] at h
  have hc : ∀ (_ : ∀ p c u t, o ≠ .unlearnedEdge p c u t),
      applyTrieTests C (countTrieOutcome s o (prefixOf x (walkStartIn s.expanded x)))
        = .inr .consistent →
      seqRate C.disagreementRate C.disagreementAlternative (C.level s.stretches)
        (s.successProbes + 1) (s.successDisagree + if x ∈ trieDis read s then 1 else 0) :=
    fun hne h' => by
    have hf := runTrieTests_consistent C (applyTrieTests_inr C h')
    obtain ⟨hj, -, -, -, hn, hd⟩ := counted_step read ho hne
    rw [hj, hn, hd] at hf
    exact hf
  cases o with
  | unlearnedEdge p c u t => simp only [eval_pure] at h; cases h
  | wrongEdge p c u t =>
    simp only at h
    split at h
    · rw [eval_map] at h; cases h
    · rw [eval_pure] at h
      exact hc (fun _ _ _ _ h => by cases h) h
  | agree => rw [eval_pure] at h; exact hc (fun _ _ _ _ h => by cases h) h
  | undecidedStart z => rw [eval_pure] at h; exact hc (fun _ _ _ _ h => by cases h) h
  | undecided zs cl => rw [eval_pure] at h; exact hc (fun _ _ _ _ h => by cases h) h

open scoped Classical in
/-- A false success within a stretch is a firing of its success test over the stretch's probes,
counting those in its disagreement set. -/
theorem segB_fires (s₀ : TrieState α) :
    ∀ (xs ys : List (FreeMonoid α)) (s : TrieState α), s.stretches = s₀.stretches →
      trieDis read s = trieDis read s₀ → s.successProbes = ys.length →
      s.successDisagree = ys.countP (fun y => decide (y ∈ trieDis read s₀)) →
      SegB (trieStep C cap read) (FalseConsistent C cap read D) SameStretch s xs →
      C.disagreementRate < D.real (trieDis read s₀) ∧ ∃ n, n ≤ xs.length ∧
        seqRate C.disagreementRate C.disagreementAlternative (C.level s₀.stretches)
          (ys.length + n) ((ys ++ xs.take n).countP fun y => decide (y ∈ trieDis read s₀))
  | [], _, _, _, _, _, _, h => by simp [SegB] at h
  | x :: xs, ys, s, hj, hP, hn, hh, h => by
    rcases h with ⟨hc, hbad⟩ | ⟨s', hst, hsm, hseg⟩
    · refine ⟨hP ▸ hbad, 1, by simp, ?_⟩
      have hf := trieStep_consistent C cap read hc
      rw [hn, hh, hj, hP] at hf
      convert hf using 2
      simp [List.countP_append]
    · rcases trieStep_inl C cap read hst with ⟨hj', hT, hE, hX, hn', hh'⟩ | ⟨hj', -, -⟩
      · have hP' : trieDis read s' = trieDis read s₀ := by
          rw [← hP]; unfold trieDis; rw [hT, hE, hX]
        obtain ⟨hb, n, hnl, hf⟩ := segB_fires s₀ xs (ys ++ [x]) s' (hj'.trans hj) hP'
          (by rw [hn', hn]; simp) (by rw [hh', hh, hP]; simp [List.countP_append]) hseg
        refine ⟨hb, n + 1, by simp; omega, ?_⟩
        have e1 : ys.length + (n + 1) = (ys ++ [x]).length + n := by simp; omega
        have e2 : ys ++ (x :: xs).take (n + 1) = (ys ++ [x]) ++ xs.take n := by simp
        rw [e1, e2]
        exact hf
      · simp only [SameStretch] at hsm
        omega

theorem trieSeg_le [IsProbabilityMeasure D] (s₀ : TrieState α) (h0 : s₀.successProbes = 0)
    (h0' : s₀.successDisagree = 0) (hρ0 : 0 < C.disagreementRate) (hρ1 : C.disagreementRate < 1)
    (hδ : 0 < C.δ) (N : ℕ) :
    (Measure.pi fun _ : Fin N => D)
        {xs | SegB (trieStep C cap read) (FalseConsistent C cap read D) SameStretch s₀
          (List.ofFn xs)}
      ≤ ENNReal.ofReal (C.level s₀.stretches) := by
  classical
  by_cases hbad : C.disagreementRate < D.real (trieDis read s₀)
  · have hsub : {xs : Fin N → FreeMonoid α | SegB (trieStep C cap read)
          (FalseConsistent C cap read D) SameStretch s₀ (List.ofFn xs)}
        ⊆ {xs | ∃ n ≤ N, seqRate C.disagreementRate C.disagreementAlternative
          (C.level s₀.stretches) n
            (((List.ofFn xs).take n).countP fun x => decide (x ∈ trieDis read s₀))} := by
      intro xs hxs
      obtain ⟨-, n, hn, hf⟩ := segB_fires C cap read D s₀ (List.ofFn xs) [] s₀ rfl rfl
        (by simp [h0]) (by simp [h0']) hxs
      exact ⟨n, by simpa using hn, by simpa using hf⟩
    have hsound := seqRateSound D (· ∈ trieDis read s₀) C.disagreementRate
      C.disagreementAlternative (C.level s₀.stretches) N hρ0 hρ1
      (by unfold Level3Settings.disagreementAlternative; positivity)
      (by unfold Level3Settings.disagreementAlternative; linarith)
      (by unfold Level3Settings.level; positivity)
      (mul_nonpos_of_nonneg_of_nonpos (by simp only [Set.setOf_mem_eq]; linarith)
        (by unfold Level3Settings.disagreementAlternative; linarith))
    calc _ ≤ _ := measure_mono hsub
      _ = ENNReal.ofReal ((Measure.pi fun _ : Fin N => D).real _) :=
          (ofReal_measureReal (measure_ne_top _ _)).symm
      _ ≤ _ := ENNReal.ofReal_le_ofReal (by convert hsound)
  · have : {xs : Fin N → FreeMonoid α | SegB (trieStep C cap read)
        (FalseConsistent C cap read D) SameStretch s₀ (List.ofFn xs)} = ∅ := by
      ext xs
      simp only [Set.mem_setOf_eq, Set.mem_empty_iff_false, iff_false]
      intro hxs
      exact hbad (segB_fires C cap read D s₀ (List.ofFn xs) [] s₀ rfl rfl (by simp [h0])
        (by simp [h0']) hxs).1
    rw [this, measure_empty]
    exact bot_le

/-- A run ending consistent at more than `ε` disagreeing probes has a falsely consistent step. -/
theorem anyB_of_run : ∀ (s : TrieState α) (xs : List (FreeMonoid α)),
    ((runTrieProbesQ C cap s xs).eval read).2 = .consistent →
    C.disagreementRate < D.real (trieDis read ((runTrieProbesQ C cap s xs).eval read).1) →
    AnyB (trieStep C cap read) (FalseConsistent C cap read D) s xs
  | s, [], h, _ => by simp [runTrieProbesQ] at h
  | s, x :: xs, h, hb => by
    unfold runTrieProbesQ at h hb
    rw [eval_bind] at h hb
    rcases hst : (applyTrieProbeQ C cap s x).eval read with s' | e
    · rw [hst] at h hb
      exact Or.inr ⟨s', hst, anyB_of_run s' xs h hb⟩
    · rw [hst, eval_pure] at h hb
      simp only at h hb
      subst h
      exact Or.inl ⟨hst, hb⟩

end Random

open MeasureTheory Random

theorem level3TrieConsistentSound : Level3TrieConsistentSound := by
  intro α _ _ C cap read D _ hδ hρ0 hρ1
  classical
  have hsub : {draws : Fin C.probeBudget → FreeMonoid α |
      let r := indecisiveLStarLevel3Trie C cap read draws
      r.ending = .consistent
        ∧ C.disagreementRate < D.real (trieDisagreeSet read r.expanded r.hypothesis)}
      ⊆ {xs | AnyB (trieStep C cap read) (FalseConsistent C cap read D) trieInitialState
          (List.ofFn xs)} := by
    intro xs hxs
    simp only [Set.mem_setOf_eq, indecisiveLStarLevel3Trie, level3TrieQ, Query.eval_map] at hxs
    exact anyB_of_run C cap read D _ _ hxs.1 hxs.2
  have hany := anyB_le_tail (trieStep C cap read) (FalseConsistent C cap read D) SameStretch D
    (fun s => s.stretches) (fun j => ENNReal.ofReal (C.level j))
    (fun s => s.successProbes = 0 ∧ s.successDisagree = 0)
    (fun _ _ _ _ hsm => hsm)
    (fun s x s' hst hsm => by
      rcases trieStep_inl C cap read hst with ⟨hj, -⟩ | ⟨hj, h0, h0'⟩
      · exact absurd hj hsm
      · exact ⟨⟨h0, h0'⟩, hj⟩)
    (fun s hF N => trieSeg_le C cap read D s hF.1 hF.2 hρ0 hρ1 hδ N) C.probeBudget trieInitialState
  have hseg0 := trieSeg_le C cap read D (trieInitialState : TrieState α) rfl rfl hρ0 hρ1 hδ
    C.probeBudget
  have htot : (Measure.pi fun _ : Fin C.probeBudget => D)
      {xs | AnyB (trieStep C cap read) (FalseConsistent C cap read D) trieInitialState
        (List.ofFn xs)} ≤ ENNReal.ofReal (C.δ / 8) := by
    calc _ ≤ _ := hany
      _ ≤ ENNReal.ofReal (C.level 0) + stretchTail (fun j => ENNReal.ofReal (C.level j)) 0 :=
          add_le_add hseg0 le_rfl
      _ = ∑' i, ENNReal.ofReal (C.level i) :=
          (tsum_eq_stretchTail (fun j => ENNReal.ofReal (C.level j))).symm
      _ ≤ _ := tsum_level_le C hδ.le
  calc _ ≤ (Measure.pi fun _ : Fin C.probeBudget => D).real
        {xs | AnyB (trieStep C cap read) (FalseConsistent C cap read D) trieInitialState
          (List.ofFn xs)} := measureReal_mono hsub
    _ ≤ C.δ / 8 := ENNReal.toReal_le_of_le_ofReal (by positivity) htot
    _ ≤ C.δ := by linarith

theorem level3TrieConsistentCorrect : Level3TrieConsistentCorrect := by
  intro α _ _ C cap read L D _ hδ hρ0 hρ1
  refine le_trans (measureReal_mono fun draws hd => ?_)
    (level3TrieConsistentSound C cap read D hδ hρ0 hρ1)
  refine ⟨hd.1, lt_of_not_ge fun hle => absurd hd.2 (not_lt.2 ?_)⟩
  exact (trieSinglePieceCorrect read _ _ (level3TrieRootOne C cap read draws) L D).trans
    (by linarith)

end OrthoDFA

end
