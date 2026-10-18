import OrthoDFA.Proofs.RandomRisk
import OrthoDFA.Proofs.Level2Steps

/-!
# The tree and the steps under reads that are never wrong

A decided read is on its state's side, so two strings of one state that both sift decidedly
reach the same leaf, and the leaves past the first two are at most `|Q|`. Every split is on a
letter and the midfix where two leaves part, so the tree is in `classSet |Q|`.

A step that keeps the hypothesis counts one more probe into the stretch; one that changes it
starts a fresh stretch and lowers `Ψ`, #417's count of the splits and learnings left.
-/

namespace OrthoDFA

namespace Random


variable {α : Type*}

/-- No read is on its state's wrong side. -/
def NoWrong {σ : Type*} (M : DFA α σ) (side : σ → Bool) (read : FreeMonoid α → ARU) : Prop :=
  ∀ z, read z ≠ if side (M.eval z.toList) then .reject else .accept

open scoped Classical in
/-- The midfixes of the tree's nodes. -/
noncomputable def mids : DTree α → Finset (FreeMonoid α)
  | .leaf => ∅
  | .node m r a => insert m (mids r ∪ mids a)

open scoped Classical in
/-- The strings a probe of `x` can read: a prefix of length at least `k` followed by a midfix. -/
noncomputable def pot (k : ℕ) (x : FreeMonoid α) (T : DTree α) : Finset (FreeMonoid α) :=
  (Finset.Icc k x.toList.length).biUnion fun p => (mids T).image (prefixOf x p * ·)

/-- The probes that can read a string at a good read-state undecided. -/
def PotGood {σ : Type*} (M : DFA α σ) (U : σ → ℝ) (θ : ℝ) (k : ℕ) (read : FreeMonoid α → ARU)
    (T : DTree α) : Set (FreeMonoid α) :=
  {x | ∃ z ∈ pot k x T, ¬ BadAt U θ (M.eval z.toList) ∧ read z = .undecided}

open scoped Classical in
/-- The trees at most `n` splits reach, each at a leaf on a letter and the midfix where two
leaves part. -/
noncomputable def classSet [Fintype α] : ℕ → Finset (DTree α)
  | 0 => {.node 1 .leaf .leaf}
  | n + 1 => classSet n ∪ (classSet n).biUnion fun T =>
      (T.leaves.toFinset ×ˢ (Finset.univ : Finset α) ×ˢ T.leaves.toFinset ×ˢ
        T.leaves.toFinset).image
        fun q => T.splitAt (FreeMonoid.of q.2.1 * T.midfixAt (commonPrefix q.2.2.1 q.2.2.2)) q.1

variable [DecidableEq α] (read : FreeMonoid α → ARU) (C : Level3Settings)

/-- The splits and learnings left, as #417 counts them, within `nσ + 2` leaves. -/
def Ψr [Fintype α] (nσ : ℕ) (s : Level3State α) : ℕ :=
  Level2.Ψ nσ (Level2.newHypothesis s.tree s.edges 0)

/-- The invariant that holds whatever the reads. -/
structure Inv [Fintype α] (s : Level3State α) : Prop where
  edges : EdgesOK read s.tree s.edges
  cls : s.tree ∈ classSet (s.tree.leaves.length - 2)
  two : 2 ≤ s.tree.leaves.length

/-! ## The tree -/

section Tree

omit [DecidableEq α] in
/-- Two strings of one state that both sift decidedly reach the same leaf. -/
theorem sift_inl_congr {σ : Type*} {M : DFA α σ} {side : σ → Bool} (hW : NoWrong M side read)
    {z z' : FreeMonoid α} (hz : M.eval z.toList = M.eval z'.toList) :
    ∀ (T : DTree α) (p p' : List Bool), T.sift read z = .inl p → T.sift read z' = .inl p' →
      p = p'
  | .leaf, p, p', h, h' => by
    simp only [DTree.sift, Sum.inl.injEq] at h h'; rw [← h, ← h']
  | .node m r a, p, p', h, h' => by
    have hs : M.eval (z * m).toList = M.eval (z' * m).toList := by
      simp only [FreeMonoid.toList_mul, DFA.eval, DFA.evalFrom_of_append]
      exact congrArg (M.evalFrom · m.toList) hz
    have hw := hW (z * m)
    have hw' := hW (z' * m)
    rw [hs] at hw
    simp only [DTree.sift] at h h'
    split at h <;> rename_i hr <;> split at h' <;> rename_i hr'
    · rcases hq : a.sift read z with q | q <;> rw [hq] at h <;> simp only [Sum.map_inl,
        Sum.map_inr, Sum.inl.injEq, reduceCtorEq] at h
      rcases hq' : a.sift read z' with q' | q' <;> rw [hq'] at h' <;> simp only [Sum.map_inl,
        Sum.map_inr, Sum.inl.injEq, reduceCtorEq] at h'
      rw [← h, ← h', sift_inl_congr hW hz a q q' hq hq']
    · exfalso
      rw [hr] at hw; rw [hr'] at hw'
      cases hsd : side (M.eval (z' * m).toList) <;> simp_all
    · simp at h'
    · exfalso
      rw [hr] at hw; rw [hr'] at hw'
      cases hsd : side (M.eval (z' * m).toList) <;> simp_all
    · rcases hq : r.sift read z with q | q <;> rw [hq] at h <;> simp only [Sum.map_inl,
        Sum.map_inr, Sum.inl.injEq, reduceCtorEq] at h
      rcases hq' : r.sift read z' with q' | q' <;> rw [hq'] at h' <;> simp only [Sum.map_inl,
        Sum.map_inr, Sum.inl.injEq, reduceCtorEq] at h'
      rw [← h, ← h', sift_inl_congr hW hz r q q' hq hq']
    · simp at h'
    · simp at h
    · simp at h
    · simp at h

omit [DecidableEq α] in
theorem leaves_le_of_reached' {σ : Type*} [Fintype σ] {M : DFA α σ} {side : σ → Bool}
    (hW : NoWrong M side read) {T : DTree α} (h : Reached read T) :
    T.leaves.length ≤ Fintype.card σ + 2 := by
  classical
  rw [← List.toFinset_card_of_nodup (DTree.leaves_nodup T)]
  set A := T.leaves.toFinset.filter fun p => ∃ w, T.sift read w = .inl p
  have hsub : T.leaves.toFinset ⊆ A ∪ {[false], [true]} := by
    intro p hp
    rcases h p (List.mem_toFinset.1 hp) with h1 | h1 | h1
    · simp [h1]
    · simp [h1]
    · exact Finset.mem_union_left _ (Finset.mem_filter.2 ⟨hp, h1⟩)
  have hA : A.card ≤ Fintype.card σ := by
    let f : List Bool → σ := fun p =>
      if hp : ∃ w, T.sift read w = .inl p then M.eval hp.choose.toList else M.eval []
    refine Finset.card_le_card_of_injOn f (fun _ _ => Finset.mem_univ _) ?_
    intro p hp p' hp' hf
    have h1 := (Finset.mem_filter.1 hp).2
    have h2 := (Finset.mem_filter.1 hp').2
    simp only [f, h1, h2, dite_true] at hf
    exact sift_inl_congr read hW hf T p p' h1.choose_spec h2.choose_spec
  calc T.leaves.toFinset.card ≤ (A ∪ {[false], [true]}).card := Finset.card_le_card hsub
    _ ≤ A.card + ({[false], [true]} : Finset (List Bool)).card := Finset.card_union_le _ _
    _ ≤ Fintype.card σ + 2 := by
      have : ({[false], [true]} : Finset (List Bool)).card ≤ 2 := Finset.card_le_two
      omega

theorem classSet_mono [Fintype α] (n : ℕ) :
    (classSet n : Finset (DTree α)) ⊆ classSet (n + 1) := by
  classical
  intro T h
  simp only [classSet]
  exact Finset.mem_union_left _ h

theorem classSet_mono' [Fintype α] {n n' : ℕ} (h : n ≤ n') :
    (classSet n : Finset (DTree α)) ⊆ classSet n' := by
  induction h with
  | refl => exact le_rfl
  | step _ ih => exact ih.trans (classSet_mono _)

theorem classSet_split [Fintype α] {n : ℕ} {T : DTree α} (hT : T ∈ classSet n) {p t t₀ : List Bool}
    {c : α} (hp : p ∈ T.leaves) (ht : t ∈ T.leaves) (ht₀ : t₀ ∈ T.leaves) :
    T.splitAt (FreeMonoid.of c * T.midfixAt (commonPrefix t t₀)) p ∈ classSet (n + 1) := by
  classical
  simp only [classSet]
  refine Finset.mem_union_right _ (Finset.mem_biUnion.2 ⟨T, hT, Finset.mem_image.2
    ⟨(p, c, t, t₀), ?_, rfl⟩⟩)
  simp [hp, ht, ht₀]

end Tree

/-! ## The steps -/

/-- Where a probe disagrees at decided reads: the edge and the target there. -/
def recOf : Outcome α → Option (List Bool × α × List Bool)
  | .wrongEdge p c _ t => some (p, c, t)
  | _ => none

section Steps

variable [Fintype α] {σ : Type*} [Fintype σ] {M : DFA α σ} {side : σ → Bool}

omit [DecidableEq α] in
theorem inv_start : Inv read (initialState : Level3State α) where
  edges := fun _ _ _ _ _ h => by simp [initialState, newHypothesis] at h
  cls := by simp [initialState, newHypothesis, DTree.leaves, classSet]
  two := by simp [initialState, newHypothesis, DTree.leaves]

theorem reached_start : Reached read (initialState : Level3State α).tree := by
  intro p hp
  simp only [initialState, newHypothesis, DTree.leaves, List.map_cons, List.map_nil,
    List.cons_append, List.nil_append, List.mem_cons, List.not_mem_nil, or_false] at hp
  rcases hp with rfl | rfl
  · exact Or.inl rfl
  · exact Or.inr (Or.inl rfl)

/-- `Inv` with every leaf past the first two reached, which reads that are never wrong keep. -/
def RInv [Fintype α] (s : Level3State α) : Prop := Inv read s ∧ Reached read s.tree

theorem Ψr_start (nσ : ℕ) :
    Ψr nσ (initialState : Level3State α) + 1 ≤ Level2.stretches nσ (Fintype.card α) :=
  Level2.Ψ_start nσ

omit [DecidableEq α] in
theorem finish_inl {t s' : Level3State α} (h : applyTests C t = .inl s') : s' = t
    ∧ runTests C t = none := by
  unfold applyTests at h
  split at h
  · simp at h
  · rename_i hl
    exact ⟨by simpa using h.symm, hl⟩

theorem countProbe_tree (s : Level3State α) (x : FreeMonoid α) :
    (countProbe read C s x).tree = s.tree := rfl
theorem countProbe_edges (s : Level3State α) (x : FreeMonoid α) :
    (countProbe read C s x).edges = s.edges := rfl
theorem countProbe_probeCount (s : Level3State α) (x : FreeMonoid α) :
    (countProbe read C s x).probeCount = s.probeCount + 1 := rfl

/-- The steps that keep the hypothesis: a look after counting a probe that does not disagree at
decided reads. -/
theorem step_cases {s s' : Level3State α} {x : FreeMonoid α} (hs : Inv read s)
    (h : applyProbe read C s x = .inl s') :
    (∃ p c u t, probeOrAgree read s.tree s.edges C.walkStart x = .unlearnedEdge p c u t
        ∧ s.edges (p, c) = none ∧
        s.tree.sift read u = .inl p ∧ s.tree.sift read (u * FreeMonoid.of c) = .inl t ∧
        p ∈ s.tree.leaves ∧
        s' = newHypothesis s.tree (Function.update s.edges (p, c) (some (t, u))) (s.hypotheses + 1))
    ∨ (∃ p c u t t₀ w₀, probeOrAgree read s.tree s.edges C.walkStart x = .wrongEdge p c u t ∧
        s.edges (p, c) = some (t₀, w₀) ∧ t₀ ≠ t ∧ p ∈ s.tree.leaves ∧
        s.tree.sift read u = .inl p ∧ s.tree.sift read (u * FreeMonoid.of c) = .inl t ∧
        s' = splitLeaf read s p c t t₀)
    ∨ (s' = countProbe read C s x ∧ runTests C s' = none
        ∧ recOf (probeOrAgree read s.tree s.edges C.walkStart x) = none) := by
  have hOK := probeOrAgree_ok read s.tree s.edges C.walkStart hs.edges x
  unfold applyProbe at h
  generalize hg : probeOrAgree read s.tree s.edges C.walkStart x = o at h hOK
  cases o with
  | unlearnedEdge p c u t =>
    obtain ⟨hp, hu, huc, hE⟩ := hOK
    exact Or.inl ⟨p, c, u, t, rfl, hE, hu, huc, hp, by simpa using h.symm⟩
  | wrongEdge p c u t =>
    obtain ⟨hp, hu, huc, t₀, w₀, hE, hne⟩ := hOK
    simp only [hE, Sum.inl.injEq] at h
    exact Or.inr (Or.inl ⟨p, c, u, t, t₀, w₀, rfl, hE, hne, hp, hu, huc, h.symm⟩)
  | agree | undecidedStart _ | undecided _ _ =>
    obtain ⟨h1, h2⟩ := finish_inl C h
    exact Or.inr (Or.inr ⟨h1, h1 ▸ h2, rfl⟩)

/-- A step that keeps the hypothesis counts one more probe. -/
theorem step_same {s s' : Level3State α} (hs : Inv read s) {x : FreeMonoid α}
    (h : applyProbe read C s x = .inl s') (ht : s'.tree = s.tree) (he : s'.edges = s.edges) :
    Inv read s' ∧ (∀ nσ, Ψr nσ s' = Ψr nσ s) ∧ s'.probeCount = s.probeCount + 1
      ∧ s'.hypotheses = s.hypotheses := by
  rcases step_cases read C hs h with
    ⟨p, c, u, t, -, hE, -, -, -, rfl⟩ | ⟨p, c, u, t, t₀, w₀, -, hE, hne, hp, -, -, rfl⟩ |
    ⟨rfl, -, -⟩
  · exfalso
    have := congrFun he (p, c)
    simp only [newHypothesis, Function.update_self, hE] at this
    exact absurd this (by simp)
  · exfalso
    have := congrArg List.length (congrArg DTree.leaves ht)
    simp only [splitLeaf, newHypothesis] at this
    rw [DTree.length_leaves_splitAt _ _ _ hp] at this
    omega
  · exact ⟨⟨hs.edges, hs.cls, hs.two⟩, fun _ => rfl, rfl, rfl⟩

omit [DecidableEq α] in
/-- A split on a letter and the midfix where its targets part keeps every leaf past the first
two reached. -/
theorem split_reached {s : Level3State α} (hr : Reached read s.tree) {p t t₀ : List Bool} {c : α}
    {u w₀ : FreeMonoid α} (hp : p ∈ s.tree.leaves) (hu : s.tree.sift read u = .inl p)
    (huc : s.tree.sift read (u * FreeMonoid.of c) = .inl t) (hw₀ : s.tree.sift read w₀ = .inl p)
    (hw₀c : s.tree.sift read (w₀ * FreeMonoid.of c) = .inl t₀) (hne : t₀ ≠ t) :
    Reached read (s.tree.splitAt (FreeMonoid.of c * s.tree.midfixAt (commonPrefix t t₀)) p) := by
  have hpart := DTree.sift_part read s.tree _ _ t t₀ huc hw₀c (Ne.symm hne)
  refine reached_splitAt read hr hp hu hw₀ ?_ ?_ ?_
  · simpa [mul_assoc] using hpart.1
  · simpa [mul_assoc] using hpart.2.1
  · simpa [mul_assoc] using hpart.2.2

/-- A split keeps the invariant and the tree in the class, and lowers `Ψr`. -/
theorem split_ok (hW : NoWrong M side read) {s : Level3State α} (hs : Inv read s)
    (hr : Reached read s.tree)
    {p t t₀ : List Bool} {c : α} {u w₀ : FreeMonoid α} (hp : p ∈ s.tree.leaves)
    (hu : s.tree.sift read u = .inl p) (huc : s.tree.sift read (u * FreeMonoid.of c) = .inl t)
    (hw₀ : s.tree.sift read w₀ = .inl p) (hw₀c : s.tree.sift read (w₀ * FreeMonoid.of c) = .inl t₀)
    (hne : t₀ ≠ t) :
    RInv read (splitLeaf read s p c t t₀)
      ∧ Ψr (Fintype.card σ) (splitLeaf read s p c t t₀) < Ψr (Fintype.card σ) s := by
  have hreach := split_reached read hr hp hu huc hw₀ hw₀c hne
  have hlen :=
    DTree.length_leaves_splitAt (FreeMonoid.of c * s.tree.midfixAt (commonPrefix t t₀)) s.tree p hp
  have ht := DTree.sift_inl_mem read _ _ t huc
  have ht₀ := DTree.sift_inl_mem read _ _ t₀ hw₀c
  refine ⟨⟨⟨edgesOK_retarget read hs.edges hp _, ?_,
    by simp only [splitLeaf, newHypothesis]; rw [hlen]; have := hs.two; omega⟩, hreach⟩, ?_⟩
  · simp only [splitLeaf, newHypothesis]
    rw [hlen, show s.tree.leaves.length + 1 - 2 = s.tree.leaves.length - 2 + 1 by
      have := hs.two; omega]
    exact classSet_split hs.cls hp ht ht₀
  · exact Level2.Ψ_lt_split _ (s := Level2.newHypothesis s.tree s.edges 0) hp
      (leaves_le_of_reached' read hW hreach) _ 0

/-- A step that changes the hypothesis starts a fresh stretch and lowers `Ψr`. -/
theorem step_change (hW : NoWrong M side read) {s s' : Level3State α} (hs : RInv read s)
    {x : FreeMonoid α} (h : applyProbe read C s x = .inl s')
    (hne : ¬ (s'.tree = s.tree ∧ s'.edges = s.edges)) :
    RInv read s' ∧ s' = newHypothesis s'.tree s'.edges s'.hypotheses
      ∧ s'.hypotheses = s.hypotheses + 1 ∧ Ψr (Fintype.card σ) s' < Ψr (Fintype.card σ) s := by
  rcases step_cases read C hs.1 h with
    ⟨p, c, u, t, -, hE, hu, huc, hp, rfl⟩ |
    ⟨p, c, u, t, t₀, w₀, -, hE, htt, hp, hu, huc, rfl⟩ | ⟨rfl, -, -⟩
  · exact ⟨⟨⟨edgesOK_upd read hs.1.edges hu huc, hs.1.cls, hs.1.two⟩, hs.2⟩, rfl, rfl,
      Level2.Ψ_lt_learn _ (s := Level2.newHypothesis s.tree s.edges 0) hp hE _ 0⟩
  · obtain ⟨hw₀, hw₀c⟩ := hs.1.edges p hp c t₀ w₀ hE
    obtain ⟨hinv, hΨ⟩ := split_ok read hW hs.1 hs.2 hp hu huc hw₀ hw₀c htt
    exact ⟨hinv, rfl, rfl, hΨ⟩
  · exact absurd ⟨rfl, rfl⟩ hne

omit [Fintype σ] in
theorem applyProbe_ne_failed (s : Level3State α) (x : FreeMonoid α) :
    applyProbe read C s x ≠ .inr .failed := by
  have hlook : ∀ t : Level3State α, applyTests C t ≠ .inr .failed := by
    intro t ht
    unfold applyTests runTests at ht
    split_ifs at ht <;> simp_all
  unfold applyProbe
  split
  · simp
  · split
    · simp
    · exact hlook _
  · exact hlook _

theorem cls_mem (hW : NoWrong M side read) {s : Level3State α} (hs : RInv read s) :
    s.tree ∈ classSet (Fintype.card σ) :=
  classSet_mono' (by have := leaves_le_of_reached' read hW hs.2; omega) hs.1.cls

end Steps

end Random

end OrthoDFA
