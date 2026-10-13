import OrthoDFA.Level3_5Claim
import OrthoDFA.Proofs.RandomTree

/-!
# Level 3.5's steps

As at level 3 before its splits became immediate: a step that keeps the hypothesis counts one more
probe into the stretch; one that changes it starts a fresh stretch and lowers `Ψr`, which counts
the splits left and each edge's learnings and redirects.
-/

namespace OrthoDFA

namespace Random

namespace Eps

open OrthoDFA.Ideal (probe probeOrAgree Outcome retarget EdgesOK Reached OutcomeOK)

variable {α : Type*} [DecidableEq α] (read : FreeMonoid α → ARU) (C : Level3_5Settings)

section Potential

variable [Fintype α]

/-- What an edge has left to change before the next split: learning and redirecting, or one. -/
def wt (s : Level3_5State α) (qc : List Bool × α) : ℕ :=
  match s.edges qc with
  | none => 2
  | some _ => if s.redirected qc then 0 else 1

def Φ (s : Level3_5State α) : ℕ := ∑ qc ∈ s.tree.leaves.toFinset ×ˢ Finset.univ, wt s qc

def X : ℕ := 2 * C.maxLeaves * Fintype.card α + 1

/-- The splits, learnings and redirects left. -/
def Ψr (s : Level3_5State α) : ℕ := (C.maxLeaves - s.tree.leaves.length) * X (α := α) C + Φ s

omit [DecidableEq α] [Fintype α] in
theorem wt_le (s : Level3_5State α) (qc : List Bool × α) : wt s qc ≤ 2 := by
  unfold wt; split
  · exact le_rfl
  · split <;> omega

omit [DecidableEq α] in
theorem Φ_le (s : Level3_5State α) : Φ s ≤ 2 * s.tree.leaves.length * Fintype.card α := by
  unfold Φ
  calc ∑ qc ∈ s.tree.leaves.toFinset ×ˢ Finset.univ, wt s qc
      ≤ (s.tree.leaves.toFinset ×ˢ (Finset.univ : Finset α)).card • 2 :=
        Finset.sum_le_card_nsmul _ _ _ fun qc _ => wt_le s qc
    _ ≤ 2 * s.tree.leaves.length * Fintype.card α := by
      rw [Finset.card_product, Finset.card_univ, smul_eq_mul]
      have := List.toFinset_card_le s.tree.leaves
      nlinarith

/-- Changing one edge of a leaf to a lower weight lowers `Ψr`. -/
theorem Ψ_lt_upd {s : Level3_5State α} {p : List Bool} {c : α} (hp : p ∈ s.tree.leaves)
    (v : Option (List Bool × FreeMonoid α)) (b : Bool)
    (hlt : wt (newHypothesis s.tree (Function.update s.edges (p, c) v)
      (Function.update s.redirected (p, c) b)) (p, c) < wt s (p, c)) :
    Ψr C (newHypothesis s.tree (Function.update s.edges (p, c) v)
      (Function.update s.redirected (p, c) b)) < Ψr C s := by
  unfold Ψr
  have : Φ (newHypothesis s.tree (Function.update s.edges (p, c) v)
      (Function.update s.redirected (p, c) b)) < Φ s := by
    unfold Φ
    apply Finset.sum_lt_sum
    · intro qc _
      by_cases h : qc = (p, c)
      · subst h; exact hlt.le
      · unfold wt
        simp only [newHypothesis, Function.update_of_ne h, le_refl]
    · exact ⟨(p, c), by simp [newHypothesis, hp], hlt⟩
  simp only [newHypothesis] at this ⊢
  omega

omit [DecidableEq α] in
theorem Ψ_lt_split {s : Level3_5State α} {p : List Bool} (hp : p ∈ s.tree.leaves) {d : FreeMonoid α}
    (hsz : (s.tree.splitAt d p).leaves.length ≤ C.maxLeaves) (E : Edges α)
    (mv : List Bool × α → Bool) :
    Ψr C (newHypothesis (s.tree.splitAt d p) E mv) < Ψr C s := by
  unfold Ψr
  have hlen := DTree.length_leaves_splitAt d s.tree p hp
  have hΦ := Φ_le (newHypothesis (s.tree.splitAt d p) E mv)
  simp only [newHypothesis] at hΦ ⊢
  rw [hlen] at hΦ hsz ⊢
  have h1 : 2 * (s.tree.leaves.length + 1) * Fintype.card α + 1 ≤ X (α := α) C := by
    unfold X
    have := Nat.mul_le_mul_right (Fintype.card α) (Nat.mul_le_mul_left 2 hsz)
    omega
  have h2 : C.maxLeaves - s.tree.leaves.length = C.maxLeaves - (s.tree.leaves.length + 1) + 1 := by
    omega
  rw [h2, Nat.add_mul, one_mul]
  omega

end Potential

structure Inv [Fintype α] (s : Level3_5State α) : Prop where
  edges : EdgesOK read s.tree s.edges
  recs_lt : ∀ r, s.records.count r < C.recordsToMove
  size : s.tree.leaves.length ≤ C.maxLeaves
  reached : Reached read s.tree
  cls : s.tree ∈ classSet (s.tree.leaves.length - 2)
  two : 2 ≤ s.tree.leaves.length

section Steps

variable [Fintype α] {σ : Type*} [Fintype σ] {M : DFA α σ} {side : σ → Bool}

theorem inv_start (hm : 1 ≤ C.recordsToMove) (h2 : 2 ≤ C.maxLeaves) :
    Inv read C (initialState : Level3_5State α) where
  edges := fun _ _ _ _ _ h => by simp [initialState, newHypothesis] at h
  recs_lt := fun _ => by simp [initialState, newHypothesis]; omega
  size := by simp [initialState, newHypothesis, DTree.leaves]; omega
  reached := by
    intro p hp
    simp only [initialState, newHypothesis, DTree.leaves, List.map_cons, List.map_nil,
      List.cons_append, List.nil_append, List.mem_cons, List.not_mem_nil, or_false] at hp
    rcases hp with rfl | rfl
    · exact Or.inl rfl
    · exact Or.inr (Or.inl rfl)
  cls := by simp [initialState, newHypothesis, DTree.leaves, classSet]
  two := by simp [initialState, newHypothesis, DTree.leaves]

omit [DecidableEq α] in
theorem leaves_root : (DTree.node 1 .leaf .leaf : DTree α).leaves = [[false], [true]] := by
  simp [DTree.leaves]

theorem Ψr_start (h2 : 2 ≤ C.maxLeaves) :
    Ψr C (initialState : Level3_5State α) + 1 ≤ stretches C (Fintype.card α) := by
  have hΦ := Φ_le (initialState : Level3_5State α)
  unfold Ψr stretches
  simp only [initialState, newHypothesis] at hΦ ⊢
  rw [leaves_root] at hΦ ⊢
  simp only [List.length_cons, List.length_nil, zero_add, Nat.reduceAdd] at hΦ ⊢
  have hX : X (α := α) C = 2 * C.maxLeaves * Fintype.card α + 1 := rfl
  rw [hX]
  have hL : C.maxLeaves * (2 * C.maxLeaves * Fintype.card α + 1)
      = (C.maxLeaves - 2) * (2 * C.maxLeaves * Fintype.card α + 1)
        + 2 * (2 * C.maxLeaves * Fintype.card α + 1) := by
    rw [← Nat.add_mul, Nat.sub_add_cancel h2]
  have : 4 * Fintype.card α ≤ 2 * (2 * C.maxLeaves * Fintype.card α + 1) := by
    have := Nat.mul_le_mul_right (Fintype.card α) h2
    nlinarith
  omega

theorem finish_inl {t s' : Level3_5State α} (h : applyTests C t = .inl s') : s' = t
    ∧ runTests C t = none := by
  unfold applyTests at h
  split at h
  · simp at h
  · rename_i hl
    exact ⟨by simpa using h.symm, hl⟩

theorem countProbe_tree (s : Level3_5State α) (x : FreeMonoid α) :
    (countProbe read C s x).tree = s.tree := rfl
theorem countProbe_edges (s : Level3_5State α) (x : FreeMonoid α) :
    (countProbe read C s x).edges = s.edges := rfl
theorem countProbe_redirected (s : Level3_5State α) (x : FreeMonoid α) :
    (countProbe read C s x).redirected = s.redirected := rfl
theorem countProbe_records (s : Level3_5State α) (x : FreeMonoid α) :
    (countProbe read C s x).records = s.records := rfl
theorem countProbe_probeCount (s : Level3_5State α) (x : FreeMonoid α) :
    (countProbe read C s x).probeCount = s.probeCount + 1 := rfl

/-- The steps that keep the hypothesis: a look after counting the probe. -/
theorem step_cases {s s' : Level3_5State α} {x : FreeMonoid α} (hs : Inv read C s)
    (h : applyProbe read C s x = .inl s') :
    (∃ p c u t, probeOrAgree true read s.tree s.edges C.walkStart x = .unlearnedEdge p c u t
        ∧ s.edges (p, c) = none ∧
        s.tree.sift read u = .inl p ∧ s.tree.sift read (u * FreeMonoid.of c) = .inl t ∧
        p ∈ s.tree.leaves ∧
        s' = newHypothesis s.tree (Function.update s.edges (p, c) (some (t, u)))
          (Function.update s.redirected (p, c) false))
    ∨ (∃ p c u t t₀ w₀, probeOrAgree true read s.tree s.edges C.walkStart x = .wrongEdge p c u t ∧
        s.edges (p, c) = some (t₀, w₀) ∧ t₀ ≠ t ∧ p ∈ s.tree.leaves ∧
        s.tree.sift read u = .inl p ∧ s.tree.sift read (u * FreeMonoid.of c) = .inl t ∧
        C.recordsToMove ≤ (s.records ++ [(p, c, t)]).count (p, c, t) ∧
        redirectOrSplit read C s p c u t = .inl s')
    ∨ (s' = countProbe read C
          { s with records :=
            s.records ++ (recOf (probeOrAgree true read s.tree s.edges C.walkStart x)).toList } x
        ∧ runTests C s' = none
        ∧ ∀ p c u t, probeOrAgree true read s.tree s.edges C.walkStart x = .wrongEdge p c u t →
          ¬ C.recordsToMove ≤ (s.records ++ [(p, c, t)]).count (p, c, t)) := by
  have hOK := Ideal.probeOrAgree_ok true read s.tree s.edges C.walkStart hs.edges x
  unfold applyProbe at h
  generalize hg : probeOrAgree true read s.tree s.edges C.walkStart x = o at h hOK
  cases o with
  | unlearnedEdge p c u t =>
    obtain ⟨hp, hu, huc, hE⟩ := hOK
    exact Or.inl ⟨p, c, u, t, rfl, hE, hu, huc, hp, by simpa using h.symm⟩
  | wrongEdge p c u t =>
    obtain ⟨hp, hu, huc, t₀, w₀, hE, hne⟩ := hOK
    simp only at h
    split_ifs at h with hc
    · exact Or.inr (Or.inl ⟨p, c, u, t, t₀, w₀, rfl, hE, hne, hp, hu, huc, hc, h⟩)
    · obtain ⟨h1, h2⟩ := finish_inl C h
      refine Or.inr (Or.inr ⟨h1, h1 ▸ h2, ?_⟩)
      intro p' c' u' t' he
      cases he
      exact hc
  | agree =>
    obtain ⟨h1, h2⟩ := finish_inl C h
    exact Or.inr (Or.inr ⟨by simpa [recOf] using h1, h1 ▸ h2, fun _ _ _ _ he => by cases he⟩)
  | undecided k zs cl =>
    obtain ⟨h1, h2⟩ := finish_inl C h
    exact Or.inr (Or.inr ⟨by simpa [recOf] using h1, h1 ▸ h2, fun _ _ _ _ he => by cases he⟩)

/-- A step that keeps the hypothesis counts one more probe. -/
theorem step_same {s s' : Level3_5State α} (hs : Inv read C s) {x : FreeMonoid α}
    (h : applyProbe read C s x = .inl s') (ht : s'.tree = s.tree) (he : s'.edges = s.edges) :
    Inv read C s' ∧ Ψr C s' = Ψr C s
        ∧ s'.probeCount = s.probeCount + 1 := by
  rcases step_cases read C hs h with
    ⟨p, c, u, t, -, hE, -, -, -, rfl⟩ | ⟨p, c, u, t, t₀, w₀, -, hE, hne, hp, -, -, -, hfix⟩ |
    ⟨rfl, -, hlt⟩
  · exfalso
    have := congrFun he (p, c)
    simp only [newHypothesis, Function.update_self, hE] at this
    exact absurd this (by simp)
  · exfalso
    unfold redirectOrSplit at hfix
    rw [hE] at hfix
    simp only at hfix
    split_ifs at hfix with hmv
    · unfold splitLeaf at hfix
      simp only at hfix
      split_ifs at hfix
      simp only [Sum.inl.injEq] at hfix
      subst hfix
      have := congrArg List.length (congrArg DTree.leaves ht)
      simp only [newHypothesis] at this
      rw [DTree.length_leaves_splitAt _ _ _ hp] at this
      omega
    · simp only [Sum.inl.injEq] at hfix
      subst hfix
      have := congrFun he (p, c)
      simp only [newHypothesis, Function.update_self, hE, Option.some.injEq, Prod.mk.injEq] at this
      exact hne this.1.symm
  · refine ⟨⟨hs.edges, ?_, hs.size, hs.reached, hs.cls, hs.two⟩, rfl, rfl⟩
    intro r
    simp only [countProbe_records]
    generalize hg : probeOrAgree true read s.tree s.edges C.walkStart x = o at hlt
    cases o with
    | wrongEdge p c u t =>
      by_cases hr : r = (p, c, t)
      · subst hr; have := hlt p c u t rfl; simp only [recOf, Option.toList_some]; omega
      · simp only [recOf, Option.toList_some, List.count_append, List.count_singleton]
        have := hs.recs_lt r
        simp [Ne.symm hr, this]
    | _ => simpa [recOf] using hs.recs_lt r

/-- A split on a letter and the midfix where its targets part keeps every leaf past the first
two reached. -/
theorem split_reached {s : Level3_5State α} (hs : Inv read C s) {p t t₀ : List Bool} {c : α}
    {u w₀ : FreeMonoid α} (hp : p ∈ s.tree.leaves) (hu : s.tree.sift read u = .inl p)
    (huc : s.tree.sift read (u * FreeMonoid.of c) = .inl t) (hw₀ : s.tree.sift read w₀ = .inl p)
    (hw₀c : s.tree.sift read (w₀ * FreeMonoid.of c) = .inl t₀) (hne : t₀ ≠ t) :
    Reached read (s.tree.splitAt (FreeMonoid.of c * s.tree.midfixAt (commonPrefix t t₀)) p) := by
  have hpart := DTree.sift_part read s.tree _ _ t t₀ huc hw₀c (Ne.symm hne)
  refine Ideal.reached_splitAt read hs.reached hp hu hw₀ ?_ ?_ ?_
  · simpa [mul_assoc] using hpart.1
  · simpa [mul_assoc] using hpart.2.1
  · simpa [mul_assoc] using hpart.2.2

/-- A split that fits keeps the invariant, the tree in the class, and lowers `Ψr`. -/
theorem split_ok {s : Level3_5State α} (hs : Inv read C s) (hm : 1 ≤ C.recordsToMove)
    {p t t₀ : List Bool} {c : α} {u w₀ : FreeMonoid α} (hp : p ∈ s.tree.leaves)
    (hu : s.tree.sift read u = .inl p) (huc : s.tree.sift read (u * FreeMonoid.of c) = .inl t)
    (hw₀ : s.tree.sift read w₀ = .inl p) (hw₀c : s.tree.sift read (w₀ * FreeMonoid.of c) = .inl t₀)
    (hne : t₀ ≠ t)
    (hsz : (s.tree.splitAt (FreeMonoid.of c * s.tree.midfixAt (commonPrefix t t₀)) p).leaves.length
      ≤ C.maxLeaves) :
    let T' := s.tree.splitAt (FreeMonoid.of c * s.tree.midfixAt (commonPrefix t t₀)) p
    Inv read C (newHypothesis T' (retarget read T' p s.edges) fun _ => false)
      ∧ Ψr C (newHypothesis T' (retarget read T' p s.edges) fun _ => false) < Ψr C s := by
  intro T'
  have hreach := split_reached read C hs hp hu huc hw₀ hw₀c hne
  have hlen :=
    DTree.length_leaves_splitAt (FreeMonoid.of c * s.tree.midfixAt (commonPrefix t t₀)) s.tree p hp
  have ht := DTree.sift_inl_mem read _ _ t huc
  have ht₀ := DTree.sift_inl_mem read _ _ t₀ hw₀c
  refine ⟨⟨Ideal.edgesOK_retarget read hs.edges hp _, by simp [newHypothesis]; omega, hsz, hreach,
    ?_, by simp only [newHypothesis, T']; rw [hlen]; have := hs.two; omega⟩, ?_⟩
  · simp only [newHypothesis, T']
    rw [hlen, show s.tree.leaves.length + 1 - 2 = s.tree.leaves.length - 2 + 1 by
      have := hs.two; omega]
    exact classSet_split hs.cls hp ht ht₀
  · exact Ψ_lt_split C hp hsz _ _

/-- A step that changes the hypothesis starts a fresh stretch and lowers `Ψr`. -/
theorem step_change (hm : 1 ≤ C.recordsToMove) {s s' : Level3_5State α} (hs : Inv read C s)
    {x : FreeMonoid α} (h : applyProbe read C s x = .inl s')
    (hne : ¬ (s'.tree = s.tree ∧ s'.edges = s.edges)) :
    Inv read C s' ∧ s' = newHypothesis s'.tree s'.edges s'.redirected ∧ Ψr C s' < Ψr C s := by
  rcases step_cases read C hs h with
    ⟨p, c, u, t, -, hE, hu, huc, hp, rfl⟩ |
    ⟨p, c, u, t, t₀, w₀, -, hE, htt, hp, hu, huc, -, hfix⟩ | ⟨rfl, -, -⟩
  · refine ⟨⟨Ideal.edgesOK_upd read hs.edges hu huc, fun r => by simp [newHypothesis]; omega,
      hs.size, hs.reached, hs.cls, hs.two⟩, rfl, ?_⟩
    exact Ψ_lt_upd C hp _ _ (by simp [wt, newHypothesis, Function.update_self, hE])
  · obtain ⟨hw₀, hw₀c⟩ := hs.edges p hp c t₀ w₀ hE
    unfold redirectOrSplit at hfix
    rw [hE] at hfix
    simp only at hfix
    split_ifs at hfix with hmv
    · unfold splitLeaf at hfix
      simp only at hfix
      split_ifs at hfix with hlt
      simp only [Sum.inl.injEq] at hfix
      subst hfix
      obtain ⟨hinv, hΨ⟩ := split_ok read C hs hm hp hu huc hw₀ hw₀c htt (by omega)
      exact ⟨hinv, rfl, hΨ⟩
    · simp only [Sum.inl.injEq] at hfix
      subst hfix
      refine ⟨⟨Ideal.edgesOK_upd read hs.edges hu huc, fun r => by simp [newHypothesis]; omega,
        hs.size, hs.reached, hs.cls, hs.two⟩, rfl, ?_⟩
      exact Ψ_lt_upd C hp _ _ (by simp [wt, newHypothesis, Function.update_self, hE, hmv])
  · exact absurd ⟨rfl, rfl⟩ hne

/-- The round ends too big only at a split, of an edge redirected since it was learned, at
its `recordsToMove`-th record of a new target. -/
theorem step_tooBig {s : Level3_5State α} (hs : Inv read C s) {x : FreeMonoid α}
    (h : applyProbe read C s x = .inr .failed) :
    ∃ p c u t t₀ w₀, probeOrAgree true read s.tree s.edges C.walkStart x = .wrongEdge p c u t ∧
      s.edges (p, c) = some (t₀, w₀) ∧ t₀ ≠ t ∧ p ∈ s.tree.leaves ∧
      s.tree.sift read u = .inl p ∧ s.tree.sift read (u * FreeMonoid.of c) = .inl t ∧
      s.redirected (p, c) = true ∧ C.recordsToMove ≤ (s.records ++ [(p, c, t)]).count (p, c, t) ∧
      C.maxLeaves < (s.tree.splitAt
        (FreeMonoid.of c * s.tree.midfixAt (commonPrefix t t₀)) p).leaves.length := by
  have hOK := Ideal.probeOrAgree_ok true read s.tree s.edges C.walkStart hs.edges x
  have hlook : ∀ t : Level3_5State α, applyTests C t ≠ .inr .failed := by
    intro t ht
    unfold applyTests runTests at ht
    split_ifs at ht <;> simp_all
  unfold applyProbe at h
  generalize hg : probeOrAgree true read s.tree s.edges C.walkStart x = o at h hOK
  cases o with
  | unlearnedEdge p c u t => simp at h
  | wrongEdge p c u t =>
    obtain ⟨hp, hu, huc, t₀, w₀, hE, htt⟩ := hOK
    simp only at h
    split_ifs at h with hc
    · unfold redirectOrSplit at h
      rw [hE] at h
      simp only at h
      split_ifs at h with hmv
      · unfold splitLeaf at h
        dsimp only at h
        split_ifs at h with hlt
        exact ⟨p, c, u, t, t₀, w₀, rfl, hE, htt, hp, hu, huc, hmv, hc, hlt⟩
    · exact absurd h (hlook _)
  | agree => exact absurd h (hlook _)
  | undecided k zs cl => exact absurd h (hlook _)

theorem step_ne_tooBig (hW : NoWrong M side read) (hcap : Fintype.card σ + 2 ≤ C.maxLeaves)
    {s : Level3_5State α} (hs : Inv read C s) (x : FreeMonoid α) :
    applyProbe read C s x ≠ .inr .failed := by
  intro h
  obtain ⟨p, c, u, t, t₀, w₀, -, hE, htt, hp, hu, huc, -, -, hlt⟩ := step_tooBig read C hs h
  obtain ⟨hw₀, hw₀c⟩ := hs.edges p hp c t₀ w₀ hE
  have := leaves_le_of_reached' read hW (split_reached read C hs hp hu huc hw₀ hw₀c htt)
  omega

theorem cls_mem (hW : NoWrong M side read) {s : Level3_5State α} (hs : Inv read C s) :
    s.tree ∈ classSet (Fintype.card σ) :=
  classSet_mono' (by have := leaves_le_of_reached' read hW hs.reached; omega) hs.cls

end Steps

end Eps

end Random

end OrthoDFA
