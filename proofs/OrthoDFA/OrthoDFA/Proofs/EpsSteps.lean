import OrthoDFA.Level4Claim
import OrthoDFA.Proofs.RandomTree

/-!
# Level 4's steps

A step that keeps the hypothesis counts one more probe into the stretch; one that changes it
starts a fresh stretch and lowers `Ψr`, which counts the splits left and each edge's learning and
redirect. An edge's records are kept until a split, so an edge whose target has `recordsToMove`
records was redirected there and is fixed next by a split; no record is at a target its edge does
not point at `recordsToMove` times.
-/

namespace OrthoDFA

namespace Random

namespace Eps

open OrthoDFA.Ideal (probe probeOrAgree Outcome retarget EdgesOK Reached OutcomeOK)

variable {α : Type*} [DecidableEq α] (read : FreeMonoid α → ARU) (C : Level4Settings)

section Potential

variable [Fintype α]

/-- What an edge has left to change before the next split: learning and redirecting, or
redirecting; none once its target has `recordsToMove` records. -/
def wt (s : Level4State α) (qc : List Bool × α) : ℕ :=
  match s.edges qc with
  | none => 2
  | some (t₀, _) => if C.recordsToMove ≤ s.records.count (qc.1, qc.2, t₀) then 0 else 1

def Φ (s : Level4State α) : ℕ := ∑ qc ∈ s.tree.leaves.toFinset ×ˢ Finset.univ, wt C s qc

def X : ℕ := 2 * C.maxLeaves * Fintype.card α + 1

/-- The splits, learnings and redirects left. -/
def Ψr (s : Level4State α) : ℕ := (C.maxLeaves - s.tree.leaves.length) * X (α := α) C + Φ C s

omit [Fintype α] in
theorem wt_le (s : Level4State α) (qc : List Bool × α) : wt C s qc ≤ 2 := by
  unfold wt; split
  · exact le_rfl
  · split <;> omega

theorem Φ_le (s : Level4State α) : Φ C s ≤ 2 * s.tree.leaves.length * Fintype.card α := by
  unfold Φ
  calc ∑ qc ∈ s.tree.leaves.toFinset ×ˢ Finset.univ, wt C s qc
      ≤ (s.tree.leaves.toFinset ×ˢ (Finset.univ : Finset α)).card • 2 :=
        Finset.sum_le_card_nsmul _ _ _ fun qc _ => wt_le C s qc
    _ ≤ 2 * s.tree.leaves.length * Fintype.card α := by
      rw [Finset.card_product, Finset.card_univ, smul_eq_mul]
      have := List.toFinset_card_le s.tree.leaves
      nlinarith

/-- Lowering one edge's weight, the others kept, lowers `Ψr`. -/
theorem Ψ_lt_upd {s s' : Level4State α} {p : List Bool} {c : α} (hp : p ∈ s.tree.leaves)
    (ht : s'.tree = s.tree) (hwt : ∀ qc, qc ≠ (p, c) → wt C s' qc = wt C s qc)
    (hlt : wt C s' (p, c) < wt C s (p, c)) : Ψr C s' < Ψr C s := by
  unfold Ψr
  have : Φ C s' < Φ C s := by
    unfold Φ
    rw [ht]
    apply Finset.sum_lt_sum
    · intro qc _
      by_cases h : qc = (p, c)
      · subst h; exact hlt.le
      · exact (hwt qc h).le
    · exact ⟨(p, c), by simp [hp], hlt⟩
  rw [ht]
  omega

/-- Keeping every edge's weight keeps `Ψr`. -/
theorem Ψr_eq {s s' : Level4State α} (ht : s'.tree = s.tree) (hwt : ∀ qc, wt C s' qc = wt C s qc) :
    Ψr C s' = Ψr C s := by
  unfold Ψr Φ
  rw [ht]
  simp only [hwt]

omit [Fintype α] in
theorem wt_none {s : Level4State α} {qc : List Bool × α} (h : s.edges qc = none) :
    wt C s qc = 2 := by
  unfold wt; rw [h]

omit [Fintype α] in
theorem wt_some {s : Level4State α} {qc : List Bool × α} {t₀ : List Bool} {w : FreeMonoid α}
    (h : s.edges qc = some (t₀, w)) :
    wt C s qc = if C.recordsToMove ≤ s.records.count (qc.1, qc.2, t₀) then 0 else 1 := by
  unfold wt; rw [h]

omit [Fintype α] in
theorem wt_zero {s : Level4State α} {qc : List Bool × α} {t₀ : List Bool} {w : FreeMonoid α}
    (h : s.edges qc = some (t₀, w)) (hle : C.recordsToMove ≤ s.records.count (qc.1, qc.2, t₀)) :
    wt C s qc = 0 := by
  rw [wt_some C h]; exact if_pos hle

omit [Fintype α] in
theorem wt_one {s : Level4State α} {qc : List Bool × α} {t₀ : List Bool} {w : FreeMonoid α}
    (h : s.edges qc = some (t₀, w)) (hlt : s.records.count (qc.1, qc.2, t₀) < C.recordsToMove) :
    wt C s qc = 1 := by
  rw [wt_some C h]; exact if_neg (not_le.2 hlt)

omit [Fintype α] in
/-- An edge's weight depends on its target and the records there. -/
theorem wt_congr {s s' : Level4State α} {qc : List Bool × α} (he : s'.edges qc = s.edges qc)
    (hc : ∀ t₀ w, s.edges qc = some (t₀, w) →
      s'.records.count (qc.1, qc.2, t₀) = s.records.count (qc.1, qc.2, t₀)) :
    wt C s' qc = wt C s qc := by
  unfold wt
  rw [he]
  rcases h0 : s.edges qc with _ | ⟨t₀, w⟩
  · rfl
  · simp only [hc t₀ w h0]

theorem Ψ_lt_split {s : Level4State α} {p : List Bool} (hp : p ∈ s.tree.leaves) {d : FreeMonoid α}
    (hsz : (s.tree.splitAt d p).leaves.length ≤ C.maxLeaves) (E : Edges α)
    (rs : List (List Bool × α × List Bool)) :
    Ψr C (newHypothesis (s.tree.splitAt d p) E rs) < Ψr C s := by
  unfold Ψr
  have hlen := DTree.length_leaves_splitAt d s.tree p hp
  have hΦ := Φ_le C (newHypothesis (s.tree.splitAt d p) E rs)
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

theorem count_append_other {rs : List (List Bool × α × List Bool)} {p q : List Bool} {c d : α}
    {t t' : List Bool} (h : (q, d) ≠ (p, c)) :
    (rs ++ [(p, c, t)]).count (q, d, t') = rs.count (q, d, t') := by
  rw [List.count_append, List.count_singleton]
  have : (p, c, t) ≠ (q, d, t') := fun he => h (by simp only [Prod.mk.injEq] at he ⊢; tauto)
  simp [this]

theorem count_append_target {rs : List (List Bool × α × List Bool)} {p : List Bool} {c : α}
    {t t' : List Bool} (h : t' ≠ t) :
    (rs ++ [(p, c, t)]).count (p, c, t') = rs.count (p, c, t') := by
  rw [List.count_append, List.count_singleton]
  have : (p, c, t) ≠ (p, c, t') := fun he =>
    h (by simp only [Prod.mk.injEq] at he; exact he.2.2.symm)
  simp [this]

structure Inv [Fintype α] (s : Level4State α) : Prop where
  edges : EdgesOK read s.tree s.edges
  recs_lt : ∀ p c t, (s.edges (p, c)).map Prod.fst ≠ some t →
    s.records.count (p, c, t) < C.recordsToMove
  size : s.tree.leaves.length ≤ C.maxLeaves
  reached : Reached read s.tree
  cls : s.tree ∈ classSet (s.tree.leaves.length - 2)
  two : 2 ≤ s.tree.leaves.length

section Steps

variable [Fintype α] {σ : Type*} [Fintype σ] {M : DFA α σ} {side : σ → Bool}

theorem inv_start (hm : 1 ≤ C.recordsToMove) (h2 : 2 ≤ C.maxLeaves) :
    Inv read C (initialState : Level4State α) where
  edges := fun _ _ _ _ _ h => by simp [initialState, newHypothesis] at h
  recs_lt := fun _ _ _ _ => by simp [initialState, newHypothesis]; omega
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
    Ψr C (initialState : Level4State α) + 1 ≤ stretches C (Fintype.card α) := by
  have hΦ := Φ_le C (initialState : Level4State α)
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

theorem finish_inl {t s' : Level4State α} (h : applyTests C t = .inl s') : s' = t
    ∧ runTests C t = none := by
  unfold applyTests at h
  split at h
  · simp at h
  · rename_i hl
    exact ⟨by simpa using h.symm, hl⟩

theorem countProbe_tree (s : Level4State α) (x : FreeMonoid α) :
    (countProbe read C s x).tree = s.tree := rfl
theorem countProbe_edges (s : Level4State α) (x : FreeMonoid α) :
    (countProbe read C s x).edges = s.edges := rfl
theorem countProbe_records (s : Level4State α) (x : FreeMonoid α) :
    (countProbe read C s x).records = s.records := rfl
theorem countProbe_probeCount (s : Level4State α) (x : FreeMonoid α) :
    (countProbe read C s x).probeCount = s.probeCount + 1 := rfl

/-- The steps that keep the hypothesis: a look after counting the probe. -/
theorem step_cases {s s' : Level4State α} {x : FreeMonoid α} (hs : Inv read C s)
    (h : applyProbe read C s x = .inl s') :
    (∃ p c u t, probeOrAgree false read s.tree s.edges C.walkStart x = .unlearnedEdge p c u t
        ∧ s.edges (p, c) = none ∧
        s.tree.sift read u = .inl p ∧ s.tree.sift read (u * FreeMonoid.of c) = .inl t ∧
        p ∈ s.tree.leaves ∧
        s' = newHypothesis s.tree (Function.update s.edges (p, c) (some (t, u))) s.records)
    ∨ (∃ p c u t t₀ w₀, probeOrAgree false read s.tree s.edges C.walkStart x = .wrongEdge p c u t ∧
        s.edges (p, c) = some (t₀, w₀) ∧ t₀ ≠ t ∧ p ∈ s.tree.leaves ∧
        s.tree.sift read u = .inl p ∧ s.tree.sift read (u * FreeMonoid.of c) = .inl t ∧
        C.recordsToMove ≤ (s.records ++ [(p, c, t)]).count (p, c, t) ∧
        redirectOrSplit read C s (s.records ++ [(p, c, t)]) p c u t = .inl s')
    ∨ (s' = countProbe read C
          { s with records :=
            s.records ++ (recOf (probeOrAgree false read s.tree s.edges C.walkStart x)).toList } x
        ∧ runTests C s' = none
        ∧ ∀ p c u t, probeOrAgree false read s.tree s.edges C.walkStart x = .wrongEdge p c u t →
          ¬ C.recordsToMove ≤ (s.records ++ [(p, c, t)]).count (p, c, t)) := by
  have hOK := Ideal.probeOrAgree_ok false read s.tree s.edges C.walkStart hs.edges x
  unfold applyProbe at h
  generalize hg : probeOrAgree false read s.tree s.edges C.walkStart x = o at h hOK
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
  | agree | undecidedStart _ | undecided _ _ =>
    obtain ⟨h1, h2⟩ := finish_inl C h
    exact Or.inr (Or.inr ⟨by simpa [recOf] using h1, h1 ▸ h2, fun _ _ _ _ he => by cases he⟩)

/-- A probe's record is at an edge out of a leaf, at a target the edge does not point at. -/
theorem recOf_target {s : Level4State α} (hs : Inv read C s) {x : FreeMonoid α}
    {p : List Bool} {c : α} {t : List Bool}
    (h : recOf (probeOrAgree false read s.tree s.edges C.walkStart x) = some (p, c, t)) :
    p ∈ s.tree.leaves ∧ (s.edges (p, c)).map Prod.fst ≠ some t := by
  have hOK := Ideal.probeOrAgree_ok false read s.tree s.edges C.walkStart hs.edges x
  generalize probeOrAgree false read s.tree s.edges C.walkStart x = o at h hOK
  cases o with
  | wrongEdge p' c' u t' =>
    simp only [recOf, Option.some.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl, rfl⟩ := h
    obtain ⟨hp, -, -, t₀, w₀, hE, hne⟩ := hOK
    exact ⟨hp, by simp [hE, hne]⟩
  | _ => simp [recOf] at h

/-- Counting a probe that keeps the hypothesis keeps each edge's weight. -/
theorem wt_counted {s : Level4State α} (hs : Inv read C s) (x : FreeMonoid α) (qc : List Bool × α) :
    wt C (countProbe read C { s with records :=
      s.records ++ (recOf (probeOrAgree false read s.tree s.edges C.walkStart x)).toList } x) qc
      = wt C s qc := by
  refine wt_congr C rfl fun t₀ w h0 => ?_
  rw [countProbe_records]
  rcases hr : recOf (probeOrAgree false read s.tree s.edges C.walkStart x) with _ | ⟨p, c, t⟩
  · simp
  · obtain ⟨-, hne⟩ := recOf_target read C hs hr
    simp only [Option.toList_some]
    by_cases hqc : qc = (p, c)
    · subst hqc
      exact count_append_target (fun h' => hne (by simp [h0, h']))
    · exact count_append_other (by simpa using hqc)

/-- A step that keeps the hypothesis counts one more probe. -/
theorem step_same {s s' : Level4State α} (hs : Inv read C s) {x : FreeMonoid α}
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
  · refine ⟨⟨hs.edges, ?_, hs.size, hs.reached, hs.cls, hs.two⟩,
      Ψr_eq C rfl (wt_counted read C hs x), rfl⟩
    intro p c t hnt
    simp only [countProbe_records]
    rcases hr : recOf (probeOrAgree false read s.tree s.edges C.walkStart x) with _ | ⟨p', c', t'⟩
    · simpa using hs.recs_lt p c t hnt
    · simp only [Option.toList_some]
      by_cases heq : (p, c, t) = (p', c', t')
      · simp only [Prod.mk.injEq] at heq
        obtain ⟨rfl, rfl, rfl⟩ := heq
        generalize hg : probeOrAgree false read s.tree s.edges C.walkStart x = o at hr hlt
        cases o with
        | wrongEdge p₁ c₁ u₁ t₁ =>
          simp only [recOf, Option.some.injEq, Prod.mk.injEq] at hr
          obtain ⟨rfl, rfl, rfl⟩ := hr
          exact not_le.1 (hlt _ _ u₁ _ rfl)
        | _ => simp [recOf] at hr
      · have h1 : [(p', c', t')].count (p, c, t) = 0 := List.count_eq_zero.2 (by simpa using heq)
        rw [List.count_append, h1, add_zero]
        exact hs.recs_lt p c t hnt

/-- A split on a letter and the midfix where its targets part keeps every leaf past the first
two reached. -/
theorem split_reached {s : Level4State α} (hs : Inv read C s) {p t t₀ : List Bool} {c : α}
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
theorem split_ok {s : Level4State α} (hs : Inv read C s) (hm : 1 ≤ C.recordsToMove)
    {p t t₀ : List Bool} {c : α} {u w₀ : FreeMonoid α} (hp : p ∈ s.tree.leaves)
    (hu : s.tree.sift read u = .inl p) (huc : s.tree.sift read (u * FreeMonoid.of c) = .inl t)
    (hw₀ : s.tree.sift read w₀ = .inl p) (hw₀c : s.tree.sift read (w₀ * FreeMonoid.of c) = .inl t₀)
    (hne : t₀ ≠ t)
    (hsz : (s.tree.splitAt (FreeMonoid.of c * s.tree.midfixAt (commonPrefix t t₀)) p).leaves.length
      ≤ C.maxLeaves) :
    let T' := s.tree.splitAt (FreeMonoid.of c * s.tree.midfixAt (commonPrefix t t₀)) p
    Inv read C (newHypothesis T' (retarget read T' p s.edges) [])
      ∧ Ψr C (newHypothesis T' (retarget read T' p s.edges) []) < Ψr C s := by
  intro T'
  have hreach := split_reached read C hs hp hu huc hw₀ hw₀c hne
  have hlen :=
    DTree.length_leaves_splitAt (FreeMonoid.of c * s.tree.midfixAt (commonPrefix t t₀)) s.tree p hp
  have ht := DTree.sift_inl_mem read _ _ t huc
  have ht₀ := DTree.sift_inl_mem read _ _ t₀ hw₀c
  refine ⟨⟨Ideal.edgesOK_retarget read hs.edges hp _, fun _ _ _ _ => by simp [newHypothesis]; omega,
    hsz, hreach, ?_, by simp only [newHypothesis, T']; rw [hlen]; have := hs.two; omega⟩, ?_⟩
  · simp only [newHypothesis, T']
    rw [hlen, show s.tree.leaves.length + 1 - 2 = s.tree.leaves.length - 2 + 1 by
      have := hs.two; omega]
    exact classSet_split hs.cls hp ht ht₀
  · exact Ψ_lt_split C hp hsz _ _

/-- A step that changes the hypothesis starts a fresh stretch, keeping the records unless it
splits, and lowers `Ψr`. -/
theorem step_change (hm : 1 ≤ C.recordsToMove) {s s' : Level4State α} (hs : Inv read C s)
    {x : FreeMonoid α} (h : applyProbe read C s x = .inl s')
    (hne : ¬ (s'.tree = s.tree ∧ s'.edges = s.edges)) :
    Inv read C s' ∧ s' = newHypothesis s'.tree s'.edges s'.records ∧ Ψr C s' < Ψr C s := by
  rcases step_cases read C hs h with
    ⟨p, c, u, t, -, hE, hu, huc, hp, rfl⟩ |
    ⟨p, c, u, t, t₀, w₀, -, hE, htt, hp, hu, huc, hc, hfix⟩ | ⟨rfl, -, -⟩
  · refine ⟨⟨Ideal.edgesOK_upd read hs.edges hu huc, fun p' c' t' hnt => ?_,
      hs.size, hs.reached, hs.cls, hs.two⟩, rfl, ?_⟩
    · simp only [newHypothesis]
      by_cases hpc : (p', c') = (p, c)
      · simp only [Prod.mk.injEq] at hpc
        obtain ⟨rfl, rfl⟩ := hpc
        exact hs.recs_lt p' c' t' (by simp [hE])
      · simp only [newHypothesis, Function.update_of_ne hpc] at hnt
        exact hs.recs_lt p' c' t' hnt
    · refine Ψ_lt_upd C (p := p) (c := c) hp rfl (fun qc hqc => ?_) ?_
      · exact wt_congr C (by simp [newHypothesis, Function.update_of_ne hqc]) fun _ _ _ => rfl
      · have := hs.recs_lt p c t (by simp [hE])
        rw [wt_none C hE, wt_some C (t₀ := t) (w := u) (by simp [newHypothesis])]
        split <;> omega
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
      refine ⟨⟨Ideal.edgesOK_upd read hs.edges hu huc, fun p' c' t' hnt => ?_,
        hs.size, hs.reached, hs.cls, hs.two⟩, rfl, ?_⟩
      · simp only [newHypothesis] at hnt ⊢
        by_cases hpc : (p', c') = (p, c)
        · simp only [Prod.mk.injEq] at hpc
          obtain ⟨rfl, rfl⟩ := hpc
          simp only [Function.update_self, Option.map_some, ne_eq, Option.some.injEq] at hnt
          by_cases ht₀ : t' = t₀
          · subst ht₀; omega
          · rw [count_append_target (Ne.symm hnt)]
            exact hs.recs_lt p' c' t' (by simp [hE, Ne.symm ht₀])
        · rw [Function.update_of_ne hpc] at hnt
          rw [count_append_other (q := p') (d := c') hpc]
          exact hs.recs_lt p' c' t' hnt
      · refine Ψ_lt_upd C (p := p) (c := c) hp rfl (fun qc hqc => ?_) ?_
        · exact wt_congr C (by simp [newHypothesis, Function.update_of_ne hqc]) fun _ _ _ =>
            count_append_other (by simpa using hqc)
        · have := count_append_target (rs := s.records) (p := p) (c := c) htt
          rw [wt_one C hE (show s.records.count (p, c, t₀) < C.recordsToMove by omega),
            wt_zero C (t₀ := t) (w := u) (by simp [newHypothesis]) hc]
          omega
  · exact absurd ⟨rfl, rfl⟩ hne

/-- The round ends too big only at a split, of an edge whose target has `recordsToMove` records,
at its `recordsToMove`-th record of a new target. -/
theorem step_tooBig {s : Level4State α} (hs : Inv read C s) {x : FreeMonoid α}
    (h : applyProbe read C s x = .inr .failed) :
    ∃ p c u t t₀ w₀, probeOrAgree false read s.tree s.edges C.walkStart x = .wrongEdge p c u t ∧
      s.edges (p, c) = some (t₀, w₀) ∧ t₀ ≠ t ∧ p ∈ s.tree.leaves ∧
      s.tree.sift read u = .inl p ∧ s.tree.sift read (u * FreeMonoid.of c) = .inl t ∧
      C.recordsToMove ≤ s.records.count (p, c, t₀) ∧
      C.recordsToMove ≤ (s.records ++ [(p, c, t)]).count (p, c, t) ∧
      C.maxLeaves < (s.tree.splitAt
        (FreeMonoid.of c * s.tree.midfixAt (commonPrefix t t₀)) p).leaves.length := by
  have hOK := Ideal.probeOrAgree_ok false read s.tree s.edges C.walkStart hs.edges x
  have hlook : ∀ t : Level4State α, applyTests C t ≠ .inr .failed := by
    intro t ht
    unfold applyTests runTests at ht
    split_ifs at ht <;> simp_all
  unfold applyProbe at h
  generalize hg : probeOrAgree false read s.tree s.edges C.walkStart x = o at h hOK
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
        rw [count_append_target htt] at hmv
        exact ⟨p, c, u, t, t₀, w₀, rfl, hE, htt, hp, hu, huc, hmv, hc, hlt⟩
    · exact absurd h (hlook _)
  | agree | undecidedStart _ | undecided _ _ => exact absurd h (hlook _)

theorem step_ne_tooBig (hW : NoWrong M side read) (hcap : Fintype.card σ + 2 ≤ C.maxLeaves)
    {s : Level4State α} (hs : Inv read C s) (x : FreeMonoid α) :
    applyProbe read C s x ≠ .inr .failed := by
  intro h
  obtain ⟨p, c, u, t, t₀, w₀, -, hE, htt, hp, hu, huc, -, -, hlt⟩ := step_tooBig read C hs h
  obtain ⟨hw₀, hw₀c⟩ := hs.edges p hp c t₀ w₀ hE
  have := leaves_le_of_reached' read hW (split_reached read C hs hp hu huc hw₀ hw₀c htt)
  omega

theorem cls_mem (hW : NoWrong M side read) {s : Level4State α} (hs : Inv read C s) :
    s.tree ∈ classSet (Fintype.card σ) :=
  classSet_mono' (by have := leaves_le_of_reached' read hW hs.reached; omega) hs.cls

end Steps

end Eps

end Random

end OrthoDFA
