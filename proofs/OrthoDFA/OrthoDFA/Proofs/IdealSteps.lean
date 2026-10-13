import OrthoDFA.Proofs.IdealProbe

/-!
# A round's steps under ideal reads

The invariant `Inv`: every learned edge out of a leaf is witnessed, every record lies between
leaves, and the counts stay below the thresholds that would end the stretch. `Reached`: every leaf
past the first two is reached by some string.

A split's two witnesses read decidedly apart at its midfix, so both new leaves are reached, and
the reached leaves are at most `|Q|`. Each step that does not end the round lowers `Θ`: a change of
the hypothesis lowers `Ψ`, which counts the splits left and each edge's learnings and redirects
left, and resets the stretch; within a stretch, each probe raises the records and undecided
strings, which stay below `R`, or the clean run, which stays below `cleanProbesToEnd`.
-/

namespace OrthoDFA

namespace Ideal

open Finset

variable {α : Type*} (read : FreeMonoid α → ARU) (C : Level2Settings)

/-- Every leaf past the first two is reached by some string. -/
def Reached (T : DTree α) : Prop :=
  ∀ p ∈ T.leaves, p = [false] ∨ p = [true] ∨ ∃ w, T.sift read w = .inl p

/-- What the end of a round promises. -/
def EndOK : RoundEnd (UndecidedSite α) → Prop
  | .consistent => True
  | .harvest _ => True
  | .failed => False

/-! ## The tree's size -/

theorem leaves_le_of_reached {σ : Type*} [Fintype σ] {M : DFA α σ} {side : σ → Bool}
    (hI : IdealReads read M side) {T : DTree α} (h : Reached read T) :
    T.leaves.length ≤ Fintype.card σ + 2 := by
  classical
  rw [← List.toFinset_card_of_nodup (DTree.leaves_nodup T)]
  set A := T.leaves.toFinset.filter fun p => ∃ w, T.sift read w = .inl p
  have hsub : T.leaves.toFinset ⊆ A ∪ {[false], [true]} := by
    intro p hp
    rcases h p (List.mem_toFinset.1 hp) with h1 | h1 | h1
    · simp [h1]
    · simp [h1]
    · exact mem_union_left _ (mem_filter.2 ⟨hp, h1⟩)
  have hA : A.card ≤ Fintype.card σ := by
    let f : List Bool → σ := fun p =>
      if hp : ∃ w, T.sift read w = .inl p then M.eval hp.choose.toList else M.eval []
    refine card_le_card_of_injOn f (fun _ _ => mem_univ _) ?_
    intro p hp p' hp' hf
    have h1 := (mem_filter.1 hp).2
    have h2 := (mem_filter.1 hp').2
    simp only [f, h1, h2, dite_true] at hf
    have := DTree.sift_congr read _ _ (read_mul_congr read hI hf) T p h1.choose_spec
    rw [h2.choose_spec] at this
    exact (Sum.inl.inj this).symm
  calc T.leaves.toFinset.card ≤ (A ∪ {[false], [true]}).card := card_le_card hsub
    _ ≤ A.card + ({[false], [true]} : Finset (List Bool)).card := card_union_le _ _
    _ ≤ Fintype.card σ + 2 := by
      have : ({[false], [true]} : Finset (List Bool)).card ≤ 2 := card_le_two
      omega

/-! ## Splits -/

theorem sift_splitAt_ne {T : DTree α} {d z : FreeMonoid α} {p q : List Bool}
    (h : T.sift read z = .inl q) (hne : q ≠ p) : (T.splitAt d p).sift read z = .inl q := by
  rw [DTree.sift_splitAt, h]; simp [hne]

theorem sift_splitAt_eq {T : DTree α} {d z : FreeMonoid α} {p : List Bool}
    (h : T.sift read z = .inl p) :
    (T.splitAt d p).sift read z = ((DTree.node d .leaf .leaf).sift read z).map (p ++ ·) id := by
  rw [DTree.sift_splitAt, h]; simp

/-- A split whose leaf holds two strings reading decidedly apart at its midfix keeps every leaf
past the first two reached. -/
theorem reached_splitAt {T : DTree α} (hr : Reached read T) {p : List Bool} (hp : p ∈ T.leaves)
    {d w₁ w₂ : FreeMonoid α} (h₁ : T.sift read w₁ = .inl p) (h₂ : T.sift read w₂ = .inl p)
    (hd₁ : read (w₁ * d) ≠ .undecided) (hd₂ : read (w₂ * d) ≠ .undecided)
    (hd : read (w₁ * d) ≠ read (w₂ * d)) : Reached read (T.splitAt d p) := by
  have hchild : ∀ b, ∃ w, (T.splitAt d p).sift read w = .inl (p ++ [b]) := by
    have key : ∀ w, T.sift read w = .inl p → read (w * d) ≠ .undecided →
        (T.splitAt d p).sift read w = .inl (p ++ [decide (read (w * d) = .accept)]) := by
      intro w hw hu
      rw [sift_splitAt_eq read hw]
      cases hrd : read (w * d) <;> simp_all [DTree.sift]
    intro b
    cases hb : decide (read (w₁ * d) = .accept) with
    | true =>
      cases b
      · refine ⟨w₂, ?_⟩
        rw [key w₂ h₂ hd₂]
        have : read (w₁ * d) = .accept := by simpa using hb
        cases hr2 : read (w₂ * d) <;> simp_all
      · exact ⟨w₁, by rw [key w₁ h₁ hd₁, hb]⟩
    | false =>
      cases b
      · exact ⟨w₁, by rw [key w₁ h₁ hd₁, hb]⟩
      · refine ⟨w₂, ?_⟩
        rw [key w₂ h₂ hd₂]
        have : read (w₁ * d) ≠ .accept := by simpa using hb
        cases hr1 : read (w₁ * d) <;> cases hr2 : read (w₂ * d) <;> simp_all
  intro q hq
  rcases (DTree.mem_leaves_splitAt d T p hp q).1 hq with ⟨hq, hne⟩ | rfl | rfl
  · rcases hr q hq with h | h | ⟨w, hw⟩
    · exact Or.inl h
    · exact Or.inr (Or.inl h)
    · exact Or.inr (Or.inr ⟨w, sift_splitAt_ne read hw hne⟩)
  · exact Or.inr (Or.inr (hchild false))
  · exact Or.inr (Or.inr (hchild true))

theorem prefix_of_mem_leaves_splitAt {T : DTree α} {p q : List Bool} {d : FreeMonoid α}
    (hp : p ∈ T.leaves) (hq : q ∈ (T.splitAt d p).leaves) (hpq : p.isPrefixOf q = false) :
    q ∈ T.leaves ∧ q ≠ p := by
  rcases (DTree.mem_leaves_splitAt d T p hp q).1 hq with h | rfl | rfl
  · exact h
  · rw [List.isPrefixOf_iff_prefix.2 (List.prefix_append _ _)] at hpq; simp at hpq
  · rw [List.isPrefixOf_iff_prefix.2 (List.prefix_append _ _)] at hpq; simp at hpq

theorem edgesOK_retarget {T : DTree α} {E : Edges α} (hE : EdgesOK read T E) {p : List Bool}
    (hp : p ∈ T.leaves) (d : FreeMonoid α) :
    EdgesOK read (T.splitAt d p) (retarget read (T.splitAt d p) p E) := by
  intro q hq c t w h
  unfold retarget at h
  cases hpq : p.isPrefixOf q
  · obtain ⟨hq', hne⟩ := prefix_of_mem_leaves_splitAt hp hq hpq
    simp only [hpq, Bool.false_eq_true, if_false] at h
    rcases hEq : E (q, c) with _ | ⟨t₁, w₁⟩ <;> rw [hEq] at h
    · simp at h
    · obtain ⟨hw, hwc⟩ := hE q hq' c t₁ w₁ hEq
      simp only at h
      split at h
      · rename_i ht₁
        rcases hs : (T.splitAt d p).sift read (w₁ * FreeMonoid.of c) with t' | z <;>
          rw [hs] at h <;> simp only [Sum.elim_inl, Sum.elim_inr, Option.some.injEq,
            Prod.mk.injEq, reduceCtorEq] at h
        obtain ⟨rfl, rfl⟩ := h
        exact ⟨sift_splitAt_ne read hw hne, hs⟩
      · rename_i ht₁
        simp only [Option.some.injEq, Prod.mk.injEq] at h
        obtain ⟨rfl, rfl⟩ := h
        exact ⟨sift_splitAt_ne read hw hne, sift_splitAt_ne read hwc ht₁⟩
  · simp [hpq] at h

variable [DecidableEq α]

structure Inv (s : Level2State α) : Prop where
  edges : EdgesOK read s.tree s.edges
  recs_mem : ∀ r ∈ s.records, r.1 ∈ s.tree.leaves ∧ r.2.2 ∈ s.tree.leaves
  recs_lt : ∀ r, s.records.count r < C.recordsToMove
  und_lt : ∀ k, (s.undecided k).length < C.harvestSize
  clean_lt : s.cleanRun < C.cleanProbesToEnd
  size : s.tree.leaves.length ≤ C.maxLeaves

theorem edgesOK_upd {T : DTree α} {E : Edges α} (hE : EdgesOK read T E) {p t : List Bool} {c : α}
    {u : FreeMonoid α} (hu : T.sift read u = .inl p)
    (huc : T.sift read (u * FreeMonoid.of c) = .inl t) :
    EdgesOK read T (Function.update E (p, c) (some (t, u))) := by
  intro q hq c' t' w h
  by_cases hqc : (q, c') = (p, c)
  · simp only [Prod.mk.injEq] at hqc
    obtain ⟨rfl, rfl⟩ := hqc
    rw [Function.update_self] at h
    simp only [Option.some.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
    exact ⟨hu, huc⟩
  · rw [Function.update_of_ne hqc] at h
    exact hE q hq c' t' w h

/-! ## The potential -/

section Potential

variable [Fintype α]

/-- What an edge has left to change before the next split: learning and redirecting, or one. -/
def wt (s : Level2State α) (qc : List Bool × α) : ℕ :=
  match s.edges qc with
  | none => 2
  | some _ => if s.redirected qc then 0 else 1

def Φ (s : Level2State α) : ℕ := ∑ qc ∈ s.tree.leaves.toFinset ×ˢ univ, wt s qc

def X : ℕ := 2 * C.maxLeaves * Fintype.card α + 1

def Ψ (s : Level2State α) : ℕ := (C.maxLeaves - s.tree.leaves.length) * X (α := α) C + Φ s

def R : ℕ := (C.recordsToMove - 1) * C.maxLeaves ^ 2 * Fintype.card α
  + (C.harvestSize - 1) * (C.maxLeaves * Fintype.card α + 2)

def classes (T : DTree α) : Finset (UndecidedSite α) :=
  insert .start (insert .stop ((T.leaves.toFinset ×ˢ univ).image fun qc => .edge qc.1 qc.2))

def Dc (s : Level2State α) : ℕ := s.records.length + ∑ k ∈ classes s.tree, (s.undecided k).length

def ρ (s : Level2State α) : ℕ :=
  (R (α := α) C - Dc s) * C.cleanProbesToEnd + (C.cleanProbesToEnd - 1 - s.cleanRun)

def B : ℕ := (R (α := α) C + 1) * C.cleanProbesToEnd

def Θ (s : Level2State α) : ℕ := Ψ C s * B (α := α) C + ρ C s

omit [DecidableEq α] [Fintype α] in
theorem wt_le (s : Level2State α) (qc : List Bool × α) : wt s qc ≤ 2 := by
  unfold wt; split
  · exact le_rfl
  · split <;> omega

omit [DecidableEq α] in
theorem Φ_le (s : Level2State α) : Φ s ≤ 2 * s.tree.leaves.length * Fintype.card α := by
  unfold Φ
  calc ∑ qc ∈ s.tree.leaves.toFinset ×ˢ univ, wt s qc
      ≤ (s.tree.leaves.toFinset ×ˢ (univ : Finset α)).card • 2 :=
        sum_le_card_nsmul _ _ _ fun qc _ => wt_le s qc
    _ ≤ 2 * s.tree.leaves.length * Fintype.card α := by
      rw [card_product, card_univ, smul_eq_mul]
      have := List.toFinset_card_le s.tree.leaves
      nlinarith

theorem mem_classes {T : DTree α} {k : UndecidedSite α} (h : SiteOK T k) : k ∈ classes T := by
  unfold classes
  cases k with
  | start => simp
  | stop => simp
  | edge q c =>
    simp only [mem_insert, reduceCtorEq, mem_image, mem_product, List.mem_toFinset, mem_univ,
      and_true, false_or]
    exact ⟨(q, c), h, rfl⟩

theorem card_classes_le (T : DTree α) :
    (classes T).card ≤ T.leaves.length * Fintype.card α + 2 := by
  unfold classes
  calc _ ≤ (insert UndecidedSite.stop ((T.leaves.toFinset ×ˢ univ).image
        fun qc : List Bool × α => UndecidedSite.edge qc.1 qc.2)).card + 1 := card_insert_le _ _
    _ ≤ ((T.leaves.toFinset ×ˢ univ).image
          fun qc : List Bool × α => UndecidedSite.edge qc.1 qc.2).card
        + 1 + 1 := by gcongr; exact card_insert_le _ _
    _ ≤ (T.leaves.toFinset ×ˢ (univ : Finset α)).card + 2 := by
      have := card_image_le (s := T.leaves.toFinset ×ˢ (univ : Finset α))
        (f := fun qc : List Bool × α => UndecidedSite.edge qc.1 qc.2)
      omega
    _ ≤ T.leaves.length * Fintype.card α + 2 := by
      rw [card_product, card_univ]
      have := List.toFinset_card_le T.leaves
      have := Nat.mul_le_mul_right (Fintype.card α) this
      omega

theorem length_eq_sum_count {β : Type*} [BEq β] [LawfulBEq β] (K : Finset β) :
    ∀ l : List β, (∀ x ∈ l, x ∈ K) → l.length = ∑ x ∈ K, l.count x
  | [], _ => by simp
  | a :: l, hK => by
    classical
    have ha : a ∈ K := hK a (by simp)
    have ih := length_eq_sum_count K l fun x hx => hK x (by simp [hx])
    simp only [List.count_cons, List.length_cons, sum_add_distrib, beq_iff_eq, ← ih]
    simp [ha]

theorem length_le_of_count {β : Type*} [BEq β] [LawfulBEq β] (l : List β) (K : Finset β)
    (c : ℕ) (hK : ∀ x ∈ l, x ∈ K) (hc : ∀ x, l.count x ≤ c) : l.length ≤ K.card * c := by
  rw [length_eq_sum_count K l hK, ← smul_eq_mul]
  exact sum_le_card_nsmul _ _ _ fun x _ => hc x

theorem Dc_le {s : Level2State α} (hs : Inv read C s) : Dc s ≤ R (α := α) C := by
  unfold Dc R
  have hL := hs.size
  have h1 : s.records.length
      ≤ (s.tree.leaves.toFinset ×ˢ ((univ : Finset α) ×ˢ s.tree.leaves.toFinset)).card
        * (C.recordsToMove - 1) := by
    refine length_le_of_count _ _ _ (fun r hr => ?_) (fun r => by have := hs.recs_lt r; omega)
    obtain ⟨h1, h2⟩ := hs.recs_mem r hr
    simp [mem_product, h1, h2]
  have h2 : ∑ k ∈ classes s.tree, (s.undecided k).length
      ≤ (classes s.tree).card • (C.harvestSize - 1) :=
    sum_le_card_nsmul _ _ _ fun k _ => by have := hs.und_lt k; omega
  rw [smul_eq_mul] at h2
  have h3 := card_classes_le s.tree
  rw [card_product, card_product, card_univ] at h1
  have h4 := List.toFinset_card_le s.tree.leaves
  have h5 : s.tree.leaves.toFinset.card ≤ C.maxLeaves := h4.trans hL
  have h6 : s.tree.leaves.toFinset.card * (Fintype.card α * s.tree.leaves.toFinset.card)
      ≤ C.maxLeaves ^ 2 * Fintype.card α := by
    calc _ ≤ C.maxLeaves * (Fintype.card α * C.maxLeaves) := by gcongr
      _ = C.maxLeaves ^ 2 * Fintype.card α := by ring
  have h7 : (classes s.tree).card ≤ C.maxLeaves * Fintype.card α + 2 :=
    h3.trans (by have := Nat.mul_le_mul_right (Fintype.card α) hL; omega)
  have h8 := Nat.mul_le_mul_right (C.recordsToMove - 1) h6
  have h9 := Nat.mul_le_mul_right (C.harvestSize - 1) h7
  nlinarith

/-- Within a stretch: the hypothesis unchanged, and the records and undecided strings up, or the
clean run up by one. -/
theorem Θ_lt_same {s s' : Level2State α} (hs : Inv read C s) (hs' : Inv read C s')
    (ht : s'.tree = s.tree) (he : s'.edges = s.edges) (hm : s'.redirected = s.redirected)
    (hD : Dc s ≤ Dc s')
    (hcl : s'.cleanRun = s.cleanRun + 1 ∨ (s'.cleanRun = 0 ∧ Dc s + 1 ≤ Dc s')) :
    Θ C s' < Θ C s := by
  have hΨ : Ψ C s' = Ψ C s := by
    unfold Ψ Φ wt; rw [ht, he, hm]
  unfold Θ ρ
  rw [hΨ]
  have hD' := Dc_le read C hs'
  have hc := hs.clean_lt
  have hc' := hs'.clean_lt
  rcases hcl with hcl | ⟨hcl, hD1⟩
  · have := Nat.mul_le_mul_right C.cleanProbesToEnd
      (show R (α := α) C - Dc s' ≤ R (α := α) C - Dc s by omega)
    omega
  · have := Nat.mul_le_mul_right C.cleanProbesToEnd
      (show R (α := α) C - Dc s' + 1 ≤ R (α := α) C - Dc s by omega)
    rw [Nat.add_mul, one_mul] at this
    omega

theorem Θ_lt_change {s : Level2State α} (hn : 1 ≤ C.cleanProbesToEnd) (T : DTree α) (E : Edges α)
    (mv : List Bool × α → Bool) (hΨ : Ψ C (newHypothesis T E mv) < Ψ C s) :
    Θ C (newHypothesis T E mv) < Θ C s := by
  unfold Θ
  have hρ : ρ C (newHypothesis T E mv) + 1 ≤ B (α := α) C := by
    unfold ρ B Dc newHypothesis
    simp only [List.length_nil, List.length_nil, sum_const_zero, add_zero, Nat.sub_zero]
    rw [Nat.add_mul, one_mul]
    omega
  have := Nat.mul_le_mul_right (B (α := α) C) (show Ψ C (newHypothesis T E mv) + 1 ≤ Ψ C s by omega)
  rw [Nat.add_mul, one_mul] at this
  omega

/-- Changing one edge of a leaf to a lower weight lowers `Ψ`. -/
theorem Ψ_lt_upd {s : Level2State α} {p : List Bool} {c : α} (hp : p ∈ s.tree.leaves)
    (v : Option (List Bool × FreeMonoid α)) (b : Bool)
    (hlt : wt (newHypothesis s.tree (Function.update s.edges (p, c) v)
      (Function.update s.redirected (p, c) b)) (p, c) < wt s (p, c)) :
    Ψ C (newHypothesis s.tree (Function.update s.edges (p, c) v)
      (Function.update s.redirected (p, c) b)) < Ψ C s := by
  unfold Ψ
  have : Φ (newHypothesis s.tree (Function.update s.edges (p, c) v)
      (Function.update s.redirected (p, c) b)) < Φ s := by
    unfold Φ
    apply sum_lt_sum
    · intro qc _
      by_cases h : qc = (p, c)
      · subst h; exact hlt.le
      · unfold wt
        simp only [newHypothesis, Function.update_of_ne h, le_refl]
    · exact ⟨(p, c), by simp [newHypothesis, hp], hlt⟩
  simp only [newHypothesis] at this ⊢
  omega

omit [DecidableEq α] in
theorem Ψ_lt_split {s : Level2State α} {p : List Bool} (hp : p ∈ s.tree.leaves) {d : FreeMonoid α}
    (hsz : (s.tree.splitAt d p).leaves.length ≤ C.maxLeaves) (E : Edges α)
    (mv : List Bool × α → Bool) :
    Ψ C (newHypothesis (s.tree.splitAt d p) E mv) < Ψ C s := by
  unfold Ψ
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

/-! ## One step -/

section Step

variable [Fintype α] {σ : Type*} [Fintype σ] {M : DFA α σ} {side : σ → Bool}

omit [Fintype α] in
theorem inv_fresh {T : DTree α} {E : Edges α} {mv : List Bool × α → Bool}
    (hm : 1 ≤ C.recordsToMove) (hh : 1 ≤ C.harvestSize) (hn : 1 ≤ C.cleanProbesToEnd)
    (hE : EdgesOK read T E)
    (hsz : T.leaves.length ≤ C.maxLeaves) : Inv read C (newHypothesis T E mv) where
  edges := hE
  recs_mem := by simp [newHypothesis]
  recs_lt := by simp [newHypothesis]; omega
  und_lt := by simp [newHypothesis]; omega
  clean_lt := by simp [newHypothesis]; omega
  size := hsz

/-- What a step from `s` promises. -/
def StepOK (s : Level2State α) : Level2State α ⊕ RoundEnd (UndecidedSite α) → Prop
  | .inl s' => Inv read C s' ∧ Reached read s'.tree ∧ Θ C s' < Θ C s
  | .inr e => EndOK e

/-- Undecided strings charged to a class of the tree. -/
theorem charge_ok {s : Level2State α} (hs : Inv read C s) (hr : Reached read s.tree)
    {k : UndecidedSite α}
    {zs : List (FreeMonoid α)} {cl : Bool} (hk : SiteOK s.tree k)
    (hne : cl = false → zs ≠ []) :
    StepOK read C s (addUndecided C s k zs cl) := by
  have hkc := mem_classes hk
  have hDc : ∀ c', Dc { s with
      undecided := Function.update s.undecided k (s.undecided k ++ zs), cleanRun := c' }
      = Dc s + zs.length := by
    intro c'
    unfold Dc
    have : ∀ k', (Function.update s.undecided k (s.undecided k ++ zs) k').length
        = Function.update (fun k' => (s.undecided k').length) k
          ((s.undecided k).length + zs.length) k' := by
      intro k'
      by_cases hk' : k' = k
      · subst hk'; simp
      · simp [hk']
    simp only [this, sum_update_of_mem hkc]
    rw [← add_sum_erase _ _ hkc]
    simp only [sdiff_singleton_eq_erase]
    omega
  have hinv : ∀ c', c' < C.cleanProbesToEnd → (s.undecided k ++ zs).length < C.harvestSize →
      Inv read C { s with
        undecided := Function.update s.undecided k (s.undecided k ++ zs), cleanRun := c' } := by
    intro c' hc' hh
    refine ⟨hs.edges, hs.recs_mem, hs.recs_lt, ?_, hc', hs.size⟩
    intro k'
    by_cases hk' : k' = k
    · subst hk'; simpa using hh
    · simp only [ne_eq, hk', not_false_eq_true, Function.update_of_ne]; exact hs.und_lt k'
  unfold addUndecided extendCleanRun
  simp only [Function.update_self]
  split_ifs with hh hcl hc
  · trivial
  · trivial
  · have hc' : s.cleanRun + 1 < C.cleanProbesToEnd := by omega
    have hi := hinv _ hc' (by omega)
    exact ⟨hi, hr, Θ_lt_same read C hs hi rfl rfl rfl (by rw [hDc]; omega) (Or.inl rfl)⟩
  · have hn : 0 < C.cleanProbesToEnd := by have := hs.clean_lt; omega
    have hz : zs ≠ [] := hne (by simpa using hcl)
    have hzl : 0 < zs.length := List.length_pos_iff.2 hz
    have hi := hinv 0 hn (by omega)
    exact ⟨hi, hr, Θ_lt_same read C hs hi rfl rfl rfl (by rw [hDc]; omega)
      (Or.inr ⟨rfl, by rw [hDc]; omega⟩)⟩

theorem step_ok (hI : IdealReads read M side) (hcap : Fintype.card σ + 2 ≤ C.maxLeaves)
    (hm : 1 ≤ C.recordsToMove) (hh : 1 ≤ C.harvestSize) (hn : 1 ≤ C.cleanProbesToEnd)
    {s : Level2State α} (hs : Inv read C s)
    (hr : Reached read s.tree) (x : FreeMonoid α) : StepOK read C s (applyProbe read C s x) := by
  have hOK := probeOrAgree_ok read s.tree s.edges C.walkStart hs.edges x
  unfold applyProbe
  generalize probeOrAgree read s.tree s.edges C.walkStart x = o at hOK
  cases o with
  | agree =>
    simp only
    unfold extendCleanRun
    split_ifs with hc
    · trivial
    · have hc' : s.cleanRun + 1 < C.cleanProbesToEnd := by omega
      have hinv : Inv read C { s with cleanRun := s.cleanRun + 1 } :=
        ⟨hs.edges, hs.recs_mem, hs.recs_lt, hs.und_lt, hc', hs.size⟩
      exact ⟨hinv, hr, Θ_lt_same read C hs hinv rfl rfl rfl le_rfl (Or.inl rfl)⟩
  | undecided k zs cl =>
    exact charge_ok read C hs hr hOK.1 hOK.2.1
  | unlearnedEdge p c u t =>
    obtain ⟨hp, hu, huc, hE⟩ := hOK
    refine ⟨inv_fresh read C hm hh hn (edgesOK_upd read hs.edges hu huc) hs.size, hr, ?_⟩
    refine Θ_lt_change C hn _ _ _ (Ψ_lt_upd C hp _ _ ?_)
    simp [wt, newHypothesis, hE]
  | wrongEdge p c u t =>
    obtain ⟨hp, hu, huc, t₀, w₀, hE, ht₀⟩ := hOK
    have ht : t ∈ s.tree.leaves := DTree.sift_inl_mem read _ _ t huc
    simp only
    unfold recordDisagreement
    simp only
    rw [hE]
    simp only
    split_ifs with hcnt hmv
    · -- a second target after a redirect: split
      obtain ⟨hw₀, hw₀c⟩ := hs.edges p hp c t₀ w₀ hE
      have hpart := DTree.sift_part read s.tree _ _ t t₀ huc hw₀c (Ne.symm ht₀)
      have hreach : Reached read
          (s.tree.splitAt (FreeMonoid.of c * s.tree.midAt (commonPrefix t t₀)) p) := by
        refine reached_splitAt read hr hp hu hw₀ ?_ ?_ ?_
        · simpa [mul_assoc] using hpart.1
        · simpa [mul_assoc] using hpart.2.1
        · simpa [mul_assoc] using hpart.2.2
      have hsz := (leaves_le_of_reached read hI hreach).trans hcap
      unfold splitLeaf
      simp only
      rw [if_neg (by omega)]
      refine ⟨inv_fresh read C hm hh hn (edgesOK_retarget read hs.edges hp _) hsz, hreach, ?_⟩
      exact Θ_lt_change C hn _ _ _ (Ψ_lt_split C hp hsz _ _)
    · -- one target reaches `recordsToMove`: redirect
      refine ⟨inv_fresh read C hm hh hn (edgesOK_upd read hs.edges hu huc) hs.size, hr, ?_⟩
      refine Θ_lt_change C hn _ _ _ (Ψ_lt_upd C hp _ _ ?_)
      simp [wt, newHypothesis, hE, hmv]
    · have hinv : Inv read C { s with records := s.records ++ [(p, c, t)], cleanRun := 0 } := by
        refine ⟨hs.edges, ?_, ?_, hs.und_lt,
          show 0 < C.cleanProbesToEnd by have := hs.clean_lt; omega, hs.size⟩
        · intro r hr'
          rcases List.mem_append.1 hr' with hr' | hr'
          · exact hs.recs_mem r hr'
          · simp only [List.mem_singleton] at hr'; subst hr'; exact ⟨hp, ht⟩
        · intro r
          by_cases hrr : r = (p, c, t)
          · subst hrr
            change List.count (p, c, t) (s.records ++ [(p, c, t)]) < C.recordsToMove
            omega
          · have := hs.recs_lt r
            simp only [List.count_append, List.count_singleton]
            simp [Ne.symm hrr, this]
      refine ⟨hinv, hr, Θ_lt_same read C hs hinv rfl rfl rfl ?_ (Or.inr ⟨rfl, ?_⟩)⟩
      · unfold Dc; simp
      · unfold Dc; simp

end Step

end Ideal

end OrthoDFA
