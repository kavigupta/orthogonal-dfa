import OrthoDFA.EpsRound
import OrthoDFA.Proofs.RandomStretch

/-!
# The tree when reads can be wrong

A state's true leaf is where its strings sift when no read on the way is undecided or wrong. A
target is real at an edge where some state at the edge's leaf reaches it under the edge's letter.
A split is genuine where two states at its leaf part at its midfix; a split between two real
targets is. Each genuine split adds a true leaf, so a tree grown by genuine splits has at most
`|Q| + 1` leaves (`GInv`).

A fix is fake where its target is not real. A step that is not a fake fix keeps `GInv`, and only
a fake fix can make the tree too big. A fake fix needs `m` records at a target that is not real,
each from a probe that can read a string wrong.
-/

namespace OrthoDFA

namespace Random

open OrthoDFA.Ideal (DTree Edges pre lcp probe Outcome located search agrees walk walk_cons
  length_walk_le pre_of_le pre_succ Found OutcomeOK EdgesOK)

variable {α : Type*} {σ : Type*} (M : DFA α σ) (side : σ → Bool)

/-- The leaf a state's strings sift to where no read on the way is undecided or wrong. -/
def leafOf : DTree α → σ → List Bool
  | .leaf, _ => []
  | .node m r a, q =>
    if side (M.evalFrom q m.toList) then true :: leafOf a q else false :: leafOf r q

/-- Some state at the leaf `p` reaches the target `t` under `c`. -/
def RealT (T : DTree α) (p : List Bool) (c : α) (t : List Bool) : Prop :=
  ∃ q, leafOf M side T q = p ∧ leafOf M side T (M.evalFrom q [c]) = t

/-- Two states at the leaf `p` part at the midfix `d`. -/
def GenuineSplit (T : DTree α) (p : List Bool) (d : FreeMonoid α) : Prop :=
  ∃ q₁ q₂, leafOf M side T q₁ = p ∧ leafOf M side T q₂ = p ∧
    side (M.evalFrom q₁ d.toList) ≠ side (M.evalFrom q₂ d.toList)

theorem eval_mul (z m : FreeMonoid α) :
    M.eval (z * m).toList = M.evalFrom (M.eval z.toList) m.toList := by
  simp [FreeMonoid.toList_mul, DFA.eval, DFA.evalFrom_of_append]

theorem leafOf_mem : ∀ (T : DTree α) (q : σ), leafOf M side T q ∈ T.leaves
  | .leaf, q => by simp [leafOf, DTree.leaves]
  | .node m r a, q => by
    unfold leafOf
    split_ifs
    · simp [DTree.leaves, leafOf_mem a q]
    · simp [DTree.leaves, leafOf_mem r q]

/-- A sift with no wrong read on its way that ends at a leaf ends at the true leaf. -/
theorem sift_true (read : FreeMonoid α → ARU) : ∀ (T : DTree α) (z : FreeMonoid α) (p : List Bool),
    T.sift read z = .inl p →
    (∀ m ∈ mids T, read (z * m) ≠ if side (M.eval (z * m).toList) then .reject else .accept) →
    p = leafOf M side T (M.eval z.toList)
  | .leaf, z, p, h, _ => by simp only [DTree.sift, Sum.inl.injEq] at h; simp [leafOf, ← h]
  | .node m r a, z, p, h, hw => by
    classical
    have hm := hw m (by simp [mids])
    have hsub : ∀ T', (T' = r ∨ T' = a) → ∀ m' ∈ mids T', read (z * m') ≠
        if side (M.eval (z * m').toList) then .reject else .accept :=
      fun T' hT' m' hm' => hw m' (by rcases hT' with rfl | rfl <;> simp [mids, hm'])
    simp only [DTree.sift] at h
    unfold leafOf
    rw [← eval_mul]
    split at h
    · rename_i hr
      have hs : side (M.eval (z * m).toList) = true := by
        cases hsd : side (M.eval (z * m).toList)
        · exact absurd (by rw [hr, hsd]; simp) hm
        · rfl
      rcases hq : a.sift read z with q | q <;> rw [hq] at h <;> simp only [Sum.map_inl,
        Sum.map_inr, Sum.inl.injEq, reduceCtorEq] at h
      rw [if_pos hs, ← h, sift_true read a z q hq (hsub a (Or.inr rfl))]
    · rename_i hr
      have hs : side (M.eval (z * m).toList) = false := by
        cases hsd : side (M.eval (z * m).toList)
        · rfl
        · exact absurd (by rw [hr, hsd]; simp) hm
      rcases hq : r.sift read z with q | q <;> rw [hq] at h <;> simp only [Sum.map_inl,
        Sum.map_inr, Sum.inl.injEq, reduceCtorEq] at h
      rw [if_neg (by rw [hs]; exact Bool.false_ne_true), ← h,
        sift_true read r z q hq (hsub r (Or.inl rfl))]
    · simp at h

/-- Two states at different true leaves part at the node where their paths part. -/
theorem leafOf_part : ∀ (T : DTree α) (q₁ q₂ : σ), leafOf M side T q₁ ≠ leafOf M side T q₂ →
    side (M.evalFrom q₁ (T.midAt (lcp (leafOf M side T q₁) (leafOf M side T q₂))).toList)
      ≠ side (M.evalFrom q₂ (T.midAt (lcp (leafOf M side T q₁) (leafOf M side T q₂))).toList)
  | .leaf, q₁, q₂, h => by simp [leafOf] at h
  | .node m r a, q₁, q₂, h => by
    unfold leafOf at h ⊢
    by_cases h1 : side (M.evalFrom q₁ m.toList) <;> by_cases h2 : side (M.evalFrom q₂ m.toList)
    · rw [if_pos h1, if_pos h2] at h ⊢
      have h' : leafOf M side a q₁ ≠ leafOf M side a q₂ := fun e => h (by rw [e])
      simpa [lcp, DTree.midAt] using leafOf_part a q₁ q₂ h'
    · rw [if_pos h1, if_neg h2]; simp [lcp, DTree.midAt, h1, h2]
    · rw [if_neg h1, if_pos h2]; simp [lcp, DTree.midAt, h1, h2]
    · rw [if_neg h1, if_neg h2] at h ⊢
      have h' : leafOf M side r q₁ ≠ leafOf M side r q₂ := fun e => h (by rw [e])
      simpa [lcp, DTree.midAt] using leafOf_part r q₁ q₂ h'

/-- A split between two real targets is genuine. -/
theorem genuine_of_real {T : DTree α} {p t t₀ : List Bool} {c : α} (ht : RealT M side T p c t)
    (ht₀ : RealT M side T p c t₀) (hne : t ≠ t₀) :
    GenuineSplit M side T p (FreeMonoid.of c * T.midAt (lcp t t₀)) := by
  obtain ⟨q₁, h1, h1'⟩ := ht
  obtain ⟨q₂, h2, h2'⟩ := ht₀
  refine ⟨q₁, q₂, h1, h2, ?_⟩
  have := leafOf_part M side T (M.evalFrom q₁ [c]) (M.evalFrom q₂ [c]) (by rw [h1', h2']; exact hne)
  rw [h1', h2'] at this
  simpa [FreeMonoid.toList_mul, DFA.evalFrom_of_append] using this

theorem leafOf_splitAt (d : FreeMonoid α) : ∀ (T : DTree α) (p : List Bool), p ∈ T.leaves →
    ∀ q, leafOf M side (T.splitAt d p) q = if leafOf M side T q = p
      then p ++ [side (M.evalFrom q d.toList)] else leafOf M side T q
  | .leaf, p, hp, q => by
    simp only [DTree.leaves, List.mem_singleton] at hp
    subst hp
    cases h : side (M.evalFrom q d.toList) <;> simp [DTree.splitAt, leafOf, h]
  | .node m r a, [], hp, q => by simp [DTree.leaves] at hp
  | .node m r a, false :: p, hp, q => by
    have hp' : p ∈ r.leaves := by simpa [DTree.leaves] using hp
    simp only [DTree.splitAt, leafOf]
    by_cases hs : side (M.evalFrom q m.toList)
    · simp [hs]
    · simp only [hs, Bool.false_eq_true, if_false, List.cons.injEq, true_and,
        leafOf_splitAt d r p hp' q]
      split_ifs <;> simp
  | .node m r a, true :: p, hp, q => by
    have hp' : p ∈ a.leaves := by simpa [DTree.leaves] using hp
    simp only [DTree.splitAt, leafOf]
    by_cases hs : side (M.evalFrom q m.toList)
    · simp only [hs, if_true, List.cons.injEq, true_and, leafOf_splitAt d a p hp' q]
      split_ifs <;> simp
    · simp [hs]

theorem leaves_snoc_not_mem : ∀ (T : DTree α) (p : List Bool), p ∈ T.leaves → ∀ b : Bool,
    p ++ [b] ∉ T.leaves
  | .leaf, p, hp, b => by simp [DTree.leaves] at hp ⊢
  | .node m r a, [], hp, b => by simp [DTree.leaves] at hp
  | .node m r a, false :: p, hp, b => by
    have hp' : p ∈ r.leaves := by simpa [DTree.leaves] using hp
    simpa [DTree.leaves] using leaves_snoc_not_mem r p hp' b
  | .node m r a, true :: p, hp, b => by
    have hp' : p ∈ a.leaves := by simpa [DTree.leaves] using hp
    simpa [DTree.leaves] using leaves_snoc_not_mem a p hp' b

variable [Fintype σ]

open scoped Classical in
/-- The true leaves. -/
noncomputable def img (T : DTree α) : Finset (List Bool) :=
  (Finset.univ : Finset σ).image (leafOf M side T)

theorem img_card_le (T : DTree α) : (img M side T).card ≤ Fintype.card σ :=
  Finset.card_image_le.trans (by simp)

/-- A genuine split adds a true leaf. -/
theorem img_card_splitAt {T : DTree α} {p : List Bool} {d : FreeMonoid α} (hp : p ∈ T.leaves)
    (hg : GenuineSplit M side T p d) :
    (img M side T).card + 1 ≤ (img M side (T.splitAt d p)).card := by
  classical
  obtain ⟨q₁, q₂, h1, h2, h12⟩ := hg
  have hpimg : p ∈ img M side T := Finset.mem_image.2 ⟨q₁, Finset.mem_univ _, h1⟩
  have hsub : ((img M side T).erase p) ∪ {p ++ [true], p ++ [false]}
      ⊆ img M side (T.splitAt d p) := by
    intro y hy
    rcases Finset.mem_union.1 hy with hy | hy
    · obtain ⟨hyp, hy⟩ := Finset.mem_erase.1 hy
      obtain ⟨q, -, rfl⟩ := Finset.mem_image.1 hy
      refine Finset.mem_image.2 ⟨q, Finset.mem_univ _, ?_⟩
      rw [leafOf_splitAt M side d T p hp q, if_neg hyp]
    · have hq : ∀ q, leafOf M side T q = p →
          p ++ [side (M.evalFrom q d.toList)] ∈ img M side (T.splitAt d p) := fun q hq =>
        Finset.mem_image.2 ⟨q, Finset.mem_univ _, by
          rw [leafOf_splitAt M side d T p hp q, if_pos hq]⟩
      simp only [Finset.mem_insert, Finset.mem_singleton] at hy
      have hb : ∀ b : Bool, ∃ q, leafOf M side T q = p ∧ side (M.evalFrom q d.toList) = b := by
        intro b
        by_cases hs1 : side (M.evalFrom q₁ d.toList) = b
        · exact ⟨q₁, h1, hs1⟩
        · refine ⟨q₂, h2, ?_⟩
          cases b <;> cases h1' : side (M.evalFrom q₁ d.toList) <;>
            cases h2' : side (M.evalFrom q₂ d.toList) <;> simp_all
      rcases hy with rfl | rfl
      · obtain ⟨q, hq1, hq2⟩ := hb true
        rw [← hq2]; exact hq q hq1
      · obtain ⟨q, hq1, hq2⟩ := hb false
        rw [← hq2]; exact hq q hq1
  have hdisj : Disjoint ((img M side T).erase p) {p ++ [true], p ++ [false]} := by
    rw [Finset.disjoint_left]
    intro y hy hy'
    obtain ⟨-, hy⟩ := Finset.mem_erase.1 hy
    obtain ⟨q, -, rfl⟩ := Finset.mem_image.1 hy
    have := leafOf_mem M side T q
    simp only [Finset.mem_insert, Finset.mem_singleton] at hy'
    rcases hy' with h | h <;> rw [h] at this <;> exact leaves_snoc_not_mem T p hp _ this
  have h2 : ({p ++ [true], p ++ [false]} : Finset (List Bool)).card = 2 := by
    rw [Finset.card_insert_of_notMem (by simp), Finset.card_singleton]
  have := Finset.card_le_card hsub
  rw [Finset.card_union_of_disjoint hdisj, h2, Finset.card_erase_of_mem hpimg] at this
  have := Finset.card_pos.2 ⟨p, hpimg⟩
  omega

variable [Fintype α] [DecidableEq α] (read : FreeMonoid α → ARU) (C : Cfg)

/-- The tree has a true leaf for each leaf past the first, and every redirected edge has a real
target. -/
structure GInv (s : RState α) : Prop where
  leaves : s.tree.leaves.length ≤ (img M side s.tree).card + 1
  real : ∀ p c t₀ w, s.edges p c = some (t₀, w) → s.moved p c = true →
    RealT M side s.tree p c t₀

/-- A fix at a target that is not real. -/
def FakeFix (s : RState α) (x : FreeMonoid α) : Prop :=
  ∃ p c u t, probeR read s.tree s.edges C.k x = .edge p c u t ∧
    C.m ≤ (s.recs ++ [(p, c, t)]).count (p, c, t) ∧ ¬ RealT M side s.tree p c t

theorem gInv_start : GInv M side (start : RState α) where
  leaves := by
    have : 0 < (img M side (start : RState α).tree).card :=
      Finset.card_pos.2 ⟨_, Finset.mem_image_of_mem _ (Finset.mem_univ M.start)⟩
    simp only [start, fresh, DTree.leaves] at this ⊢
    simp only [List.map_cons, List.map_nil, List.cons_append, List.nil_append, List.length_cons,
      List.length_nil]
    omega
  real := fun _ _ _ _ h => by simp [start, fresh] at h

theorem gInv_leaves {s : RState α} (hg : GInv M side s) :
    s.tree.leaves.length ≤ Fintype.card σ + 1 :=
  hg.leaves.trans (by have := img_card_le M side s.tree; omega)

theorem gInv_cls {s : RState α} (hs : Inv read C s) (hg : GInv M side s) :
    s.tree ∈ classSet (Fintype.card σ) :=
  classSet_mono' (by have := gInv_leaves M side hg; omega) hs.cls

/-- A split of an edge redirected to a real target, at the `m`-th record of a real one, is
genuine and keeps `GInv`'s count. -/
theorem gInv_split {s : RState α} (hg : GInv M side s) {p t t₀ : List Bool} {c : α}
    {w₀ : FreeMonoid α} (hp : p ∈ s.tree.leaves) (hE : s.edges p c = some (t₀, w₀))
    (hmv : s.moved p c = true) (ht : RealT M side s.tree p c t) (hne : t₀ ≠ t) :
    (s.tree.splitAt (FreeMonoid.of c * s.tree.midAt (lcp t t₀)) p).leaves.length
      ≤ (img M side (s.tree.splitAt (FreeMonoid.of c * s.tree.midAt (lcp t t₀)) p)).card + 1 := by
  have hgen := genuine_of_real M side ht (hg.real p c t₀ w₀ hE hmv) (Ne.symm hne)
  have h1 := img_card_splitAt M side hp hgen
  rw [DTree.length_leaves_splitAt _ _ _ hp]
  have := hg.leaves
  omega

/-- Only a fake fix can make the tree too big. -/
theorem step_ne_tooBig_g (hcap : Fintype.card σ + 2 ≤ C.Lmax) {s : RState α}
    (hs : Inv read C s) (hg : GInv M side s) {x : FreeMonoid α}
    (hnf : ¬ FakeFix M side read C s x) : step read C s x ≠ .inr .tooBig := by
  intro h
  obtain ⟨p, c, u, t, t₀, w₀, ho, hE, htt, hp, -, -, hmv, hc, hlt⟩ := step_tooBig read C hs h
  have ht : RealT M side s.tree p c t := by
    by_contra hr
    exact hnf ⟨p, c, u, t, ho, hc, hr⟩
  have := gInv_split M side hg hp hE hmv ht htt
  have := img_card_le M side (s.tree.splitAt (FreeMonoid.of c * s.tree.midAt (lcp t t₀)) p)
  omega

/-- A step that changes the hypothesis without a fake fix keeps `GInv`. -/
theorem gInv_change {s s' : RState α} (hs : Inv read C s) (hg : GInv M side s)
    {x : FreeMonoid α} (h : step read C s x = .inl s')
    (hne : ¬ (s'.tree = s.tree ∧ s'.edges = s.edges)) (hnf : ¬ FakeFix M side read C s x) :
    GInv M side s' := by
  rcases step_cases read C hs h with
    ⟨p, c, u, t, -, hE, -, -, -, rfl⟩ |
    ⟨p, c, u, t, t₀, w₀, ho, hE, htt, hp, -, -, hc, hfix⟩ | ⟨rfl, -, -⟩
  · refine ⟨hg.leaves, fun p' c' t₁ w h1 h2 => ?_⟩
    by_cases hpc : (p', c') = (p, c)
    · simp only [Prod.mk.injEq] at hpc
      obtain ⟨rfl, rfl⟩ := hpc
      simp [fresh, Ideal.upd2_same] at h2
    · simp only [fresh, Ideal.upd2_ne _ _ hpc] at h1 h2
      exact hg.real p' c' t₁ w h1 h2
  · have ht : RealT M side s.tree p c t := by
      by_contra hr
      exact hnf ⟨p, c, u, t, ho, hc, hr⟩
    unfold fix at hfix
    rw [hE] at hfix
    simp only at hfix
    split_ifs at hfix with hmv
    · unfold split at hfix
      simp only at hfix
      split_ifs at hfix with hlt
      simp only [Sum.inl.injEq] at hfix
      subst hfix
      exact ⟨gInv_split M side hg hp hE hmv ht htt, fun _ _ _ _ _ h2 => by simp [fresh] at h2⟩
    · simp only [Sum.inl.injEq] at hfix
      subst hfix
      refine ⟨hg.leaves, fun p' c' t₁ w h1 h2 => ?_⟩
      by_cases hpc : (p', c') = (p, c)
      · simp only [Prod.mk.injEq] at hpc
        obtain ⟨rfl, rfl⟩ := hpc
        simp only [fresh, Ideal.upd2_same, Option.some.injEq, Prod.mk.injEq] at h1
        rw [← h1.1]
        exact ht
      · simp only [fresh, Ideal.upd2_ne _ _ hpc] at h1 h2
        exact hg.real p' c' t₁ w h1 h2
  · exact absurd ⟨rfl, rfl⟩ hne

end Random

end OrthoDFA
