import OrthoDFA.Proofs.Power

/-!
# The round's steps in order

`roundTrace` lists the steps the round takes, each with the accumulator before it; `roundFind`
is the first of them at which its predicate holds. A leaf once split stays an inner node, so no
later step tests at its key.
-/

namespace OrthoDFA

variable {α : Type*} [Fintype α] [DecidableEq α]

namespace DTree

/-- `p` leads to an inner node. -/
def Inner : DTree α → List Bool → Prop
  | .leaf, _ => False
  | .node _ _ _, [] => True
  | .node _ r _, false :: p => Inner r p
  | .node _ _ a, true :: p => Inner a p

omit [Fintype α] [DecidableEq α] in
theorem Inner.not_mem_paths : ∀ {t : DTree α} {p : List Bool}, t.Inner p → p ∉ t.paths
  | .leaf, _, h => h.elim
  | .node _ _ _, [], _ => paths_ne_nil_of_node
  | .node _ r _, false :: p, h => by
    have := Inner.not_mem_paths (t := r) (p := p) h
    simpa [paths] using this
  | .node _ _ a, true :: p, h => by
    have := Inner.not_mem_paths (t := a) (p := p) h
    simpa [paths] using this

omit [Fintype α] [DecidableEq α] in
theorem Inner.splitAt (d : FreeMonoid α) :
    ∀ {t : DTree α} {p : List Bool} (q : List Bool), t.Inner p → (t.splitAt d q).Inner p
  | .leaf, _, _, h => h.elim
  | .node _ _ _, _, [], h => h
  | .node _ _ _, [], false :: _, _ => trivial
  | .node _ _ _, [], true :: _, _ => trivial
  | .node _ r _, false :: _, false :: q, h => Inner.splitAt d (t := r) q h
  | .node _ _ _, false :: _, true :: _, h => h
  | .node _ _ _, true :: _, false :: _, h => h
  | .node _ _ a, true :: _, true :: q, h => Inner.splitAt d (t := a) q h

omit [Fintype α] [DecidableEq α] in
theorem inner_splitAt (d : FreeMonoid α) :
    ∀ (t : DTree α) (q : List Bool), q ∈ t.paths → (t.splitAt d q).Inner q
  | .leaf, [], _ => trivial
  | .leaf, _ :: _, h => by simp [paths] at h
  | .node _ _ _, [], h => absurd h paths_ne_nil_of_node
  | .node _ r _, false :: q, h => inner_splitAt d r q (by simpa [paths] using h)
  | .node _ _ a, true :: q, h => inner_splitAt d a q (by simpa [paths] using h)

end DTree

section Seed

variable (K : StageKnobs α) (R : CutReads α)

/-- A step's test key is at a leaf, and away from the forced keys the step splits exactly where
the test does. -/
theorem seedStep_key_spec {t : DTree α} {pool : List (FreeMonoid α)} {edges : Edges α}
    {skip : TestKey α → FreeMonoid α → Prop} {forced : Set (TestKey α)} {k : ℕ}
    {x : FreeMonoid α} {ps : List (List Bool)} {fd : ℕ} {κ : TestKey α}
    (h : (seedStep K R t pool edges skip forced k x ps fd).key = some κ) :
    κ.1 ∈ t.paths ∧ (κ ∉ forced →
      ((∃ y sp, seedStep K R t pool edges skip forced k x ps fd = .split κ.2 κ.1 y sp)
        ↔ verdict K R t pool κ.1 κ.2 (t.paths.length * Fintype.card α) (skip κ) = .split)) := by
  revert h
  unfold seedStep
  simp only []
  split
  · simp [SeedResult.key]
  rename_i c hc
  split
  · simp [SeedResult.key]
  rename_i s2 y he
  split_ifs with h1
  · simp [SeedResult.key]
  split
  · simp [SeedResult.key]
  rename_i p hsp
  split_ifs with h2
  · simp [SeedResult.key]
  split
  · simp [SeedResult.key]
  · simp [SeedResult.key]
  rename_i d hd
  push Not at h2
  have hmem : ps.getD (fd - 1 - k) [] ∈ t.paths := h2.1 ▸ DTree.sift_mem_paths _ _ _ hsp
  split_ifs with hf
  · intro hk
    simp only [SeedResult.key, Option.some.injEq] at hk
    subst hk
    exact ⟨hmem, fun hn => absurd hf hn⟩
  · split
    · rename_i hv
      intro hk
      simp only [SeedResult.key, Option.some.injEq] at hk
      subst hk
      exact ⟨hmem, fun _ => ⟨fun _ => hv, fun _ => ⟨_, _, rfl⟩⟩⟩
    · rename_i hv
      intro hk
      simp only [SeedResult.key, Option.some.injEq] at hk
      subst hk
      refine ⟨hmem, fun _ => ⟨fun ⟨_, _, h⟩ => by simp at h, fun h => absurd h ?_⟩⟩
      exact hv

end Seed

section Step

variable (C : StrongCfg α) (R : CutReads α)

theorem stepTest_spec {A : RoundAcc α} {x : FreeMonoid α} {κ : TestKey α}
    {ts : List (FreeMonoid α × Bool)} (h : stepTest C R A x = some (κ, ts)) :
    ∃ ps fd, probeOutcome R A.s.tree A.s.edges C.k x = .edge ps fd
      ∧ (seedStep C.K R A.s.tree A.s.pool A.s.edges (stepSkip C.K R C.k A.s x) C.K.forced C.k x
        ps fd).key = some κ := by
  unfold stepTest at h
  split at h
  · rename_i ps fd ho
    obtain ⟨κ', hk, hκ⟩ := Option.map_eq_some_iff.1 h
    simp only [Prod.mk.injEq] at hκ
    exact ⟨ps, fd, ho, hκ.1 ▸ hk⟩
  · exact absurd h (by simp)

/-- A step whose test splits splits the test's leaf on its distinguisher. -/
theorem strongStep_of_split (h0 : C.K.forced = ∅) {A : RoundAcc α} {x : FreeMonoid α}
    {κ : TestKey α} {ts : List (FreeMonoid α × Bool)} (h : stepTest C R A x = some (κ, ts))
    (hv : verdict C.K R A.s.tree A.s.pool κ.1 κ.2 (A.s.tree.paths.length * Fintype.card α)
      (stepSkip C.K R C.k A.s x κ) = .split) :
    κ.1 ∈ A.s.tree.paths ∧ (strongStep C R A x).s.tree = A.s.tree.splitAt κ.2 κ.1
      ∧ ∃ r, (strongStep C R A x).splits = A.splits ++ [r] := by
  obtain ⟨ps, fd, ho, hk⟩ := stepTest_spec C R h
  obtain ⟨hmem, hsp⟩ := seedStep_key_spec C.K R hk
  obtain ⟨y, sp, hs⟩ := (hsp (by rw [h0]; exact Set.notMem_empty _)).2 hv
  refine ⟨hmem, ?_, ?_⟩
  · rw [(strongStep_state C R A x).1]
    unfold probeStepK
    simp only [ho, hs, closeK]
  · unfold strongStep
    simp only [ho, hs]
    exact ⟨_, rfl⟩

theorem stepTest_path {A : RoundAcc α} {x : FreeMonoid α} {κ : TestKey α}
    {ts : List (FreeMonoid α × Bool)} (h : stepTest C R A x = some (κ, ts)) :
    κ.1 ∈ A.s.tree.paths := by
  obtain ⟨ps, fd, -, hk⟩ := stepTest_spec C R h
  exact (seedStep_key_spec C.K R hk).1

end Step

section Trace

variable (C : StrongCfg α) (R : CutReads α)

/-- Every inner node of `A`'s tree is one of `B`'s. -/
def TreeLe (A B : RoundAcc α) : Prop := ∀ p, A.s.tree.Inner p → B.s.tree.Inner p

theorem TreeLe.refl (A : RoundAcc α) : TreeLe A A := fun _ h => h

theorem TreeLe.trans {A B E : RoundAcc α} (h₁ : TreeLe A B) (h₂ : TreeLe B E) : TreeLe A E :=
  fun p h => h₂ p (h₁ p h)

theorem treeLe_step (A : RoundAcc α) (x : FreeMonoid α) : TreeLe A (strongStep C R A x) := by
  intro p hp
  rw [(strongStep_state C R A x).1]
  rcases probeStepK_tree_cases C.K R C.k A.s x with h | ⟨d, q, h⟩
  · rw [h]; exact hp
  · rw [h]; exact hp.splitAt d q

/-- The steps a pass takes, each with the accumulator before it. -/
noncomputable def passTrace : RoundAcc α → List (FreeMonoid α) → List (RoundAcc α × FreeMonoid α)
  | _, [] => []
  | A, x :: xs =>
    if C.K.patience ≤ A.s.streak ∨ C.budget A.s.tree.paths.length ≤ A.used then []
    else (A, x) :: passTrace (strongStep C R A x) xs

/-- The steps the round takes, each with the accumulator before it. -/
noncomputable def roundTrace :
    (n : ℕ) → ℕ → RoundAcc α → List (FreeMonoid α) → (Fin n → C.Draws)
      → List (RoundAcc α × FreeMonoid α)
  | 0, _, _, _, _ => []
  | n + 1, j, A, first, d =>
    passTrace C R (passStart A) (first ++ List.ofFn (d 0).1) ++
      match strongReading C R j A first (d 0) with
      | (_, .inl _) => []
      | (A', .inr lv) => roundTrace n (j + 1) A' lv (Fin.tail d)

open scoped Classical in
theorem passFind_eq (P : RoundAcc α → FreeMonoid α → Prop) :
    ∀ (A : RoundAcc α) (probes : List (FreeMonoid α)),
      passFind C R P A probes = (passTrace C R A probes).find? fun a => decide (P a.1 a.2)
  | _, [] => rfl
  | A, x :: xs => by
    unfold passFind passTrace
    split_ifs with hg hp
    · rfl
    · simp [List.find?_cons, hp]
    · simp only [List.find?_cons, hp, decide_false]
      exact passFind_eq P _ xs

open scoped Classical in
theorem roundFind_eq (P : RoundAcc α → FreeMonoid α → Prop) :
    ∀ (n j : ℕ) (A : RoundAcc α) (first : List (FreeMonoid α)) (d : Fin n → C.Draws),
      roundFind C R P n j A first d
        = (roundTrace C R n j A first d).find? fun a => decide (P a.1 a.2)
  | 0, _, _, _, _ => rfl
  | n + 1, j, A, first, d => by
    simp only [roundFind, roundTrace, List.find?_append, ← passFind_eq]
    rcases passFind C R P (passStart A) (first ++ List.ofFn (d 0).1) with _ | r
    · simp only [Option.none_or]
      rcases strongReading C R j A first (d 0) with ⟨A', e | lv⟩
      · rfl
      · exact roundFind_eq P n (j + 1) A' lv (Fin.tail d)
    · rfl

theorem passTrace_le :
    ∀ (probes : List (FreeMonoid α)) (A : RoundAcc α),
      (∀ b ∈ passTrace C R A probes, TreeLe A b.1)
      ∧ (∀ b ∈ passTrace C R A probes,
        TreeLe (strongStep C R b.1 b.2) (probes.foldl (passBody C R) A))
      ∧ TreeLe A (probes.foldl (passBody C R) A)
      ∧ (passTrace C R A probes).Pairwise fun a b => TreeLe (strongStep C R a.1 a.2) b.1
  | [], A => by simp [passTrace, TreeLe.refl]
  | x :: xs, A => by
    by_cases hg : C.K.patience ≤ A.s.streak ∨ C.budget A.s.tree.paths.length ≤ A.used
    · have ht : passTrace C R A (x :: xs) = [] := by rw [passTrace.eq_2, if_pos hg]
      rw [ht, fold_of_guard C R A hg]
      simp [TreeLe.refl]
    have ht : passTrace C R A (x :: xs) = (A, x) :: passTrace C R (strongStep C R A x) xs := by
      rw [passTrace.eq_2, if_neg hg]
    have hf : (x :: xs).foldl (passBody C R) A = xs.foldl (passBody C R) (strongStep C R A x) := by
      simp only [List.foldl_cons]
      rw [show passBody C R A x = strongStep C R A x by unfold passBody; rw [if_neg hg]]
    obtain ⟨h1, h2, h3, h4⟩ := passTrace_le xs (strongStep C R A x)
    have hA := treeLe_step C R A x
    rw [ht, hf]
    refine ⟨?_, ?_, hA.trans h3, List.pairwise_cons.2 ⟨fun b hb => h1 b hb, h4⟩⟩
    · intro b hb
      rcases List.mem_cons.1 hb with rfl | hb
      · exact TreeLe.refl _
      · exact hA.trans (h1 b hb)
    · intro b hb
      rcases List.mem_cons.1 hb with rfl | hb
      · exact h3
      · exact h2 b hb

theorem roundTrace_le :
    ∀ (n j : ℕ) (A : RoundAcc α) (first : List (FreeMonoid α)) (d : Fin n → C.Draws),
      (∀ b ∈ roundTrace C R n j A first d, TreeLe A b.1)
      ∧ (roundTrace C R n j A first d).Pairwise fun a b => TreeLe (strongStep C R a.1 a.2) b.1
  | 0, _, _, _, _ => by simp [roundTrace]
  | n + 1, j, A, first, d => by
    obtain ⟨p1, p2, p3, p4⟩ := passTrace_le C R (first ++ List.ofFn (d 0).1) (passStart A)
    have hsame : (strongReading C R j A first (d 0)).1.s.tree
        = ((first ++ List.ofFn (d 0).1).foldl (passBody C R) (passStart A)).s.tree := by
      rw [(strongReading_fst C R j A first (d 0)).1, strongPass_start]
    simp only [roundTrace]
    rcases hR : strongReading C R j A first (d 0) with ⟨A', e | lv⟩
    · simp only [List.append_nil]
      exact ⟨fun b hb => p1 b hb, p4⟩
    · rw [hR] at hsame
      obtain ⟨q1, q2⟩ := roundTrace_le n (j + 1) A' lv (Fin.tail d)
      have hfA : TreeLe ((first ++ List.ofFn (d 0).1).foldl (passBody C R) (passStart A)) A' :=
        fun p hp => by simp only [] at hsame; rw [hsame]; exact hp
      refine ⟨fun b hb => ?_, List.pairwise_append.2 ⟨p4, q2, fun a ha b hb =>
        (p2 a ha).trans (hfA.trans (q1 b hb))⟩⟩
      rcases List.mem_append.1 hb with hb | hb
      · exact p1 b hb
      · exact (p3.trans hfA).trans (q1 b hb)

/-- No step after one whose test splits at `κ` reaches a test at `κ`. -/
theorem no_test_after_split (h0 : C.K.forced = ∅) {A B : RoundAcc α} {x y : FreeMonoid α}
    {κ : TestKey α} {ts ts' : List (FreeMonoid α × Bool)} (h : stepTest C R A x = some (κ, ts))
    (hv : verdict C.K R A.s.tree A.s.pool κ.1 κ.2 (A.s.tree.paths.length * Fintype.card α)
      (stepSkip C.K R C.k A.s x κ) = .split)
    (hle : TreeLe (strongStep C R A x) B) (h' : stepTest C R B y = some (κ, ts')) : False := by
  obtain ⟨hmem, htree, -⟩ := strongStep_of_split C R h0 h hv
  have hin : (strongStep C R A x).s.tree.Inner κ.1 := by
    rw [htree]; exact DTree.inner_splitAt _ _ _ hmem
  exact (hle _ hin).not_mem_paths (stepTest_path C R h')

end Trace

end OrthoDFA
