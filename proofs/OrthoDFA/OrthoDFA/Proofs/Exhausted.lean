import OrthoDFA.Proofs.Trace

/-!
# The round ends exhausted only after many readings that rerun without progress

Each reading after the first reruns a live draw first, from the tree it was drawn against, so
that probe reaches an edge, and its attempt splits or adds a member. A rerun that adds a member
reached a test that did not split: in the power case, which the power bound covers, or not.
Splits are at most `|Q|` without a noisy split, and each reading spends at most `nr + np` probes.
-/

namespace OrthoDFA

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

section Reading

variable (C : StrongCfg α) (R : CutReads α)

theorem strongReading_rerun {j : ℕ} {A A'' : RoundAcc α} {first lv : List (FreeMonoid α)}
    {y : C.Draws} (h : strongReading C R j A first y = (A'', .inr lv)) :
    lv ≠ [] ∧ lv.length ≤ C.nr
      ∧ (∀ x ∈ lv, LiveEdge R A''.s.tree A''.s.edges C.k (fun _ => False) x)
      ∧ A''.used < C.budget A''.s.tree.paths.length := by
  have hb := strongReading_rerun_budget C R (j := j) (A := A) (first := first) (y := y)
    (lv := lv) (by rw [h])
  obtain ⟨hs1, -, hs3⟩ := strongReading_fst C R j A first y
  rw [h] at hs1 hs3
  simp only [] at hs1 hs3
  have hbud : A''.used < C.budget A''.s.tree.paths.length := by rw [hs1, hs3]; exact hb
  unfold strongReading at h
  simp only [] at h
  split_ifs at h with h1 h2 h3 h4 h5 <;>
    simp only [Prod.mk.injEq, reduceCtorEq, and_false, Sum.inr.injEq] at h
  all_goals
    obtain ⟨hA, rfl⟩ := h
    refine ⟨‹_ ≠ []›, ?_, ?_, hbud⟩
    · simp only [List.length_map]
      exact (List.length_filter_le _ _).trans (by simp)
    · intro x hx
      rw [← hA]
      simp only [List.mem_map, List.mem_filter, decide_eq_true_eq] at hx
      obtain ⟨i, ⟨-, -, hl⟩, rfl⟩ := hx
      first | exact hl | (split_ifs <;> exact hl)

theorem strongReading_exhausted {j : ℕ} {A : RoundAcc α} {first : List (FreeMonoid α)}
    {y : C.Draws} (h : (strongReading C R j A first y).2 = .inl .exhausted) :
    C.budget (strongReading C R j A first y).1.s.tree.paths.length
        ≤ (strongReading C R j A first y).1.used ∧ 0 < C.nr := by
  obtain ⟨hs1, -, hs3⟩ := strongReading_fst C R j A first y
  rw [hs1, hs3]
  unfold strongReading at h
  simp only [] at h
  split_ifs at h with h1 h2 h3 h4 h5 <;> simp only [Sum.inl.injEq, reduceCtorEq] at h
  all_goals
    refine ⟨by assumption, ?_⟩
    obtain ⟨z, hz⟩ := List.exists_mem_of_ne_nil _ ‹_ ≠ []›
    simp only [List.mem_map, List.mem_filter] at hz
    obtain ⟨i, -, -⟩ := hz
    exact i.pos

theorem fold_inv' : ∀ (probes : List (FreeMonoid α)) (X : RoundAcc α), StrongInv R X →
    StrongInv R (probes.foldl (passBody C R) X)
  | [], _, h => h
  | x :: xs, X, h => by
    simp only [List.foldl_cons]
    unfold passBody
    split_ifs
    · exact fold_inv' xs X h
    · exact fold_inv' xs _ (strongStep_inv C R X x h)

theorem strongReading_inv {j : ℕ} {A : RoundAcc α} {first : List (FreeMonoid α)} {y : C.Draws}
    (h : StrongInv R A) : StrongInv R (strongReading C R j A first y).1 := by
  obtain ⟨hs1, hs2, -⟩ := strongReading_fst C R j A first y
  have := fold_inv' C R (first ++ List.ofFn y.1) (passStart A) h
  rw [← strongPass_start] at this
  unfold StrongInv
  rw [hs1, hs2]
  exact this

theorem strongStep_splits_le (A : RoundAcc α) (x : FreeMonoid α) :
    A.splits.length ≤ (strongStep C R A x).splits.length := by
  unfold strongStep
  simp only []
  repeat' split
  all_goals simp

theorem fold_splits_le : ∀ (probes : List (FreeMonoid α)) (X : RoundAcc α),
    X.splits.length ≤ (probes.foldl (passBody C R) X).splits.length
  | [], _ => le_rfl
  | x :: xs, X => by
    simp only [List.foldl_cons]
    unfold passBody
    split_ifs
    · exact fold_splits_le xs X
    · exact (strongStep_splits_le C R X x).trans (fold_splits_le xs _)

theorem fold_used_le : ∀ (probes : List (FreeMonoid α)) (X : RoundAcc α),
    (probes.foldl (passBody C R) X).used ≤ X.used + probes.length
  | [], _ => by simp
  | x :: xs, X => by
    simp only [List.foldl_cons, List.length_cons]
    by_cases hg : C.K.patience ≤ X.s.streak ∨ C.budget X.s.tree.paths.length ≤ X.used
    · rw [show passBody C R X x = X by unfold passBody; rw [if_pos hg]]
      have := fold_used_le xs X
      omega
    · rw [show passBody C R X x = strongStep C R X x by unfold passBody; rw [if_neg hg]]
      have := fold_used_le xs (strongStep C R X x)
      rw [(strongStep_state C R X x).2.1] at this
      omega

end Reading

section Rerun

variable (C : StrongCfg α) (R : CutReads α)

/-- What a reading that reruns `first` starts from. -/
def RerunStart (A : RoundAcc α) (first : List (FreeMonoid α)) : Prop :=
  StrongInv R A ∧ first.length ≤ C.nr ∧ ∀ x, first.head? = some x →
    LiveEdge R A.s.tree A.s.edges C.k (fun _ => False) x
      ∧ A.used < C.budget A.s.tree.paths.length

theorem rerunStart_next {j : ℕ} {A A'' : RoundAcc α} {first lv : List (FreeMonoid α)}
    {y : C.Draws} (hA : StrongInv R A) (h : strongReading C R j A first y = (A'', .inr lv)) :
    RerunStart C R A'' lv := by
  obtain ⟨-, hlen, hlive, hbud⟩ := strongReading_rerun C R h
  have hI := strongReading_inv C R (j := j) (first := first) (y := y) hA
  rw [h] at hI
  refine ⟨hI, hlen, fun x hx => ⟨hlive x (List.mem_of_mem_head? hx), hbud⟩⟩

theorem rerunFirsts_cons (A : RoundAcc α) (first : List (FreeMonoid α))
    (es : List (RoundAcc α × List (FreeMonoid α))) :
    rerunFirsts ((A, first) :: es) = (first.head?.map (A, ·)).toList ++ rerunFirsts es := by
  unfold rerunFirsts
  rw [List.filterMap_cons]
  rcases first with _ | ⟨x, xs⟩ <;> rfl

theorem strongEntries_succ (n j : ℕ) (A : RoundAcc α) (first : List (FreeMonoid α))
    (d : Fin (n + 1) → C.Draws) :
    strongEntries C R (n + 1) j A first d = (A, first) ::
      match strongReading C R j A first (d 0) with
      | (_, .inl _) => []
      | (A', .inr lv) => strongEntries C R n (j + 1) A' lv (Fin.tail d) := rfl

/-- Each rerunning reading starts from an invariant accumulator, with its first draw live, its
first step taken, and that step one of the round's. -/
theorem rerun_facts (hpat : 0 < C.K.patience) :
    ∀ (n j : ℕ) (A : RoundAcc α) (first : List (FreeMonoid α)) (d : Fin n → C.Draws),
      RerunStart C R A first →
      ∀ e ∈ rerunFirsts (strongEntries C R n j A first d),
        StrongInv R e.1 ∧ LiveEdge R e.1.s.tree e.1.s.edges C.k (fun _ => False) e.2
          ∧ (passStart e.1, e.2) ∈ roundTrace C R n j A first d
  | 0, _, _, _, _, _ => by simp [strongEntries, rerunFirsts]
  | n + 1, j, A, first, d, hS => by
    intro e he
    rw [strongEntries_succ, rerunFirsts_cons] at he
    simp only [roundTrace]
    rcases List.mem_append.1 he with he | he
    · rcases hf : first with _ | ⟨x, xs⟩
      · rw [hf] at he; simp at he
      rw [hf] at he
      simp only [List.head?_cons, Option.map_some, Option.toList_some,
        List.mem_singleton] at he
      subst he
      obtain ⟨hl, hb⟩ := hS.2.2 x (by rw [hf]; rfl)
      refine ⟨hS.1, hl, List.mem_append_left _ ?_⟩
      rw [List.cons_append, passTrace.eq_2, if_neg (by
        simp only [passStart]; omega)]
      exact List.mem_cons_self
    · rcases hR : strongReading C R j A first (d 0) with ⟨A'', e' | lv⟩ <;>
        rw [hR] at he <;> simp only [] at he ⊢
      · simp [rerunFirsts] at he
      · obtain ⟨h1, h2, h3⟩ := rerun_facts hpat n (j + 1) A'' lv (Fin.tail d)
          (rerunStart_next C R hS.1 hR) e he
        exact ⟨h1, h2, List.mem_append_right _ h3⟩

end Rerun

end OrthoDFA
