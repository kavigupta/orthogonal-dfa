import OrthoDFA.Proofs.Trace
import OrthoDFA.Proofs.EdgeAttempts

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

section Classify

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
variable (C : StrongCfg α) (O : Oracle μ (FreeMonoid α)) (F : Finset (FreeMonoid α)) (τ : ℝ)
  (R : CutReads α)

/-- The rerun's first step reaches a test that splits. -/
def RerunSplits (e : RoundAcc α × FreeMonoid α) : Prop :=
  ∃ κ ts, stepTest C R (passStart e.1) e.2 = some (κ, ts)
    ∧ verdict C.K R (passStart e.1).s.tree (passStart e.1).s.pool κ.1 κ.2
      ((passStart e.1).s.tree.paths.length * Fintype.card α)
      (stepSkip C.K R C.k (passStart e.1).s e.2 κ) = .split

/-- The rerun's first step reaches a test in the power case that does not split. -/
def RerunMiss (e : RoundAcc α × FreeMonoid α) : Prop :=
  ∃ κ, CaseAt C O F τ κ R (passStart e.1) e.2
    ∧ verdict C.K R (passStart e.1).s.tree (passStart e.1).s.pool κ.1 κ.2
      ((passStart e.1).s.tree.paths.length * Fintype.card α)
      (stepSkip C.K R C.k (passStart e.1).s e.2 κ) ≠ .split

/-- A rerun's first step splits, misses in the power case, or reaches no test in the power case
or splitting. -/
theorem rerun_classify {e : RoundAcc α × FreeMonoid α} (hI : StrongInv R e.1)
    (hl : LiveEdge R e.1.s.tree e.1.s.edges C.k (fun _ => False) e.2) :
    RerunSplits C R e ∨ RerunMiss C O F τ R e
      ∨ ¬ ∃ κ, Decisive C O F τ κ R (passStart e.1) e.2 := by
  obtain ⟨⟨s1, c⟩, he, -⟩ := hl
  set X := passStart e.1
  have hto : ∃ ps fd, probeOutcome R X.s.tree X.s.edges C.k e.2 = .edge ps fd := by
    unfold edgeAt at he
    split at he
    · exact ⟨_, _, by assumption⟩
    · exact absurd he (by simp)
  obtain ⟨ps, fd, ho⟩ := hto
  have hL : Learned R X.s.tree X.s.edges := hI.1
  have hnd := seedStep_ne_dropped C.K R (pool := X.s.pool) (skip := stepSkip C.K R C.k X.s e.2)
    (forced := C.K.forced) hL ho
  have hns := seedStep_ne_stopped C.K R (pool := X.s.pool) (skip := stepSkip C.K R C.k X.s e.2)
    (forced := C.K.forced) hL ho
  have hk : ∃ κ, (seedStep C.K R X.s.tree X.s.pool X.s.edges (stepSkip C.K R C.k X.s e.2)
      C.K.forced C.k e.2 ps fd).key = some κ := by
    rcases hs : seedStep C.K R X.s.tree X.s.pool X.s.edges (stepSkip C.K R C.k X.s e.2)
      C.K.forced C.k e.2 ps fd with ⟨dd, s1, y, sp⟩ | ⟨s1, sp, dd⟩ | b | _
    · exact ⟨_, rfl⟩
    · exact ⟨_, rfl⟩
    · exact absurd hs (hns b)
    · exact absurd hs hnd
  obtain ⟨κ, hk⟩ := hk
  have hst : stepTest C R X e.2 = some (κ, testStrings C.K R X.s.tree X.s.pool κ.1 κ.2
      (stepSkip C.K R C.k X.s e.2 κ)) := by
    unfold stepTest
    rw [ho]
    simp only [hk, Option.map_some]
  by_cases hv : verdict C.K R X.s.tree X.s.pool κ.1 κ.2 (X.s.tree.paths.length * Fintype.card α)
      (stepSkip C.K R C.k X.s e.2 κ) = .split
  · exact .inl ⟨κ, _, hst, hv⟩
  by_cases hc : CaseAt C O F τ κ R X e.2
  · exact .inr (.inl ⟨κ, hc, hv⟩)
  refine .inr (.inr ?_)
  rintro ⟨κ', ts', hst', hd⟩
  rw [hst] at hst'
  simp only [Option.some.injEq, Prod.mk.injEq] at hst'
  obtain ⟨rfl, rfl⟩ := hst'
  rcases hd with hd | hd
  · exact hc ⟨_, hst, hd⟩
  · exact hv hd

/-- A rerun that misses in the power case makes some key's first test in the power case or
splitting one in the power case that does not split. -/
theorem rerunMiss_first (h0 : C.K.forced = ∅) {n j : ℕ} {A : RoundAcc α}
    {first : List (FreeMonoid α)} {d : Fin n → C.Draws} {e : RoundAcc α × FreeMonoid α}
    (htr : (passStart e.1, e.2) ∈ roundTrace C R n j A first d) (hm : RerunMiss C O F τ R e) :
    ∃ κ A₁ x₁, roundFind C R (Decisive C O F τ κ R) n j A first d = some (A₁, x₁)
      ∧ CaseAt C O F τ κ R A₁ x₁
      ∧ verdict C.K R A₁.s.tree A₁.s.pool κ.1 κ.2 (A₁.s.tree.paths.length * Fintype.card α)
        (stepSkip C.K R C.k A₁.s x₁ κ) ≠ .split := by
  classical
  obtain ⟨κ, hc, hv⟩ := hm
  have hdS : Decisive C O F τ κ R (passStart e.1) e.2 := by
    obtain ⟨ts, h1, h2⟩ := hc; exact ⟨ts, h1, .inl h2⟩
  rcases hf : (roundTrace C R n j A first d).find?
      (fun a => decide (Decisive C O F τ κ R a.1 a.2)) with _ | ⟨A₁, x₁⟩
  · exact absurd (List.find?_eq_none.1 hf _ htr) (by simpa using hdS)
  obtain ⟨hp, as, bs, hl, has⟩ := List.find?_eq_some_iff_append.1 hf
  simp only [decide_eq_true_eq] at hp
  have hpw := (roundTrace_le C R n j A first d).2
  rw [hl, List.pairwise_append, List.pairwise_cons] at hpw
  rw [hl] at htr
  rcases List.mem_append.1 htr with hS | hS
  · exact absurd hdS (by simpa using has _ hS)
  rcases List.mem_cons.1 hS with hS | hS
  · simp only [Prod.mk.injEq] at hS
    obtain ⟨rfl, rfl⟩ := hS
    exact ⟨κ, _, _, by rw [roundFind_eq, hf], hc, hv⟩
  obtain ⟨ts, hst, hd⟩ := hp
  by_cases hsp : verdict C.K R A₁.s.tree A₁.s.pool κ.1 κ.2
      (A₁.s.tree.paths.length * Fintype.card α) (stepSkip C.K R C.k A₁.s x₁ κ) = .split
  · obtain ⟨ts', hst', -⟩ := hc
    exact (no_test_after_split C R h0 hst hsp (hpw.2.1.1 _ hS) hst').elim
  · rcases hd with hd | hd
    · exact ⟨κ, A₁, x₁, by rw [roundFind_eq, hf], ⟨ts, hst, hd⟩, hsp⟩
    · exact absurd hd hsp

end Classify

section Count

variable (C : StrongCfg α) (R : CutReads α)

theorem length_le_filters {β : Type*} {P₁ P₂ P₃ : β → Prop} [DecidablePred P₁]
    [DecidablePred P₂] [DecidablePred P₃] :
    ∀ l : List β, (∀ e ∈ l, P₁ e ∨ P₂ e ∨ P₃ e) →
      l.length ≤ (l.filter fun e => P₁ e).length + (l.filter fun e => P₂ e).length
        + (l.filter fun e => P₃ e).length
  | [], _ => by simp
  | e :: l, h => by
    have ih := length_le_filters l fun e he => h e (List.mem_cons_of_mem _ he)
    have he := h e List.mem_cons_self
    simp only [List.filter_cons, List.length_cons]
    split_ifs <;> (try simp only [List.length_cons] at *) <;> first | omega | (exfalso; simp only [decide_eq_true_eq] at *; tauto)

open scoped Classical in
theorem rerun_splits_count (h0 : C.K.forced = ∅) (hpat : 0 < C.K.patience) :
    ∀ (n j : ℕ) (A : RoundAcc α) (first : List (FreeMonoid α)) (d : Fin n → C.Draws),
      RerunStart C R A first →
      ((rerunFirsts (strongEntries C R n j A first d)).filter fun e => RerunSplits C R e).length
        + A.splits.length ≤ (strongRound C R n j A first d).2.1.splits.length
  | 0, _, _, _, _, _ => by simp [strongEntries, rerunFirsts, strongRound]
  | n + 1, j, A, first, d, hS => by
    have hsp := (strongReading_fst C R j A first (d 0)).2.1
    rw [strongPass_start] at hsp
    have hhead : ((first.head?.map (A, ·)).toList.filter fun e => RerunSplits C R e).length
        + A.splits.length ≤ (strongReading C R j A first (d 0)).1.splits.length := by
      rw [hsp]
      rcases hf : first with _ | ⟨x, xs⟩
      · simpa [passStart] using fold_splits_le C R _ (passStart A)
      obtain ⟨-, hb⟩ := hS.2.2 x (by rw [hf]; rfl)
      have hg : ¬ (C.K.patience ≤ (passStart A).s.streak
          ∨ C.budget (passStart A).s.tree.paths.length ≤ (passStart A).used) := by
        simp only [passStart]; omega
      have hfold : (x :: xs ++ List.ofFn (d 0).1).foldl (passBody C R) (passStart A)
          = (xs ++ List.ofFn (d 0).1).foldl (passBody C R) (strongStep C R (passStart A) x) := by
        rw [List.cons_append, List.foldl_cons]
        rw [show passBody C R (passStart A) x = strongStep C R (passStart A) x by
          unfold passBody; rw [if_neg hg]]
      rw [hfold]
      have hle := fold_splits_le C R (xs ++ List.ofFn (d 0).1) (strongStep C R (passStart A) x)
      simp only [List.head?_cons, Option.map_some, Option.toList_some, List.filter_cons,
        List.filter_nil]
      split_ifs with hr
      · obtain ⟨κ, ts, hst, hv⟩ := of_decide_eq_true hr
        obtain ⟨-, -, r, hr'⟩ := strongStep_of_split C R h0 hst hv
        have : (strongStep C R (passStart A) x).splits.length = A.splits.length + 1 := by
          rw [hr']; simp [passStart]
        simp only [List.length_cons, List.length_nil]
        omega
      · have := strongStep_splits_le C R (passStart A) x
        simp only [List.length_nil]
        exact le_trans (by simp [passStart] at this ⊢; omega) hle
    rw [strongEntries_succ, rerunFirsts_cons, List.filter_append, List.length_append]
    simp only [strongRound]
    rcases hR : strongReading C R j A first (d 0) with ⟨A'', e | lv⟩ <;>
      rw [hR] at hhead <;> simp only [] at hhead ⊢
    · simp only [rerunFirsts, List.filterMap_nil, List.filter_nil, List.length_nil]
      omega
    · have := rerun_splits_count h0 hpat n (j + 1) A'' lv (Fin.tail d)
        (rerunStart_next C R hS.1 hR)
      omega

theorem rerun_used :
    ∀ (n j : ℕ) (A : RoundAcc α) (first : List (FreeMonoid α)) (d : Fin n → C.Draws),
      first.length ≤ C.nr →
      (strongRound C R n j A first d).2.1.used
        ≤ A.used + (strongEntries C R n j A first d).length * (C.nr + C.np)
  | 0, _, _, _, _, _ => by simp [strongRound]
  | n + 1, j, A, first, d, hf => by
    have hu := (strongReading_fst C R j A first (d 0)).2.2
    rw [strongPass_start] at hu
    have hpass := fold_used_le C R (first ++ List.ofFn (d 0).1) (passStart A)
    simp only [List.length_append, List.length_ofFn] at hpass
    have hR1 : (strongReading C R j A first (d 0)).1.used ≤ A.used + (C.nr + C.np) := by
      rw [hu]; simp only [passStart] at hpass ⊢; omega
    rw [strongEntries_succ]
    simp only [strongRound]
    rcases hR : strongReading C R j A first (d 0) with ⟨A'', e | lv⟩ <;>
      rw [hR] at hR1 <;> simp only [] at hR1 ⊢
    · simp only [List.length_singleton, one_mul]; exact hR1
    · have := rerun_used n (j + 1) A'' lv (Fin.tail d) (strongReading_rerun C R hR).2.1
      rw [List.length_cons, Nat.succ_mul]
      omega

theorem rerun_length :
    ∀ (n j : ℕ) (A : RoundAcc α) (first : List (FreeMonoid α)) (d : Fin n → C.Draws),
      (strongEntries C R n j A first d).length
          ≤ (rerunFirsts (strongEntries C R n j A first d)).length + 1
        ∧ (first ≠ [] → (strongEntries C R n j A first d).length
          ≤ (rerunFirsts (strongEntries C R n j A first d)).length)
  | 0, _, _, _, _ => by simp [strongEntries, rerunFirsts]
  | n + 1, j, A, first, d => by
    rw [strongEntries_succ, rerunFirsts_cons, List.length_append, List.length_cons]
    have hh : first ≠ [] → (first.head?.map (A, ·)).toList.length = 1 := by
      intro hne
      rcases first with _ | ⟨x, xs⟩
      · exact absurd rfl hne
      · rfl
    rcases hR : strongReading C R j A first (d 0) with ⟨A'', e | lv⟩ <;> simp only []
    · simp only [rerunFirsts, List.filterMap_nil, List.length_nil]
      refine ⟨by omega, fun hne => by rw [hh hne]⟩
    · have := (rerun_length n (j + 1) A'' lv (Fin.tail d)).2 (strongReading_rerun C R hR).1
      refine ⟨by omega, fun hne => by rw [hh hne]; omega⟩

theorem rerun_exhausted :
    ∀ (n j : ℕ) (A : RoundAcc α) (first : List (FreeMonoid α)) (d : Fin n → C.Draws),
      (strongRound C R n j A first d).1 = .exhausted →
      C.budget (strongRound C R n j A first d).2.1.s.tree.paths.length
        ≤ (strongRound C R n j A first d).2.1.used ∧ 0 < C.nr
  | 0, _, _, _, _, h => by simp [strongRound] at h
  | n + 1, j, A, first, d, h => by
    have hx := strongReading_exhausted C R (j := j) (A := A) (first := first) (y := d 0)
    simp only [strongRound] at h ⊢
    rcases hR : strongReading C R j A first (d 0) with ⟨A'', e | lv⟩ <;>
      rw [hR] at h hx <;> simp only [] at h hx ⊢
    · subst h; exact hx rfl
    · exact rerun_exhausted n (j + 1) A'' lv (Fin.tail d) h

end Count

end OrthoDFA
