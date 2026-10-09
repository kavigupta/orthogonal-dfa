import OrthoDFA.Proofs.Trace
import OrthoDFA.Proofs.EdgeAttempts
import OrthoDFA.Proofs.Spurious

/-!
# The round ends exhausted only after many steps that are not quiet

A pass ends after `patience` quiet steps in a row, so it spends at most `patience` probes per
step that is not quiet and one more `patience`; a reading that reruns a live draw starts with a
step that is not quiet, so it spends at most `patience + 1` per such step. A step that is not
quiet reaches a split test: it splits, at most `|Q|` times without a noisy split; it misses in
the power case, which the power bound covers; its probe has a read off its route; or it is the
residual.
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
      exact hl

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

end Reading

section Steps

variable (C : StrongCfg α) (R : CutReads α)

/-- The step reaches a split test. -/
abbrev Loud (s : RoundAcc α × FreeMonoid α) : Prop := (stepTest C R s.1 s.2).isSome

theorem strongStep_streak (A : RoundAcc α) (x : FreeMonoid α) :
    (strongStep C R A x).s.streak = if Loud C R (A, x) then 0 else A.s.streak + 1 := by
  rw [(strongStep_state C R A x).1]
  unfold Loud probeStepK stepTest
  simp only []
  generalize probeOutcome R A.s.tree A.s.edges C.k x = o
  cases o <;> simp only [] <;> (try split) <;> simp_all [SeedResult.key, closeK]

theorem fold_used_eq : ∀ (probes : List (FreeMonoid α)) (X : RoundAcc α),
    (probes.foldl (passBody C R) X).used = X.used + (passTrace C R X probes).length
  | [], X => by simp [passTrace]
  | x :: xs, X => by
    by_cases hg : C.K.patience ≤ X.s.streak ∨ C.budget X.s.tree.paths.length ≤ X.used
    · rw [fold_of_guard C R X hg, passTrace.eq_2, if_pos hg]; simp
    · simp only [List.foldl_cons]
      rw [show passBody C R X x = strongStep C R X x by unfold passBody; rw [if_neg hg],
        passTrace.eq_2, if_neg hg, fold_used_eq xs, (strongStep_state C R X x).2.1]
      simp only [List.length_cons]
      omega

open scoped Classical in
/-- A pass from a streak within `patience` spends at most `patience` probes per step that is not
quiet, and `patience` more, less the streak it started with. -/
theorem passTrace_len : ∀ (probes : List (FreeMonoid α)) (X : RoundAcc α),
    X.s.streak ≤ C.K.patience →
    (passTrace C R X probes).length + X.s.streak
      ≤ C.K.patience * ((passTrace C R X probes).filter fun s => Loud C R s).length
        + C.K.patience
  | [], X, h => by simp [passTrace]; omega
  | x :: xs, X, h => by
    by_cases hg : C.K.patience ≤ X.s.streak ∨ C.budget X.s.tree.paths.length ≤ X.used
    · rw [passTrace.eq_2, if_pos hg]; simp; omega
    rw [passTrace.eq_2, if_neg hg]
    push Not at hg
    have hs := strongStep_streak C R X x
    by_cases hl : Loud C R (X, x)
    · rw [if_pos hl] at hs
      have ih := passTrace_len xs (strongStep C R X x) (by omega)
      rw [hs] at ih
      rw [List.filter_cons_of_pos (p := fun s => decide (Loud C R s)) (by simpa using hl),
        List.length_cons, List.length_cons,
        Nat.mul_succ]
      omega
    · rw [if_neg hl] at hs
      have ih := passTrace_len xs (strongStep C R X x) (by omega)
      rw [hs] at ih
      rw [List.filter_cons_of_neg (p := fun s => decide (Loud C R s)) (by simpa using hl),
        List.length_cons]
      omega

theorem live_loud {A : RoundAcc α} {x : FreeMonoid α} (hI : StrongInv R A)
    (hl : LiveEdge R A.s.tree A.s.edges C.k (fun _ => False) x) :
    ∃ κ ts, stepTest C R A x = some (κ, ts) := by
  obtain ⟨⟨s1, c⟩, he, -⟩ := hl
  have hto : ∃ ps fd, probeOutcome R A.s.tree A.s.edges C.k x = .edge ps fd := by
    unfold edgeAt at he
    split at he
    · exact ⟨_, _, by assumption⟩
    · exact absurd he (by simp)
  obtain ⟨ps, fd, ho⟩ := hto
  have hnd := seedStep_ne_dropped C.K R (pool := A.s.pool) (skip := stepSkip C.K R C.k A.s x)
    (forced := C.K.forced) hI.1 ho
  have hns := seedStep_ne_stopped C.K R (pool := A.s.pool) (skip := stepSkip C.K R C.k A.s x)
    (forced := C.K.forced) hI.1 ho
  have hk : ∃ κ, (seedStep C.K R A.s.tree A.s.pool A.s.edges (stepSkip C.K R C.k A.s x)
      C.K.forced C.k x ps fd).key = some κ := by
    rcases hs : seedStep C.K R A.s.tree A.s.pool A.s.edges (stepSkip C.K R C.k A.s x)
      C.K.forced C.k x ps fd with ⟨dd, s1, y, sp⟩ | ⟨s1, sp, dd⟩ | b | _
    · exact ⟨_, rfl⟩
    · exact ⟨_, rfl⟩
    · exact absurd hs (hns b)
    · exact absurd hs hnd
  obtain ⟨κ, hk⟩ := hk
  refine ⟨κ, testStrings C.K R A.s.tree A.s.pool κ.1 κ.2 (stepSkip C.K R C.k A.s x κ), ?_⟩
  unfold stepTest
  rw [ho]
  simp only [hk, Option.map_some]

/-- What a reading that reruns `first` starts from. -/
def RerunStart (A : RoundAcc α) (first : List (FreeMonoid α)) : Prop :=
  StrongInv R A ∧ ∀ x, first.head? = some x →
    LiveEdge R A.s.tree A.s.edges C.k (fun _ => False) x
      ∧ A.used < C.budget A.s.tree.paths.length

theorem rerunStart_next {j : ℕ} {A A'' : RoundAcc α} {first lv : List (FreeMonoid α)}
    {y : C.Draws} (hA : StrongInv R A) (h : strongReading C R j A first y = (A'', .inr lv)) :
    RerunStart C R A'' lv := by
  obtain ⟨-, -, hlive, hbud⟩ := strongReading_rerun C R h
  have hI := strongReading_inv C R (j := j) (first := first) (y := y) hA
  rw [h] at hI
  exact ⟨hI, fun x hx => ⟨hlive x (List.mem_of_mem_head? hx), hbud⟩⟩

open scoped Classical in
/-- The round spends at most `patience + 1` probes per step that is not quiet, and `patience`
more unless it starts by rerunning. -/
theorem roundTrace_len (hpat : 0 < C.K.patience) :
    ∀ (n j : ℕ) (A : RoundAcc α) (first : List (FreeMonoid α)) (d : Fin n → C.Draws),
      RerunStart C R A first →
      (roundTrace C R n j A first d).length
          ≤ (C.K.patience + 1) * ((roundTrace C R n j A first d).filter fun s => Loud C R s).length
            + C.K.patience
        ∧ (first ≠ [] → (roundTrace C R n j A first d).length
          ≤ (C.K.patience + 1)
            * ((roundTrace C R n j A first d).filter fun s => Loud C R s).length)
  | 0, _, _, _, _, _ => by simp [roundTrace]
  | n + 1, j, A, first, d, hS => by
    set p := C.K.patience
    set pass := passTrace C R (passStart A) (first ++ List.ofFn (d 0).1)
    have hp1 : pass.length ≤ p * (pass.filter fun s => Loud C R s).length + p := by
      have := passTrace_len C R (first ++ List.ofFn (d 0).1) (passStart A)
        (by simp [passStart])
      simp only [passStart, Nat.add_zero] at this
      exact this
    have hp2 : first ≠ [] → pass.length ≤ (p + 1) * (pass.filter fun s => Loud C R s).length := by
      intro hne
      obtain ⟨x, xs, rfl⟩ := List.exists_cons_of_ne_nil hne
      obtain ⟨hl, hb⟩ := hS.2 x rfl
      have hg : ¬ (C.K.patience ≤ (passStart A).s.streak
          ∨ C.budget (passStart A).s.tree.paths.length ≤ (passStart A).used) := by
        simp only [passStart]; omega
      have hloud : Loud C R (passStart A, x) := by
        obtain ⟨κ, ts, h⟩ := live_loud C R (A := passStart A) hS.1 hl
        unfold Loud; rw [h]; rfl
      have hs := strongStep_streak C R (passStart A) x
      rw [if_pos hloud] at hs
      have ih := passTrace_len C R (xs ++ List.ofFn (d 0).1) (strongStep C R (passStart A) x)
        (by omega)
      rw [hs] at ih
      have hpass : pass = (passStart A, x) :: passTrace C R (strongStep C R (passStart A) x)
          (xs ++ List.ofFn (d 0).1) := by
        simp only [pass, List.cons_append]; rw [passTrace.eq_2, if_neg hg]
      rw [hpass, List.filter_cons_of_pos (p := fun s => decide (Loud C R s)) (by simpa using hloud),
        List.length_cons,
        List.length_cons]
      set k := (List.filter (fun s => decide (Loud C R s)) (passTrace C R
        (strongStep C R (passStart A) x) (xs ++ List.ofFn (d 0).1))).length
      have : p * k + p + 1 ≤ (p + 1) * (k + 1) := by nlinarith
      have hpk : C.K.patience * k = p * k := rfl
      omega
    simp only [roundTrace]
    rcases hR : strongReading C R j A first (d 0) with ⟨A'', e | lv⟩
    · simp only [List.append_nil]
      refine ⟨by nlinarith, fun hne => hp2 hne⟩
    · simp only []
      have hlv := (strongReading_rerun C R hR).1
      obtain ⟨-, ih⟩ := roundTrace_len hpat n (j + 1) A'' lv (Fin.tail d)
        (rerunStart_next C R hS.1 hR)
      have ih := ih hlv
      rw [List.filter_append, List.length_append, List.length_append]
      refine ⟨by nlinarith, fun hne => by have := hp2 hne; nlinarith⟩

theorem round_used_eq :
    ∀ (n j : ℕ) (A : RoundAcc α) (first : List (FreeMonoid α)) (d : Fin n → C.Draws),
      (strongRound C R n j A first d).2.1.used
        = A.used + (roundTrace C R n j A first d).length
  | 0, _, _, _, _ => by simp [strongRound, roundTrace]
  | n + 1, j, A, first, d => by
    have hu := (strongReading_fst C R j A first (d 0)).2.2
    rw [strongPass_start, fold_used_eq] at hu
    simp only [strongRound, roundTrace]
    rcases hR : strongReading C R j A first (d 0) with ⟨A'', e | lv⟩ <;>
      rw [hR] at hu <;> simp only [] at hu ⊢
    · rw [hu]; simp [passStart]
    · rw [round_used_eq n (j + 1) A'' lv (Fin.tail d), hu, List.length_append]
      simp only [passStart]
      omega

end Steps

section Classify

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
variable (C : StrongCfg α) (O : Oracle μ (FreeMonoid α)) (F : Finset (FreeMonoid α)) (τ : ℝ)
  (R : CutReads α)

/-- The step reaches a test that splits. -/
def SplitStep (s : RoundAcc α × FreeMonoid α) : Prop :=
  ∃ κ ts, stepTest C R s.1 s.2 = some (κ, ts)
    ∧ verdict C.K R s.1.s.tree s.1.s.pool κ.1 κ.2 (s.1.s.tree.paths.length * Fintype.card α)
      (stepSkip C.K R C.k s.1.s s.2 κ) = .split

/-- The step reaches a test in the power case that does not split. -/
def MissStep (s : RoundAcc α × FreeMonoid α) : Prop :=
  ∃ κ, CaseAt C O F τ κ R s.1 s.2
    ∧ verdict C.K R s.1.s.tree s.1.s.pool κ.1 κ.2 (s.1.s.tree.paths.length * Fintype.card α)
      (stepSkip C.K R C.k s.1.s s.2 κ) ≠ .split

theorem loud_classify {s : RoundAcc α × FreeMonoid α} (h : Loud C R s) :
    SplitStep C R s ∨ MissStep C O F τ R s ∨ ¬ ∃ κ, Decisive C O F τ κ R s.1 s.2 := by
  unfold Loud at h
  obtain ⟨⟨κ, ts⟩, hst⟩ := Option.isSome_iff_exists.1 h
  by_cases hv : verdict C.K R s.1.s.tree s.1.s.pool κ.1 κ.2
      (s.1.s.tree.paths.length * Fintype.card α) (stepSkip C.K R C.k s.1.s s.2 κ) = .split
  · exact .inl ⟨κ, ts, hst, hv⟩
  by_cases hc : CaseAt C O F τ κ R s.1 s.2
  · exact .inr (.inl ⟨κ, hc, hv⟩)
  refine .inr (.inr ?_)
  rintro ⟨κ', ts', hst', hd⟩
  rw [hst] at hst'
  simp only [Option.some.injEq, Prod.mk.injEq] at hst'
  obtain ⟨rfl, rfl⟩ := hst'
  rcases hd with hd | hd
  · exact hc ⟨_, hst, hd⟩
  · exact hv hd

/-- A step of the round that misses in the power case makes some key's first test in the power
case or splitting one in the power case that does not split. -/
theorem missStep_first (h0 : C.K.forced = ∅) {n j : ℕ} {A : RoundAcc α}
    {first : List (FreeMonoid α)} {d : Fin n → C.Draws} {S : RoundAcc α × FreeMonoid α}
    (htr : S ∈ roundTrace C R n j A first d) (hm : MissStep C O F τ R S) :
    ∃ κ A₁ x₁, roundFind C R (Decisive C O F τ κ R) n j A first d = some (A₁, x₁)
      ∧ CaseAt C O F τ κ R A₁ x₁
      ∧ verdict C.K R A₁.s.tree A₁.s.pool κ.1 κ.2 (A₁.s.tree.paths.length * Fintype.card α)
        (stepSkip C.K R C.k A₁.s x₁ κ) ≠ .split := by
  classical
  obtain ⟨κ, hc, hv⟩ := hm
  have hdS : Decisive C O F τ κ R S.1 S.2 := by
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
  · subst hS
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
    split_ifs <;> (try simp only [List.length_cons] at *) <;>
      first | omega | (exfalso; simp only [decide_eq_true_eq] at *; tauto)

open scoped Classical in
theorem passTrace_splits (h0 : C.K.forced = ∅) : ∀ (probes : List (FreeMonoid α)) (X : RoundAcc α),
    ((passTrace C R X probes).filter fun s => SplitStep C R s).length + X.splits.length
      ≤ (probes.foldl (passBody C R) X).splits.length
  | [], X => by simp [passTrace]
  | x :: xs, X => by
    by_cases hg : C.K.patience ≤ X.s.streak ∨ C.budget X.s.tree.paths.length ≤ X.used
    · rw [passTrace.eq_2, if_pos hg, fold_of_guard C R X hg]; simp
    rw [passTrace.eq_2, if_neg hg]
    simp only [List.foldl_cons]
    rw [show passBody C R X x = strongStep C R X x by unfold passBody; rw [if_neg hg]]
    have ih := passTrace_splits h0 xs (strongStep C R X x)
    simp only [List.filter_cons]
    split_ifs with hsp
    · obtain ⟨κ, ts, hst, hv⟩ := of_decide_eq_true hsp
      obtain ⟨-, -, r, hr⟩ := strongStep_of_split C R h0 hst hv
      rw [hr, List.length_append] at ih
      simp only [List.length_cons] at ih ⊢
      omega
    · have := strongStep_splits_le C R X x
      omega

open scoped Classical in
theorem roundTrace_splits (h0 : C.K.forced = ∅) :
    ∀ (n j : ℕ) (A : RoundAcc α) (first : List (FreeMonoid α)) (d : Fin n → C.Draws),
      ((roundTrace C R n j A first d).filter fun s => SplitStep C R s).length + A.splits.length
        ≤ (strongRound C R n j A first d).2.1.splits.length
  | 0, _, _, _, _ => by simp [roundTrace, strongRound]
  | n + 1, j, A, first, d => by
    have hsp := (strongReading_fst C R j A first (d 0)).2.1
    rw [strongPass_start] at hsp
    have hpass := passTrace_splits C R h0 (first ++ List.ofFn (d 0).1) (passStart A)
    simp only [strongRound, roundTrace]
    rcases hR : strongReading C R j A first (d 0) with ⟨A'', e | lv⟩ <;>
      rw [hR] at hsp <;> simp only [] at hsp ⊢
    · simp only [List.append_nil]; rw [hsp]; simpa [passStart] using hpass
    · have := roundTrace_splits h0 n (j + 1) A'' lv (Fin.tail d)
      rw [List.filter_append, List.length_append]
      rw [hsp] at this
      have hps : (passStart A).splits.length = A.splits.length := rfl
      omega

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

section Spur

variable (C : StrongCfg α) (R : CutReads α)

theorem strongReading_rerun_idx {j : ℕ} {A A'' : RoundAcc α} {first lv : List (FreeMonoid α)}
    {y : C.Draws} (h : strongReading C R j A first y = (A'', .inr lv)) :
    ∃ idx : List (Fin C.nr), idx.Nodup ∧ lv = idx.map y.2.2.1 := by
  unfold strongReading at h
  simp only [] at h
  split_ifs at h <;>
    simp only [Prod.mk.injEq, reduceCtorEq, and_false, Sum.inr.injEq] at h
  all_goals
    obtain ⟨-, rfl⟩ := h
    exact ⟨_, (List.nodup_finRange _).filter _, rfl⟩

theorem passTrace_prefix : ∀ (probes : List (FreeMonoid α)) (X : RoundAcc α),
    (passTrace C R X probes).map Prod.snd <+: probes
  | [], X => by simp [passTrace]
  | x :: xs, X => by
    rw [passTrace.eq_2]
    split_ifs
    · exact List.nil_prefix
    · simp only [List.map_cons]
      exact List.cons_prefix_cons.2 ⟨rfl, passTrace_prefix xs _⟩

open scoped Classical in
/-- Whether some step of `T` satisfying `P` probes `v`. -/
noncomputable def hitBy (P : RoundAcc α × FreeMonoid α → Prop)
    (T : List (RoundAcc α × FreeMonoid α)) (v : FreeMonoid α) : ℕ :=
  if ∃ s ∈ T, P s ∧ s.2 = v then 1 else 0

theorem hitBy_mono {P : RoundAcc α × FreeMonoid α → Prop} {T T' : List (RoundAcc α × FreeMonoid α)}
    (h : ∀ s ∈ T, s ∈ T') (v : FreeMonoid α) : hitBy P T v ≤ hitBy P T' v := by
  unfold hitBy
  split_ifs with h1 h2 <;> try omega
  obtain ⟨s, hs, hp⟩ := h1
  exact absurd ⟨s, h s hs, hp⟩ h2

open scoped Classical in
/-- Steps whose probes are distinct draws: those satisfying `P` are at most the draws a step
satisfying `P` probes. -/
theorem count_le_src {ι : Type*} (P : RoundAcc α × FreeMonoid α → Prop) [DecidablePred P]
    (w : ι → FreeMonoid α) :
    ∀ (L : List (RoundAcc α × FreeMonoid α)) (src : List ι), src.Nodup →
      L.map Prod.snd <+: src.map w →
      (L.filter fun s => P s).length ≤ (src.map fun σ => hitBy P L (w σ)).sum
  | [], _, _, _ => by simp
  | s :: L, src, hnd, hpre => by
    obtain ⟨σ, src', rfl⟩ : ∃ σ src', src = σ :: src' := by
      rcases src with _ | ⟨σ, src'⟩
      · simp at hpre
      · exact ⟨σ, src', rfl⟩
    simp only [List.map_cons, List.cons_prefix_cons] at hpre
    obtain ⟨hv, hpre⟩ := hpre
    have ih := count_le_src P w L src' (List.nodup_cons.1 hnd).2 hpre
    have hmono : (src'.map fun σ => hitBy P L (w σ)).sum
        ≤ (src'.map fun σ => hitBy P (s :: L) (w σ)).sum :=
      List.sum_le_sum fun σ _ => hitBy_mono (fun t ht => List.mem_cons_of_mem _ ht) _
    simp only [List.filter_cons, List.map_cons, List.sum_cons]
    split_ifs with hp
    · have : hitBy P (s :: L) (w σ) = 1 := by
        unfold hitBy
        rw [if_pos ⟨s, List.mem_cons_self, of_decide_eq_true hp, hv⟩]
      simp only [List.length_cons]
      omega
    · omega

omit [Fintype α] [DecidableEq α] in
theorem drawAt_tail {n : ℕ} (d : Fin (n + 1) → C.Draws) (m : Fin n) (σ : Fin C.nr ⊕ Fin C.np) :
    drawAt C (Fin.tail d) (m, σ) = drawAt C d (m.succ, σ) := by
  rcases σ with i | i <;> rfl

open scoped Classical in
/-- The round's steps satisfying `P`, all probing distinct draws, are at most the draws some such
step probes: the reruns of `first`, and every reading's refusal draws and probes. -/
theorem round_count (P : RoundAcc α × FreeMonoid α → Prop) [DecidablePred P] :
    ∀ (n j : ℕ) (A : RoundAcc α) (first : List (FreeMonoid α)) (d : Fin n → C.Draws)
      (idx : List (Fin C.nr)) (br : Fin C.nr → FreeMonoid α), idx.Nodup → first = idx.map br →
      ((roundTrace C R n j A first d).filter fun s => P s).length
        ≤ (idx.map fun i => hitBy P (roundTrace C R n j A first d) (br i)).sum
          + ∑ m : Fin n, ∑ σ : Fin C.nr ⊕ Fin C.np,
            hitBy P (roundTrace C R n j A first d) (drawAt C d (m, σ))
  | 0, _, _, _, _, _, _, _, _ => by simp [roundTrace]
  | n + 1, j, A, first, d, idx, br, hnd, hfirst => by
    set T := roundTrace C R (n + 1) j A first d with hT
    set pass := passTrace C R (passStart A) (first ++ List.ofFn (d 0).1)
    have hpT : ∀ s ∈ pass, s ∈ T := fun s hs => by
      rw [hT]; simp only [roundTrace]; exact List.mem_append_left _ hs
    -- the pass: its probes are `first`'s reruns, then the reading's probes
    let w : Fin C.nr ⊕ Fin C.np → FreeMonoid α := fun σ => match σ with
      | .inl i => br i
      | .inr i => (d 0).1 i
    have hsrc : (idx.map Sum.inl ++ (List.finRange C.np).map Sum.inr).Nodup := by
      refine List.nodup_append.2 ⟨hnd.map Sum.inl_injective,
        (List.nodup_finRange _).map Sum.inr_injective, ?_⟩
      simp
    have hpre : pass.map Prod.snd <+: (idx.map Sum.inl ++ (List.finRange C.np).map Sum.inr).map w
        := by
      have := passTrace_prefix C R (first ++ List.ofFn (d 0).1) (passStart A)
      convert this using 1
      simp only [List.map_append, List.map_map, hfirst, List.ofFn_eq_map]
      rfl
    have hpass := count_le_src P w pass _ hsrc hpre
    simp only [List.map_append, List.sum_append, List.map_map] at hpass
    have hA : ((idx.map fun i => hitBy P pass (br i)).sum)
        ≤ (idx.map fun i => hitBy P T (br i)).sum :=
      List.sum_le_sum fun i _ => hitBy_mono hpT _
    have hB : ((List.finRange C.np).map fun i => hitBy P pass ((d 0).1 i)).sum
        ≤ ∑ i : Fin C.np, hitBy P T (drawAt C d (0, .inr i)) := by
      rw [Fin.sum_univ_def]
      exact List.sum_le_sum fun i _ => hitBy_mono hpT _
    rw [Fin.sum_univ_succ, Fintype.sum_sum_type]
    have hpassB : ((List.finRange C.np).map ((fun σ => hitBy P pass (w σ)) ∘ Sum.inr)).sum
        = ((List.finRange C.np).map fun i => hitBy P pass ((d 0).1 i)).sum := rfl
    have hpassA : (idx.map ((fun σ => hitBy P pass (w σ)) ∘ Sum.inl)).sum
        = (idx.map fun i => hitBy P pass (br i)).sum := rfl
    rw [hpassA, hpassB] at hpass
    rcases hR : strongReading C R j A first (d 0) with ⟨A'', e | lv⟩
    · have hTe : T = pass := by rw [hT]; simp only [roundTrace, hR, List.append_nil]; rfl
      rw [hTe] at hA hB ⊢
      have : ((pass.filter fun s => P s).length) ≤ _ := hpass
      omega
    · have hTe : T = pass ++ roundTrace C R n (j + 1) A'' lv (Fin.tail d) := by
        rw [hT]; simp only [roundTrace, hR]; rfl
      obtain ⟨idx', hnd', hlv⟩ := strongReading_rerun_idx C R hR
      have ih := round_count P n (j + 1) A'' lv (Fin.tail d) idx' (d 0).2.2.1 hnd' hlv
      have hrT : ∀ s ∈ roundTrace C R n (j + 1) A'' lv (Fin.tail d), s ∈ T := fun s hs => by
        rw [hTe]; exact List.mem_append_right _ hs
      have hC : (idx'.map fun i => hitBy P (roundTrace C R n (j + 1) A'' lv (Fin.tail d))
          ((d 0).2.2.1 i)).sum ≤ ∑ i : Fin C.nr, hitBy P T (drawAt C d (0, .inl i)) := by
        rw [← List.sum_toFinset _ hnd']
        exact (Finset.sum_le_sum fun i _ => hitBy_mono hrT _).trans
          (Finset.sum_le_univ_sum_of_nonneg fun _ => Nat.zero_le _)
      have hD : (∑ m : Fin n, ∑ σ : Fin C.nr ⊕ Fin C.np,
          hitBy P (roundTrace C R n (j + 1) A'' lv (Fin.tail d)) (drawAt C (Fin.tail d) (m, σ)))
          ≤ ∑ m : Fin n, ∑ σ : Fin C.nr ⊕ Fin C.np, hitBy P T (drawAt C d (m.succ, σ)) :=
        Finset.sum_le_sum fun m _ => Finset.sum_le_sum fun σ _ => by
          rw [drawAt_tail]; exact hitBy_mono hrT _
      rw [hTe, List.filter_append, List.length_append]
      rw [hTe] at hA hB hC hD
      omega

/-- Markov for a count of events, measurable or not. -/
theorem count_markov {β ι : Type*} [MeasurableSpace β] (P : Measure β) (s : Finset ι)
    (T : ι → Set β) [∀ k p, Decidable (p ∈ T k)] (N : ℕ) :
    (N : ENNReal) * P {p | N ≤ ∑ k ∈ s, if p ∈ T k then 1 else 0} ≤ ∑ k ∈ s, P (T k) := by
  classical
  set T' := fun k => toMeasurable P (T k)
  have hsub : {p | N ≤ ∑ k ∈ s, if p ∈ T k then 1 else 0}
      ⊆ {p | (N : ENNReal) ≤ ∑ k ∈ s, (T' k).indicator 1 p} := by
    intro p hp
    simp only [Set.mem_ofPred_eq] at hp ⊢
    calc (N : ENNReal) ≤ ((∑ k ∈ s, if p ∈ T k then 1 else 0 : ℕ) : ENNReal) := by
          exact_mod_cast hp
      _ = ∑ k ∈ s, if p ∈ T k then (1 : ENNReal) else 0 := by push_cast; rfl
      _ ≤ ∑ k ∈ s, (T' k).indicator 1 p := by
          gcongr with k
          split_ifs with h
          · rw [Set.indicator_of_mem (subset_toMeasurable P (T k) h)]; rfl
          · exact zero_le
  have hm : ∀ k, MeasurableSet (T' k) := fun k => measurableSet_toMeasurable P (T k)
  calc (N : ENNReal) * P {p | N ≤ ∑ k ∈ s, if p ∈ T k then 1 else 0}
      ≤ (N : ENNReal) * P {p | (N : ENNReal) ≤ ∑ k ∈ s, (T' k).indicator 1 p} := by
        gcongr
    _ ≤ ∫⁻ p, ∑ k ∈ s, (T' k).indicator 1 p ∂P :=
        mul_meas_ge_le_lintegral₀ (f := fun p => ∑ k ∈ s, (T' k).indicator 1 p)
          (Finset.measurable_sum s fun k _ => measurable_one.indicator (hm k)).aemeasurable _
    _ = ∑ k ∈ s, P (T k) := by
        rw [lintegral_finsetSum _ fun k _ => measurable_one.indicator (hm k)]
        simp only [lintegral_indicator_one (hm _), T', measure_toMeasurable]

end Spur

section Round

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}

open scoped Classical in
/-- A round that ends exhausted with no noisy split, fewer than `N₁` steps that are not quiet and
probe a draw with a read off its route, and fewer than `N₂` others reaching no test in the power
case or splitting, has some key's first test in the power case or splitting in the power case
and not splitting. -/
theorem exhausted_det {Q : Type*} [Fintype Q] (C : StrongCfg α) (A : DFA (FreeMonoid α) Q)
    (O : Oracle μ (FreeMonoid α)) (F : Finset (FreeMonoid α)) (side : FreeMonoid α → Bool)
    (rep : Q → FreeMonoid α) (R : CutReads α) (seed : List (FreeMonoid α)) (τ : ℝ)
    (N₁ N₂ Rmax : ℕ) (d : Fin Rmax → C.Draws) (h0 : C.K.forced = ∅) (hpat : 0 < C.K.patience)
    (hb : Monotone C.budget)
    (hB : (C.K.patience + 1) * (Fintype.card Q + N₁ + N₂) ≤ C.budget 2)
    (hex : (strongRun C R seed Rmax d).1 = .exhausted)
    (hno : noisySplits R A side rep (strongRun C R seed Rmax d).2.1.splits = 0)
    (h1 : ((roundTrace C R Rmax 0 (startAcc C R seed) [] d).filter fun s =>
      (stepTest C R s.1 s.2).isSome ∧ SpuriousAt R A side rep s.1.s.tree C.k s.2).length < N₁)
    (h2 : ((roundTrace C R Rmax 0 (startAcc C R seed) [] d).filter fun s =>
      (stepTest C R s.1 s.2).isSome ∧ ¬ SpuriousAt R A side rep s.1.s.tree C.k s.2
        ∧ ¬ ∃ κ, Decisive C O F τ κ R s.1 s.2).length < N₂) :
    ∃ κ A₁ x₁, roundFind C R (Decisive C O F τ κ R) Rmax 0 (startAcc C R seed) [] d
        = some (A₁, x₁)
      ∧ CaseAt C O F τ κ R A₁ x₁
      ∧ verdict C.K R A₁.s.tree A₁.s.pool κ.1 κ.2 (A₁.s.tree.paths.length * Fintype.card α)
        (stepSkip C.K R C.k A₁.s x₁ κ) ≠ .split := by
  by_contra hbad
  set T := roundTrace C R Rmax 0 (startAcc C R seed) [] d with hTd
  have hcls : ∀ s ∈ T.filter (fun s => Loud C R s),
      SplitStep C R s
        ∨ ((stepTest C R s.1 s.2).isSome ∧ SpuriousAt R A side rep s.1.s.tree C.k s.2)
        ∨ ((stepTest C R s.1 s.2).isSome ∧ ¬ SpuriousAt R A side rep s.1.s.tree C.k s.2
          ∧ ¬ ∃ κ, Decisive C O F τ κ R s.1 s.2) := by
    intro s hs
    obtain ⟨htr, hl⟩ := List.mem_filter.1 hs
    have hl : Loud C R s := of_decide_eq_true hl
    rcases loud_classify C O F τ R hl with h | h | h
    · exact .inl h
    · exact absurd (missStep_first C O F τ R h0 htr h) hbad
    · by_cases hsp : SpuriousAt R A side rep s.1.s.tree C.k s.2
      · exact .inr (.inl ⟨hl, hsp⟩)
      · exact .inr (.inr ⟨hl, hsp, h⟩)
  have hlen := length_le_filters _ hcls
  have hsT : List.Sublist (T.filter fun s => decide (Loud C R s)) T := List.filter_sublist
  have hs1 := (hsT.filter fun s => decide (SplitStep C R s)).length_le
  have hs2 := (hsT.filter fun s =>
    decide ((stepTest C R s.1 s.2).isSome ∧ SpuriousAt R A side rep s.1.s.tree C.k s.2)).length_le
  have hs3 := (hsT.filter fun s =>
    decide ((stepTest C R s.1 s.2).isSome ∧ ¬ SpuriousAt R A side rep s.1.s.tree C.k s.2
      ∧ ¬ ∃ κ, Decisive C O F τ κ R s.1 s.2)).length_le
  have hsp := roundTrace_splits C R h0 Rmax 0 (startAcc C R seed) [] d
  have hinv := (strongRound_preserves C R (fun _ => True) (fun _ _ _ _ _ _ => trivial)
    (fun _ _ _ _ => trivial) Rmax 0 (startAcc C R seed) [] d (startAcc_inv C R seed) trivial).1
  have hleaves := round_strong_leaves C R A side rep seed Rmax d
  have hused := round_used_eq C R Rmax 0 (startAcc C R seed) [] d
  have hT := (roundTrace_len C R hpat Rmax 0 (startAcc C R seed) [] d
    ⟨startAcc_inv C R seed, by simp⟩).1
  simp only [strongRun] at hex hno hleaves
  simp only [← hTd] at hsp hused hT
  obtain ⟨hex1, -⟩ := rerun_exhausted C R Rmax 0 _ [] d hex
  rw [hno] at hleaves
  have hsl := hinv.2
  have hs0 : (startAcc C R seed).splits.length = 0 := rfl
  have hu0 : (startAcc C R seed).used = 0 := rfl
  set P := (strongRound C R Rmax 0 (startAcc C R seed) [] d).2.1.s.tree.paths.length
  have hb2 : C.budget 2 ≤ C.budget P := hb (by omega)
  set L := (T.filter fun s => Loud C R s).length
  have hK : L + 2 ≤ Fintype.card Q + N₁ + N₂ := by omega
  have hmul := Nat.mul_le_mul_left (C.K.patience + 1) hK
  rw [Nat.mul_add] at hmul
  generalize (C.K.patience + 1) * L = M at hmul hT
  generalize (C.K.patience + 1) * (Fintype.card Q + N₁ + N₂) = W at hmul hB
  omega

open scoped Classical in
theorem round_strong_exhausted : RoundStrongExhausted := by
  intro α _ _ Ω _ μ _ Q _ C A O B F side rep D _ seed τ n ψ₀ N₁ N₂ Rmax h0 hτ hpat hb hN₁ hL hB
    ν R steps
  set φ := ENNReal.ofReal (Real.exp (-τ ^ 2))
  set bad : Ω → (Fin Rmax → C.Draws) → Prop := fun ω d =>
    ∃ κ A₁ x₁, roundFind C (readsAt O B F ω) (Decisive C O F τ κ (readsAt O B F ω)) Rmax 0
        (startAcc C (readsAt O B F ω) seed) [] d = some (A₁, x₁)
      ∧ CaseAt C O F τ κ (readsAt O B F ω) A₁ x₁
      ∧ verdict C.K (readsAt O B F ω) A₁.s.tree A₁.s.pool κ.1 κ.2
          (A₁.s.tree.paths.length * Fintype.card α)
          (stepSkip C.K (readsAt O B F ω) C.k A₁.s x₁ κ) ≠ .split with hbad
  set Sp : RoundAcc α × FreeMonoid α → Ω × (Fin Rmax → C.Draws) → Prop := fun s p =>
    (stepTest C (R p) s.1 s.2).isSome ∧ SpuriousAt (R p) A side rep s.1.s.tree C.k s.2
  set S₁ := {p : Ω × (Fin Rmax → C.Draws) | N₁ ≤ ((steps p).filter fun s => Sp s p).length}
  set S₂ := {p : Ω × (Fin Rmax → C.Draws) | N₂ ≤ ((steps p).filter fun s =>
    (stepTest C (R p) s.1 s.2).isSome ∧ ¬ SpuriousAt (R p) A side rep s.1.s.tree C.k s.2
      ∧ ¬ ∃ κ, Decisive C O F τ κ (R p) s.1 s.2).length}
  set S₃ := {p : Ω × (Fin Rmax → C.Draws) |
    noisySplits (R p) A side rep (strongRun C (R p) seed Rmax p.2).2.1.splits ≠ 0}
  have hsub : {p : Ω × (Fin Rmax → C.Draws) | (strongRun C (R p) seed Rmax p.2).1 = .exhausted}
      ⊆ {p | bad p.1 p.2} ∪ S₁ ∪ S₂ ∪ S₃ := by
    intro p hp
    by_contra hn
    simp only [Set.mem_union, not_or, Set.mem_ofPred_eq, S₁, S₂, S₃, not_le, not_not] at hn
    obtain ⟨⟨⟨hn0, hn1⟩, hn2⟩, hn3⟩ := hn
    exact hn0 (exhausted_det C A O F side rep (R p) seed τ N₁ N₂ Rmax p.2 h0 hpat hb hB hp hn3
      hn1 hn2)
  have hpow : μ.prod ν {p | bad p.1 p.2}
      ≤ φ * ∑' κ, ∫⁻ d, μ {ω | roundFind C (readsAt O B F ω)
          (Decisive C O F τ κ (readsAt O B F ω)) Rmax 0 (startAcc C (readsAt O B F ω) seed) [] d
            ≠ none} ∂ν := by
    have hU : {p : Ω × (Fin Rmax → C.Draws) | bad p.1 p.2}
        ⊆ ⋃ d, {ω | bad ω d} ×ˢ {d} := by
      intro p hp
      exact Set.mem_iUnion.2 ⟨p.2, Set.mem_prod.2 ⟨hp, rfl⟩⟩
    calc μ.prod ν {p | bad p.1 p.2}
        ≤ ∑' d, μ.prod ν ({ω | bad ω d} ×ˢ {d}) := (measure_mono hU).trans (measure_iUnion_le _)
      _ = ∑' d, μ {ω | bad ω d} * ν {d} := by simp only [Measure.prod_prod]
      _ ≤ ∑' d, (φ * ∑' κ, μ {ω | roundFind C (readsAt O B F ω)
          (Decisive C O F τ κ (readsAt O B F ω)) Rmax 0 (startAcc C (readsAt O B F ω) seed) [] d
            ≠ none}) * ν {d} :=
          ENNReal.tsum_le_tsum fun d => by
            gcongr
            exact power_tail C O B F seed τ Rmax d h0 hτ
      _ = φ * ∑' κ, ∑' d, μ {ω | roundFind C (readsAt O B F ω)
          (Decisive C O F τ κ (readsAt O B F ω)) Rmax 0 (startAcc C (readsAt O B F ω) seed) [] d
            ≠ none} * ν {d} := by
          simp only [mul_assoc, ENNReal.tsum_mul_left, ← ENNReal.tsum_mul_right]
          rw [ENNReal.tsum_comm]
      _ = _ := by simp only [lintegral_countable']
  -- the misread steps: Markov over every draw, each bounded by D82's misread chance
  set T : Fin Rmax × (Fin C.nr ⊕ Fin C.np) → Set (Ω × (Fin Rmax → C.Draws)) := fun δ =>
    {p | ∃ s ∈ steps p, Sp s p ∧ s.2 = drawAt C p.2 δ} with hT
  set Bd : Fin Rmax × (Fin C.nr ⊕ Fin C.np) → Set (Ω × (Fin Rmax → C.Draws)) := fun δ =>
    {p | ∃ s ∈ steps p, s.2 = drawAt C p.2 δ
      ∧ drawAt C p.2 δ ∈ BadRoute A O B F side rep C.k n ψ₀ s.1.s.tree} with hBd
  have hcount : S₁ ⊆ {p | N₁ ≤ ∑ δ : Fin Rmax × (Fin C.nr ⊕ Fin C.np),
      if p ∈ T δ then 1 else 0} := by
    intro p hp
    have hc := round_count C (R p) (fun s => Sp s p) Rmax 0 (startAcc C (R p) seed) [] p.2 []
      (fun _ => 1) List.nodup_nil rfl
    simp only [List.map_nil, List.sum_nil, zero_add] at hc
    rw [Set.mem_ofPred_eq, Fintype.sum_prod_type]
    refine le_trans hp (le_trans hc (le_of_eq (Finset.sum_congr rfl fun m _ =>
      Finset.sum_congr rfl fun σ _ => ?_)))
    unfold hitBy
    by_cases h : p ∈ T (m, σ)
    · exact (if_pos h).trans (if_pos h).symm
    · exact (if_neg h).trans (if_neg h).symm
  have hTδ : ∀ δ, μ.prod ν (T δ)
      ≤ ENNReal.ofReal (spurRate A O B F side rep D C.k C.L n ψ₀) + μ.prod ν (Bd δ) := by
    intro δ
    obtain ⟨j, i | i⟩ := δ
    · set g : Ω × (Fin Rmax → C.Draws) → Ω × FreeMonoid α :=
        fun p => (p.1, drawAt C p.2 (j, .inl i)) with hg
      have hgm : Measurable g := measurable_fst.prodMk (f := Prod.fst)
        (g := fun p : Ω × (Fin Rmax → C.Draws) => (p.2 j).2.2.1 i) ((measurable_pi_apply i).comp
        (measurable_fst.comp (measurable_snd.comp (measurable_snd.comp
          ((measurable_pi_apply j).comp measurable_snd)))))
      have hmap : (μ.prod ν).map g = μ.prod D := map_draw C D D (fun y => y.2.2.1 i)
        ((measurable_pi_apply i).comp (measurable_fst.comp (measurable_snd.comp measurable_snd)))
        (map_refusal C D i) j
      have hsub' : T (j, .inl i) ⊆ g ⁻¹' badAll A O B F side rep C.k n ψ₀ ∪ Bd (j, .inl i) := by
        rintro p ⟨s, hs, ⟨-, hsp⟩, he⟩
        rcases spurious_sub A O B F side rep C.k n ψ₀ s.1.s.tree p.1 s.2 hsp with h | h
        · left; simp only [Set.mem_preimage, hg]; rw [← he]; exact h
        · right; exact ⟨s, hs, he, he ▸ h⟩
      refine (measure_mono hsub').trans ((measure_union_le _ _).trans ?_)
      gcongr
      refine (Measure.le_map_apply hgm.aemeasurable _).trans ?_
      rw [hmap, ← ofReal_measureReal (measure_ne_top _ _)]
      exact ENNReal.ofReal_le_ofReal (badAll_le A O B F side rep D C.k C.L n ψ₀ hL)
    · set g : Ω × (Fin Rmax → C.Draws) → Ω × FreeMonoid α :=
        fun p => (p.1, drawAt C p.2 (j, .inr i)) with hg
      have hgm : Measurable g := measurable_fst.prodMk (f := Prod.fst)
        (g := fun p : Ω × (Fin Rmax → C.Draws) => (p.2 j).1 i) ((measurable_pi_apply i).comp
        (measurable_fst.comp ((measurable_pi_apply j).comp measurable_snd)))
      have hmap : (μ.prod ν).map g = μ.prod D := map_draw C D D (fun y => y.1 i)
        ((measurable_pi_apply i).comp measurable_fst) (map_probe C D i) j
      have hsub' : T (j, .inr i) ⊆ g ⁻¹' badAll A O B F side rep C.k n ψ₀ ∪ Bd (j, .inr i) := by
        rintro p ⟨s, hs, ⟨-, hsp⟩, he⟩
        rcases spurious_sub A O B F side rep C.k n ψ₀ s.1.s.tree p.1 s.2 hsp with h | h
        · left; simp only [Set.mem_preimage, hg]; rw [← he]; exact h
        · right; exact ⟨s, hs, he, he ▸ h⟩
      refine (measure_mono hsub').trans ((measure_union_le _ _).trans ?_)
      gcongr
      refine (Measure.le_map_apply hgm.aemeasurable _).trans ?_
      rw [hmap, ← ofReal_measureReal (measure_ne_top _ _)]
      exact ENNReal.ofReal_le_ofReal (badAll_le A O B F side rep D C.k C.L n ψ₀ hL)
  have hS₁ : μ.prod ν S₁ ≤ (∑ δ : Fin Rmax × (Fin C.nr ⊕ Fin C.np),
      (ENNReal.ofReal (spurRate A O B F side rep D C.k C.L n ψ₀) + μ.prod ν (Bd δ))) / N₁ := by
    have hN : (N₁ : ENNReal) ≠ 0 := by exact_mod_cast hN₁.ne'
    rw [ENNReal.le_div_iff_mul_le (.inl hN) (.inl (ENNReal.natCast_ne_top N₁)), mul_comm]
    calc (N₁ : ENNReal) * μ.prod ν S₁
        ≤ (N₁ : ENNReal) * μ.prod ν {p | N₁ ≤ ∑ δ : Fin Rmax × (Fin C.nr ⊕ Fin C.np),
            if p ∈ T δ then 1 else 0} := by gcongr
      _ ≤ ∑ δ : Fin Rmax × (Fin C.nr ⊕ Fin C.np), μ.prod ν (T δ) := count_markov _ _ _ _
      _ ≤ _ := Finset.sum_le_sum fun δ _ => hTδ δ
  calc μ.prod ν {p | (strongRun C (R p) seed Rmax p.2).1 = .exhausted}
      ≤ μ.prod ν ({p | bad p.1 p.2} ∪ S₁ ∪ S₂ ∪ S₃) := measure_mono hsub
    _ ≤ μ.prod ν {p | bad p.1 p.2} + μ.prod ν S₁ + μ.prod ν S₂ + μ.prod ν S₃ := by
        refine (measure_union_le _ _).trans ?_
        gcongr
        refine (measure_union_le _ _).trans ?_
        gcongr
        exact measure_union_le _ _
    _ ≤ _ := by gcongr

end Round

end OrthoDFA
