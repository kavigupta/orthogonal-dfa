import OrthoDFA.Proofs.Forced
import OrthoDFA.Proofs.TestMiss

/-!
# A split test in the power case, and the round's first such test at a key

`stepTest` is the test a step reaches, with the held-out strings it counts. `CaseAt` holds where
that test is at the key `κ`, counts only strings nothing but `κ`'s own tests has read, and its
sides' mean answers part by `τ` beyond its threshold. `Decisive` holds where, besides, the test
splits instead.
-/

namespace OrthoDFA

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α]
variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}

section Force

variable (C : StrongCfg α) (R : CutReads α) (O : Oracle μ (FreeMonoid α))
  (F : Finset (FreeMonoid α)) (τ : ℝ) (κ : TestKey α)

theorem stepTest_force (A : RoundAcc α) (x : FreeMonoid α) :
    stepTest (C.force κ) R A x = stepTest C R A x := by
  unfold stepTest
  rw [show (C.force κ).k = C.k from rfl]
  split
  · rename_i ps fd _
    rw [show (seedStep (C.force κ).K R A.s.tree A.s.pool A.s.edges
        (stepSkip (C.force κ).K R C.k A.s x) (C.force κ).K.forced C.k x ps fd).key
      = (seedStep C.K R A.s.tree A.s.pool A.s.edges (stepSkip C.K R C.k A.s x) {κ} C.k x ps
        fd).key from rfl, seedStep_key_forced C.K R {κ} C.K.forced]
    rfl
  · rfl

theorem caseAt_force (A : RoundAcc α) (x : FreeMonoid α) :
    CaseAt (C.force κ) O F τ κ R A x ↔ CaseAt C O F τ κ R A x := by
  unfold CaseAt
  rw [stepTest_force]
  rfl

theorem noSplit_of_not_decisive {A : RoundAcc α} {x : FreeMonoid α}
    (h : ¬ Decisive C O F τ κ R A x) : NoSplitAt C R κ A x := by
  intro ps fd ho hk hs
  apply h
  refine ⟨testStrings C.K R A.s.tree A.s.pool κ.1 κ.2 (stepSkip C.K R C.k A.s x κ), ?_, .inr hs⟩
  unfold stepTest
  rw [ho]
  simp only [hk, Option.map_some]

theorem caseAt_decisive {A : RoundAcc α} {x : FreeMonoid α} (h : CaseAt C O F τ κ R A x) :
    Decisive C O F τ κ R A x := by
  obtain ⟨ts, h1, h2⟩ := h
  exact ⟨ts, h1, .inl h2⟩

theorem passFind_force_none (h0 : C.K.forced = ∅) :
    ∀ (probes : List (FreeMonoid α)) (A : RoundAcc α),
      passFind C R (Decisive C O F τ κ R) A probes = none →
      passFind (C.force κ) R (CaseAt (C.force κ) O F τ κ R) A probes = none
        ∧ probes.foldl (passBody (C.force κ) R) A = probes.foldl (passBody C R) A
  | [], _, _ => ⟨rfl, rfl⟩
  | x :: xs, A, h => by
    by_cases hg : C.K.patience ≤ A.s.streak ∨ C.budget A.s.tree.paths.length ≤ A.used
    · have hg' : (C.force κ).K.patience ≤ A.s.streak
          ∨ (C.force κ).budget A.s.tree.paths.length ≤ A.used := hg
      refine ⟨by rw [passFind, if_pos hg'], ?_⟩
      rw [fold_of_guard _ R _ hg', fold_of_guard C R _ hg]
    have hg' : ¬ ((C.force κ).K.patience ≤ A.s.streak
        ∨ (C.force κ).budget A.s.tree.paths.length ≤ A.used) := hg
    rw [passFind, if_neg hg] at h
    split_ifs at h with hd
    have hc : ¬ CaseAt (C.force κ) O F τ κ R A x := fun hc =>
      hd (caseAt_decisive C R O F τ κ ((caseAt_force C R O F τ κ A x).1 hc))
    have hstep := strongStep_force C R κ h0 (noSplit_of_not_decisive C R O F τ κ hd)
    obtain ⟨ih1, ih2⟩ := passFind_force_none h0 xs _ h
    refine ⟨by rw [passFind, if_neg hg', if_neg hc, hstep]; exact ih1, ?_⟩
    simp only [List.foldl_cons]
    rw [show passBody (C.force κ) R A x = strongStep (C.force κ) R A x by
        unfold passBody; rw [if_neg hg'],
      show passBody C R A x = strongStep C R A x by unfold passBody; rw [if_neg hg], hstep]
    exact ih2

theorem passFind_force_some (h0 : C.K.forced = ∅) :
    ∀ (probes : List (FreeMonoid α)) (A A₁ : RoundAcc α) (x₁ : FreeMonoid α),
      passFind C R (Decisive C O F τ κ R) A probes = some (A₁, x₁) →
      CaseAt C O F τ κ R A₁ x₁ →
      passFind (C.force κ) R (CaseAt (C.force κ) O F τ κ R) A probes = some (A₁, x₁)
  | [], _, _, _, h, _ => by simp [passFind] at h
  | x :: xs, A, A₁, x₁, h, hc₁ => by
    by_cases hg : C.K.patience ≤ A.s.streak ∨ C.budget A.s.tree.paths.length ≤ A.used
    · rw [passFind, if_pos hg] at h; exact absurd h (by simp)
    have hg' : ¬ ((C.force κ).K.patience ≤ A.s.streak
        ∨ (C.force κ).budget A.s.tree.paths.length ≤ A.used) := hg
    rw [passFind, if_neg hg] at h
    split_ifs at h with hd
    · simp only [Option.some.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, rfl⟩ := h
      rw [passFind, if_neg hg', if_pos ((caseAt_force C R O F τ κ A x).2 hc₁)]
    · have hc : ¬ CaseAt (C.force κ) O F τ κ R A x := fun hc =>
        hd (caseAt_decisive C R O F τ κ ((caseAt_force C R O F τ κ A x).1 hc))
      have hstep := strongStep_force C R κ h0 (noSplit_of_not_decisive C R O F τ κ hd)
      rw [passFind, if_neg hg', if_neg hc, hstep]
      exact passFind_force_some h0 xs _ A₁ x₁ h hc₁

theorem strongReading_force {j : ℕ} {A : RoundAcc α} {first : List (FreeMonoid α)} {y : C.Draws}
    (hpass : strongPass (C.force κ) R A (first ++ List.ofFn y.1)
      = strongPass C R A (first ++ List.ofFn y.1)) :
    strongReading (C.force κ) R j A first y = strongReading C R j A first y := by
  unfold strongReading
  rw [hpass]
  rfl

theorem roundFind_force_none (h0 : C.K.forced = ∅) :
    ∀ (n j : ℕ) (A : RoundAcc α) (first : List (FreeMonoid α)) (d : Fin n → C.Draws),
      roundFind C R (Decisive C O F τ κ R) n j A first d = none →
      roundFind (C.force κ) R (CaseAt (C.force κ) O F τ κ R) n j A first d = none
  | 0, _, _, _, _, _ => rfl
  | n + 1, j, A, first, d, h => by
    simp only [roundFind] at h ⊢
    rcases hf : passFind C R (Decisive C O F τ κ R) (passStart A) (first ++ List.ofFn (d 0).1)
      with _ | ⟨A₁, x₁⟩
    · rw [hf] at h
      simp only [] at h
      obtain ⟨hn, hfold⟩ := passFind_force_none C R O F τ κ h0 _ (passStart A) hf
      rw [hn]
      simp only []
      have hread := strongReading_force C R κ (j := j) (A := A) (first := first) (y := d 0) (by
        rw [strongPass_start, strongPass_start]; exact hfold)
      rw [hread]
      rcases hR : strongReading C R j A first (d 0) with ⟨A'', e | lv⟩ <;> rw [hR] at h <;>
        simp only [] at h ⊢
      exact roundFind_force_none h0 n (j + 1) A'' lv (Fin.tail d) h
    · rw [hf] at h; exact absurd h (by simp)

theorem roundFind_force_some (h0 : C.K.forced = ∅) :
    ∀ (n j : ℕ) (A : RoundAcc α) (first : List (FreeMonoid α)) (d : Fin n → C.Draws)
      (A₁ : RoundAcc α) (x₁ : FreeMonoid α),
      roundFind C R (Decisive C O F τ κ R) n j A first d = some (A₁, x₁) →
      CaseAt C O F τ κ R A₁ x₁ →
      roundFind (C.force κ) R (CaseAt (C.force κ) O F τ κ R) n j A first d = some (A₁, x₁)
  | 0, _, _, _, _, _, _, h, _ => by simp [roundFind] at h
  | n + 1, j, A, first, d, A₁, x₁, h, hc => by
    simp only [roundFind] at h ⊢
    rcases hf : passFind C R (Decisive C O F τ κ R) (passStart A) (first ++ List.ofFn (d 0).1)
      with _ | ⟨A₂, x₂⟩
    · rw [hf] at h
      simp only [] at h
      obtain ⟨hn, hfold⟩ := passFind_force_none C R O F τ κ h0 _ (passStart A) hf
      rw [hn]
      simp only []
      have hread := strongReading_force C R κ (j := j) (A := A) (first := first) (y := d 0) (by
        rw [strongPass_start, strongPass_start]; exact hfold)
      rw [hread]
      rcases hR : strongReading C R j A first (d 0) with ⟨A'', e | lv⟩ <;> rw [hR] at h <;>
        simp only [] at h ⊢
      · exact absurd h (by simp)
      exact roundFind_force_some h0 n (j + 1) A'' lv (Fin.tail d) A₁ x₁ h hc
    · rw [hf] at h
      simp only [Option.some.injEq] at h
      rw [h] at hf
      rw [passFind_force_some C R O F τ κ h0 _ (passStart A) A₁ x₁ hf hc]

end Force

end OrthoDFA
