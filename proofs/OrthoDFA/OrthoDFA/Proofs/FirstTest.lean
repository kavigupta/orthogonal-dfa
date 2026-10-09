import OrthoDFA.Proofs.Masking

/-!
# The round's first step of a kind, and what decides it

`roundFind` scans the round's steps in order for the first at which `P` holds. Where `P` is
decided by what the round has logged and the step's own reads before its test, the scan and the
log it ends at are decided by the read function's values there.
-/

namespace OrthoDFA

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

section Find

variable (C : StrongCfg α) (R : CutReads α)

/-- The accumulator a pass starts from. -/
def passStart (A : RoundAcc α) : RoundAcc α :=
  { A with s := { A.s with streak := 0, log := A.s.log ∪ A.reads } }

open scoped Classical in
/-- The first step of the pass at which `P` holds, and the accumulator before it. -/
noncomputable def passFind (P : RoundAcc α → FreeMonoid α → Prop) :
    RoundAcc α → List (FreeMonoid α) → Option (RoundAcc α × FreeMonoid α)
  | _, [] => none
  | A, x :: xs =>
    if C.K.patience ≤ A.s.streak ∨ C.budget A.s.tree.paths.length ≤ A.used then none
    else if P A x then some (A, x) else passFind P (strongStep C R A x) xs

/-- The first step of the round at which `P` holds, and the accumulator before it. -/
noncomputable def roundFind (P : RoundAcc α → FreeMonoid α → Prop) :
    (n : ℕ) → ℕ → RoundAcc α → List (FreeMonoid α) → (Fin n → C.Draws)
      → Option (RoundAcc α × FreeMonoid α)
  | 0, _, _, _, _ => none
  | n + 1, j, A, first, d =>
    match passFind C R P (passStart A) (first ++ List.ofFn (d 0).1) with
    | some r => some r
    | none =>
      match strongReading C R j A first (d 0) with
      | (_, .inl _) => none
      | (A', .inr lv) => roundFind P n (j + 1) A' lv (Fin.tail d)

theorem strongPass_start (A : RoundAcc α) (probes : List (FreeMonoid α)) :
    strongPass C R A probes = probes.foldl (passBody C R) (passStart A) :=
  strongPass_eq C R A probes

theorem fold_of_guard (A : RoundAcc α)
    (hg : C.K.patience ≤ A.s.streak ∨ C.budget A.s.tree.paths.length ≤ A.used) :
    ∀ probes : List (FreeMonoid α), probes.foldl (passBody C R) A = A
  | [] => rfl
  | x :: xs => by
    simp only [List.foldl_cons]
    rw [show passBody C R A x = A by unfold passBody; rw [if_pos hg]]
    exact fold_of_guard A hg xs

theorem passFind_some_readLog (P : RoundAcc α → FreeMonoid α → Prop) :
    ∀ (probes : List (FreeMonoid α)) (A A' : RoundAcc α) (x' : FreeMonoid α),
      passFind C R P A probes = some (A', x') →
        readLog C A ⊆ readLog C A' ∧ (WitPool A.s → WitPool A'.s)
  | [], _, _, _, h => by simp [passFind] at h
  | x :: xs, A, A', x', h => by
    unfold passFind at h
    split_ifs at h with hg hp
    · simp only [Option.some.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, rfl⟩ := h
      exact ⟨le_rfl, id⟩
    · obtain ⟨h1, h2⟩ := passFind_some_readLog P xs _ A' x' h
      refine ⟨(strongStep_readLog C R A x).trans h1, fun hw => h2 ?_⟩
      rw [(strongStep_state C R A x).1]; exact probeStepK_witPool C.K R C.k A.s x hw

end Find

section Pre

variable (C : StrongCfg α)

/-- What a step reads before its test. -/
noncomputable def stepPre (F : Finset (FreeMonoid α)) (A : RoundAcc α) (x : FreeMonoid α) :
    Finset (FreeMonoid α) :=
  stepReads C.K F A.s.tree A.s.tree A.s.pool C.k x

theorem stepReads_mono {K : StageKnobs α} {F : Finset (FreeMonoid α)} {t t' : DTree α}
    {pool pool' : List (FreeMonoid α)} {k : ℕ} {x : FreeMonoid α} :
    stepReads K F t t pool k x ⊆ stepReads K F t t' (pool ++ pool') k x := by
  classical
  intro z hz
  unfold stepReads at hz ⊢
  obtain ⟨⟨⟨⟨b, e⟩, m⟩, v⟩, hm, rfl⟩ := Finset.mem_image.1 hz
  refine Finset.mem_image.2 ⟨(((b, e), m), v), ?_, rfl⟩
  simp only [Finset.mem_product, List.mem_toFinset, List.mem_append, Finset.mem_union] at hm ⊢
  obtain ⟨⟨⟨hb, he⟩, hm⟩, hv⟩ := hm
  refine ⟨⟨⟨?_, he⟩, ?_⟩, hv⟩
  · rcases hb with hb | hb
    · exact .inl (.inl hb)
    · exact .inr hb
  · rcases hm with hm | hm <;> exact .inl hm

theorem stepPre_sub_readLog (R : CutReads α) (A : RoundAcc α) (x : FreeMonoid α) :
    readLog C A ∪ ↑(stepPre C R.F A x) ⊆ readLog C (strongStep C R A x) := by
  refine Set.union_subset (strongStep_readLog C R A x) fun z hz => ?_
  left
  rw [(strongStep_state C R A x).1, probeStepK_log]
  exact Finset.mem_union_right _ (stepReads_mono hz)

end Pre

section Mask

variable (C : StrongCfg α) {B : State} {F : Finset (FreeMonoid α)} {f₁ f₂ : FreeMonoid α → ℝ}
variable (P : CutReads α → RoundAcc α → FreeMonoid α → Prop)

/-- What decides `passFind`: the log at the step it finds and that step's reads before its test,
or the log at the pass's end. -/
noncomputable def passE (R : CutReads α) (A : RoundAcc α) (probes : List (FreeMonoid α)) :
    Set (FreeMonoid α) :=
  match passFind C R (P R) A probes with
  | some (A', x') => readLog C A' ∪ ↑(stepPre C R.F A' x')
  | none => readLog C (probes.foldl (passBody C R) A)

theorem passFind_mask
    (hP : ∀ A x, WitPool A.s → (∀ z ∈ readLog C A ∪ ↑(stepPre C F A x), f₁ z = f₂ z) →
      (P (rd B F f₁) A x ↔ P (rd B F f₂) A x)) :
    ∀ (probes : List (FreeMonoid α)) (A : RoundAcc α), WitPool A.s →
      (∀ z ∈ passE C P (rd B F f₁) A probes, f₁ z = f₂ z) →
      passFind C (rd B F f₁) (P (rd B F f₁)) A probes
          = passFind C (rd B F f₂) (P (rd B F f₂)) A probes
        ∧ (passFind C (rd B F f₁) (P (rd B F f₁)) A probes = none →
          probes.foldl (passBody C (rd B F f₁)) A = probes.foldl (passBody C (rd B F f₂)) A)
  | [], A, _, _ => ⟨rfl, fun _ => by simp only [List.foldl_nil]⟩
  | x :: xs, A, hw, h => by
    by_cases hg : C.K.patience ≤ A.s.streak ∨ C.budget A.s.tree.paths.length ≤ A.used
    · refine ⟨by unfold passFind; rw [if_pos hg, if_pos hg], fun _ => ?_⟩
      rw [fold_of_guard C _ A hg, fold_of_guard C _ A hg]
    by_cases hp : P (rd B F f₁) A x
    · have hag : ∀ z ∈ readLog C A ∪ ↑(stepPre C F A x), f₁ z = f₂ z := by
        intro z hz; apply h z
        simp only [passE]; unfold passFind; rw [if_neg hg, if_pos hp]; exact hz
      have hp2 := (hP A x hw hag).1 hp
      refine ⟨by unfold passFind; rw [if_neg hg, if_pos hp, if_neg hg, if_pos hp2], fun hn => ?_⟩
      unfold passFind at hn; rw [if_neg hg, if_pos hp] at hn; exact absurd hn (by simp)
    -- the step is not found: the pass goes on from it
    have hcont : passFind C (rd B F f₁) (P (rd B F f₁)) A (x :: xs)
        = passFind C (rd B F f₁) (P (rd B F f₁)) (strongStep C (rd B F f₁) A x) xs := by
      rw [passFind, if_neg hg, if_neg hp]
    have hfold : (x :: xs).foldl (passBody C (rd B F f₁)) A
        = xs.foldl (passBody C (rd B F f₁)) (strongStep C (rd B F f₁) A x) := by
      simp only [List.foldl_cons]
      rw [show passBody C (rd B F f₁) A x = strongStep C (rd B F f₁) A x by
        unfold passBody; rw [if_neg hg]]
    have hE : passE C P (rd B F f₁) A (x :: xs)
        = passE C P (rd B F f₁) (strongStep C (rd B F f₁) A x) xs := by
      simp only [passE, hcont, hfold]
    rw [hE] at h
    have hsub : readLog C (strongStep C (rd B F f₁) A x)
        ⊆ passE C P (rd B F f₁) (strongStep C (rd B F f₁) A x) xs := by
      simp only [passE]
      rcases hf : passFind C (rd B F f₁) (P (rd B F f₁)) (strongStep C (rd B F f₁) A x) xs
        with _ | ⟨A', x'⟩
      · exact fold_readLog C _ xs _
      · exact (passFind_some_readLog C _ _ xs _ A' x' hf).1.trans Set.subset_union_left
    have hstep : strongStep C (rd B F f₁) A x = strongStep C (rd B F f₂) A x := by
      refine strongStep_mask C hw (fun z hz => h z (hsub (.inl hz))) fun b κ hb hf => ?_
      exact h b (hsub (.inr (.inr ⟨κ, hb, hf⟩)))
    have hp2 : ¬ P (rd B F f₂) A x := fun hp2 => hp ((hP A x hw fun z hz =>
      h z (hsub (stepPre_sub_readLog C (rd B F f₁) A x hz))).2 hp2)
    have hw' : WitPool (strongStep C (rd B F f₁) A x).s := by
      rw [(strongStep_state C _ A x).1]; exact probeStepK_witPool C.K _ C.k A.s x hw
    obtain ⟨ih1, ih2⟩ := passFind_mask hP xs _ hw' h
    have hcont2 : passFind C (rd B F f₂) (P (rd B F f₂)) A (x :: xs)
        = passFind C (rd B F f₂) (P (rd B F f₂)) (strongStep C (rd B F f₂) A x) xs := by
      rw [passFind, if_neg hg, if_neg hp2]
    have hfold2 : (x :: xs).foldl (passBody C (rd B F f₂)) A
        = xs.foldl (passBody C (rd B F f₂)) (strongStep C (rd B F f₂) A x) := by
      simp only [List.foldl_cons]
      rw [show passBody C (rd B F f₂) A x = strongStep C (rd B F f₂) A x by
        unfold passBody; rw [if_neg hg]]
    rw [hcont, hcont2, hfold, hfold2, ← hstep]
    exact ⟨ih1, ih2⟩

/-- What decides `roundFind`: the log at the step it finds and that step's reads before its test,
or the log at the round's end. -/
noncomputable def roundE (R : CutReads α) (n j : ℕ) (A : RoundAcc α)
    (first : List (FreeMonoid α)) (d : Fin n → C.Draws) : Set (FreeMonoid α) :=
  match roundFind C R (P R) n j A first d with
  | some (A', x') => readLog C A' ∪ ↑(stepPre C R.F A' x')
  | none => readLog C (strongRound C R n j A first d).2.1

theorem passStart_readLog (A : RoundAcc α) : readLog C A ⊆ readLog C (passStart A) := by
  rintro z (hz | hz | hz)
  · left; exact Finset.mem_union_left _ hz
  · right; left; exact hz
  · right; right; exact hz

theorem roundFind_readLog (R : CutReads α) :
    ∀ (n j : ℕ) (A : RoundAcc α) (first : List (FreeMonoid α)) (d : Fin n → C.Draws),
      readLog C A ⊆ roundE C P R n j A first d
  | 0, _, _, _, _ => by simp only [roundE, roundFind, strongRound]; exact le_rfl
  | n + 1, j, A, first, d => by
    have hpass := (passStart_readLog C A).trans (fold_readLog C R (first ++ List.ofFn (d 0).1) _)
    rw [← strongPass_start] at hpass
    have hread := hpass.trans (strongReading_fst_acc C R j A first (d 0)).1
    simp only [roundE, roundFind]
    rcases hf : passFind C R (P R) (passStart A) (first ++ List.ofFn (d 0).1) with _ | ⟨A', x'⟩
    · simp only []
      rcases hR : strongReading C R j A first (d 0) with ⟨A'', e | lv⟩ <;> rw [hR] at hread
      · simp only [strongRound, hR]; exact hread
      · simp only [strongRound, hR]
        have := roundFind_readLog R n (j + 1) A'' lv (Fin.tail d)
        simp only [roundE] at this
        exact hread.trans this
    · exact ((passStart_readLog C A).trans
        (passFind_some_readLog C R (P R) _ _ A' x' hf).1).trans Set.subset_union_left

theorem roundFind_mask
    (hP : ∀ A x, WitPool A.s → (∀ z ∈ readLog C A ∪ ↑(stepPre C F A x), f₁ z = f₂ z) →
      (P (rd B F f₁) A x ↔ P (rd B F f₂) A x)) :
    ∀ (n j : ℕ) (A : RoundAcc α) (first : List (FreeMonoid α)) (d : Fin n → C.Draws),
      WitPool A.s → (∀ z ∈ roundE C P (rd B F f₁) n j A first d, f₁ z = f₂ z) →
      roundFind C (rd B F f₁) (P (rd B F f₁)) n j A first d
        = roundFind C (rd B F f₂) (P (rd B F f₂)) n j A first d
  | 0, _, _, _, _, _, _ => rfl
  | n + 1, j, A, first, d, hw, h => by
    have hw0 : WitPool (passStart A).s := hw
    have hsubP : passE C P (rd B F f₁) (passStart A) (first ++ List.ofFn (d 0).1)
        ⊆ roundE C P (rd B F f₁) (n + 1) j A first d := by
      simp only [passE, roundE, roundFind]
      rcases hf : passFind C (rd B F f₁) (P (rd B F f₁)) (passStart A)
          (first ++ List.ofFn (d 0).1) with _ | ⟨A', x'⟩
      · simp only []
        have hread := (strongReading_fst_acc C (rd B F f₁) j A first (d 0)).1
        rw [strongPass_start] at hread
        rcases hR : strongReading C (rd B F f₁) j A first (d 0) with ⟨A'', e | lv⟩ <;>
          rw [hR] at hread
        · simp only [strongRound, hR]; exact hread
        · simp only [strongRound, hR]
          have := roundFind_readLog C P (rd B F f₁) n (j + 1) A'' lv (Fin.tail d)
          simp only [roundE] at this
          exact hread.trans this
      · exact le_rfl
    obtain ⟨h1, h2⟩ := passFind_mask C P hP (first ++ List.ofFn (d 0).1) _ hw0 fun z hz =>
      h z (hsubP hz)
    simp only [roundFind]
    rw [← h1]
    rcases hf : passFind C (rd B F f₁) (P (rd B F f₁)) (passStart A)
        (first ++ List.ofFn (d 0).1) with _ | ⟨A', x'⟩
    · simp only []
      have hpass : strongPass C (rd B F f₁) A (first ++ List.ofFn (d 0).1)
          = strongPass C (rd B F f₂) A (first ++ List.ofFn (d 0).1) := by
        rw [strongPass_start, strongPass_start]; exact h2 hf
      have hRR : readLog C (strongReading C (rd B F f₁) j A first (d 0)).1
          ⊆ roundE C P (rd B F f₁) (n + 1) j A first d := by
        simp only [roundE, roundFind, hf]
        rcases hR : strongReading C (rd B F f₁) j A first (d 0) with ⟨A'', e | lv⟩
        · simp only [strongRound, hR]; exact le_rfl
        · simp only [strongRound, hR]
          have := roundFind_readLog C P (rd B F f₁) n (j + 1) A'' lv (Fin.tail d)
          simp only [roundE] at this
          exact this
      have hread : strongReading C (rd B F f₁) j A first (d 0)
          = strongReading C (rd B F f₂) j A first (d 0) :=
        strongReading_congr C hpass fun z hz =>
          h z (hRR ((strongReading_fst_acc C (rd B F f₁) j A first (d 0)).2 hz))
      rw [← hread]
      have hw' := strongReading_witPool C (rd B F f₁) j A first (d 0) hw
      rcases hR : strongReading C (rd B F f₁) j A first (d 0) with ⟨A'', e | lv⟩
      · rfl
      · have hE : roundE C P (rd B F f₁) n (j + 1) A'' lv (Fin.tail d)
            ⊆ roundE C P (rd B F f₁) (n + 1) j A first d := by
          simp only [roundE, roundFind, hf, hR]
          rcases roundFind C (rd B F f₁) (P (rd B F f₁)) n (j + 1) A'' lv (Fin.tail d)
            with _ | ⟨A₂, x₂⟩
          · simp only [strongRound, hR]; exact le_rfl
          · exact le_rfl
        rw [hR] at hw'
        simp only []
        exact roundFind_mask hP n (j + 1) A'' lv (Fin.tail d) hw' fun z hz => h z (hE hz)
    · rfl

end Mask

end OrthoDFA
