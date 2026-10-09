import OrthoDFA.Proofs.FirstTest

/-!
# The round with one key's tests taken as not splitting

It runs as the round does until a test at that key would split.
-/

namespace OrthoDFA

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

section Seed

variable (K : StageKnobs α) (R : CutReads α)

theorem seedStep_key_forced {t : DTree α} {pool : List (FreeMonoid α)} {edges : Edges α}
    {skip : TestKey α → FreeMonoid α → Prop} (f₁ f₂ : Set (TestKey α)) {k : ℕ}
    {x : FreeMonoid α} {ps : List (List Bool)} {fd : ℕ} :
    (seedStep K R t pool edges skip f₁ k x ps fd).key
      = (seedStep K R t pool edges skip f₂ k x ps fd).key := by
  unfold seedStep
  simp only []
  split
  · rfl
  split
  · rfl
  split_ifs
  · rfl
  split
  · rfl
  split_ifs
  · rfl
  split
  · rfl
  · rfl
  split_ifs
  all_goals first | rfl | (split <;> rfl)

theorem seedStep_force {t : DTree α} {pool : List (FreeMonoid α)} {edges : Edges α}
    {skip : TestKey α → FreeMonoid α → Prop} {k : ℕ} {x : FreeMonoid α}
    {ps : List (List Bool)} {fd : ℕ} (κ : TestKey α)
    (hv : (seedStep K R t pool edges skip ∅ k x ps fd).key = some κ →
      verdict K R t pool κ.1 κ.2 (t.paths.length * Fintype.card α) (skip κ) ≠ .split) :
    seedStep K R t pool edges skip {κ} k x ps fd = seedStep K R t pool edges skip ∅ k x ps fd := by
  unfold seedStep
  simp only []
  split
  · rfl
  rename_i c hc
  split
  · rfl
  rename_i s2 y he
  split_ifs with h1
  · rfl
  split
  · rfl
  rename_i p hsp
  split_ifs with h2
  · rfl
  split
  · rfl
  · rfl
  rename_i d hd
  push Not at h1 h2
  have hk := seedStep_key (K := K) (R := R) (pool := pool) (skip := skip) (forced := ∅) hc he h1
    hsp h2 hd
  by_cases hκ : (ps.getD (fd - 1 - k) [], d) = κ
  · subst hκ
    have hns := hv hk
    rw [if_pos (Set.mem_singleton _), if_neg (Set.notMem_empty _)]
    split
    · rename_i hsplit; exact absurd hsplit hns
    · rfl
  · rw [if_neg (by simpa using hκ), if_neg (Set.notMem_empty _)]

end Seed

section Step

variable (C : StrongCfg α) (R : CutReads α)

/-- The round's settings with the key `κ` taken as not splitting. -/
abbrev StrongCfg.force (κ : TestKey α) : StrongCfg α := { C with K := { C.K with forced := {κ} } }

/-- The step's test does not split at `κ`. -/
def NoSplitAt (κ : TestKey α) (A : RoundAcc α) (x : FreeMonoid α) : Prop :=
  ∀ ps fd, probeOutcome R A.s.tree A.s.edges C.k x = .edge ps fd →
    (seedStep C.K R A.s.tree A.s.pool A.s.edges (stepSkip C.K R C.k A.s x) C.K.forced C.k x ps
      fd).key = some κ →
    verdict C.K R A.s.tree A.s.pool κ.1 κ.2 (A.s.tree.paths.length * Fintype.card α)
      (stepSkip C.K R C.k A.s x κ) ≠ .split

theorem seedStep_force_cfg (κ : TestKey α) (h0 : C.K.forced = ∅) {s : KState α}
    {x : FreeMonoid α} {ps : List (List Bool)} {fd : ℕ}
    (hv : (seedStep C.K R s.tree s.pool s.edges (stepSkip C.K R C.k s x) C.K.forced C.k x ps
      fd).key = some κ →
      verdict C.K R s.tree s.pool κ.1 κ.2 (s.tree.paths.length * Fintype.card α)
        (stepSkip C.K R C.k s x κ) ≠ .split) :
    seedStep (C.force κ).K R s.tree s.pool s.edges (stepSkip (C.force κ).K R C.k s x)
        (C.force κ).K.forced C.k x ps fd
      = seedStep C.K R s.tree s.pool s.edges (stepSkip C.K R C.k s x) C.K.forced C.k x ps fd := by
  change seedStep C.K R s.tree s.pool s.edges (stepSkip C.K R C.k s x) {κ} C.k x ps fd = _
  rw [h0]
  exact seedStep_force C.K R κ (by rw [← h0]; exact hv)

theorem strongStep_force (κ : TestKey α) (h0 : C.K.forced = ∅) {A : RoundAcc α}
    {x : FreeMonoid α} (hv : NoSplitAt C R κ A x) :
    strongStep (C.force κ) R A x = strongStep C R A x := by
  have hseed : ∀ ps fd, probeOutcome R A.s.tree A.s.edges C.k x = .edge ps fd →
      seedStep (C.force κ).K R A.s.tree A.s.pool A.s.edges (stepSkip (C.force κ).K R C.k A.s x)
          (C.force κ).K.forced C.k x ps fd
        = seedStep C.K R A.s.tree A.s.pool A.s.edges (stepSkip C.K R C.k A.s x) C.K.forced C.k x
          ps fd := fun ps fd ho => seedStep_force_cfg C R κ h0 (hv ps fd ho)
  have hp : probeStepK (C.force κ).K R C.k A.s x = probeStepK C.K R C.k A.s x := by
    unfold probeStepK
    simp only []
    split
    · rename_i ps fd ho
      rw [hseed ps fd ho]
      split <;> rfl
    · rfl
    · rfl
  unfold strongStep
  simp only []
  rw [show (C.force κ).k = C.k from rfl, hp]
  split
  · rename_i ps fd ho
    rw [hseed ps fd ho]
  · rfl

end Step

end OrthoDFA
