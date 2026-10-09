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

/-- `A` with the key `κ` taken as not splitting. -/
def forceAt (κ : TestKey α) (A : RoundAcc α) : RoundAcc α :=
  { A with s := { A.s with forced := {κ} } }

/-- The step's test does not split at `κ`. -/
def NoSplitAt (κ : TestKey α) (A : RoundAcc α) (x : FreeMonoid α) : Prop :=
  ∀ ps fd, probeOutcome R A.s.tree A.s.edges C.k x = .edge ps fd →
    (seedStep C.K R A.s.tree A.s.pool A.s.edges (stepSkip C.K R C.k A.s x) A.s.forced C.k x ps
      fd).key = some κ →
    verdict C.K R A.s.tree A.s.pool κ.1 κ.2 (A.s.tree.paths.length * Fintype.card α)
      (stepSkip C.K R C.k A.s x κ) ≠ .split

theorem probeStepK_force (κ : TestKey α) (k : ℕ) (s : KState α) (x : FreeMonoid α)
    (h0 : s.forced = ∅)
    (hv : ∀ ps fd, probeOutcome R s.tree s.edges k x = .edge ps fd →
      (seedStep C.K R s.tree s.pool s.edges (stepSkip C.K R k s x) ∅ k x ps fd).key = some κ →
      verdict C.K R s.tree s.pool κ.1 κ.2 (s.tree.paths.length * Fintype.card α)
        (stepSkip C.K R k s x κ) ≠ .split) :
    probeStepK C.K R k { s with forced := {κ} } x
      = { probeStepK C.K R k s x with forced := {κ} } := by
  have hsk : stepSkip C.K R k { s with forced := {κ} } x = stepSkip C.K R k s x := rfl
  unfold probeStepK
  simp only [hsk]
  rw [h0]
  split
  · rename_i ps fd ho
    rw [seedStep_force C.K R κ (hv ps fd ho)]
    split <;> simp [closeK]
  · simp [closeK]
  · simp [closeK]

theorem strongStep_force (κ : TestKey α) {A : RoundAcc α} {x : FreeMonoid α}
    (h0 : A.s.forced = ∅) (hv : NoSplitAt C R κ A x) :
    strongStep C R (forceAt κ A) x = forceAt κ (strongStep C R A x) := by
  have hv' : ∀ ps fd, probeOutcome R A.s.tree A.s.edges C.k x = .edge ps fd →
      (seedStep C.K R A.s.tree A.s.pool A.s.edges (stepSkip C.K R C.k A.s x) ∅ C.k x ps fd).key
        = some κ →
      verdict C.K R A.s.tree A.s.pool κ.1 κ.2 (A.s.tree.paths.length * Fintype.card α)
        (stepSkip C.K R C.k A.s x κ) ≠ .split :=
    fun ps fd ho hk => hv ps fd ho (by rw [h0]; exact hk)
  have hp := probeStepK_force C R κ C.k A.s x h0 hv'
  have hsk : stepSkip C.K R C.k { A.s with forced := {κ} } x = stepSkip C.K R C.k A.s x := rfl
  unfold strongStep forceAt
  simp only [hsk, hp]
  rw [h0]
  split
  · rename_i ps fd ho
    rw [seedStep_force C.K R κ (hv' ps fd ho)]
    split <;> rfl
  · rfl

theorem stepTest_force_key (κ : TestKey α) (A : RoundAcc α) (x : FreeMonoid α)
    {ps : List (List Bool)} {fd : ℕ} :
    (seedStep C.K R (forceAt κ A).s.tree (forceAt κ A).s.pool (forceAt κ A).s.edges
        (stepSkip C.K R C.k (forceAt κ A).s x) (forceAt κ A).s.forced C.k x ps fd).key
      = (seedStep C.K R A.s.tree A.s.pool A.s.edges (stepSkip C.K R C.k A.s x) A.s.forced C.k x
          ps fd).key :=
  seedStep_key_forced C.K R _ _

end Step

end OrthoDFA
