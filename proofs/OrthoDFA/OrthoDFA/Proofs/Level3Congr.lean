import OrthoDFA.Proofs.Level3Reads
import OrthoDFA.Proofs.SpurRounds
import OrthoDFA.Proofs.Allowance
import OrthoDFA.Code.Reads

/-!
# A level-3 run's honest reads

`runReadsFrom r C s xs` is every string the run from `s` over `xs` reads: at each step its probe's
reads and, where it splits, its split's re-sifts, stopping where a probe ends the round. Reads
agreeing there give the same run (`runProbes_congr_runReads`), so level 3's result is fixed by the
oracle's answers at these strings (`indecisiveLStarLevel3_congr_runReads`). While no step grows the
tree past `n + 2` leaves, the run reads at most

    |xs| · (probeSifts L · (n + 2) + (n + 2)·|Σ|·(n + 2))

strings (`runReadsFrom_card_le`), with `n + 2 = |Q| + f + 2` on the loop's success event.
-/

namespace OrthoDFA

namespace Random

open Eps Kappa Spur MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

/-- Reads agreeing on what the run reads give the same run. -/
theorem runProbes_congr_runReads {r r' : FreeMonoid α → ARU} (C : Level3Settings) :
    ∀ (xs : List (FreeMonoid α)) (s : Level3State α), Inv r s → Keys s →
      (∀ z ∈ runReadsFrom r C s xs, r z = r' z) →
      runProbes r C s xs = runProbes r' C s xs
  | [], _, _, _, _ => rfl
  | x :: xs, s, hinv, hk, hz => by
    have hhead : ∀ z ∈ probeReads r s.tree s.edges C.walkStart x ∪ splitReads r C s x, r z = r' z :=
      fun z hzz => hz z (by
        simp only [runReadsFrom]; exact Finset.mem_union_left _ hzz)
    have hstep : applyProbe r C s x = applyProbe r' C s x :=
      Kappa.applyProbe_congr' (C := C) hk hhead
    simp only [runProbes]
    rw [← hstep]
    rcases ha : applyProbe r C s x with s' | e
    · obtain ⟨hinv', hk'⟩ := Spur.inv_keys_step r C hinv hk ha
      have hrec : ∀ z ∈ runReadsFrom r C s' xs, r z = r' z := fun z hzz => hz z (by
        simp only [runReadsFrom, ha]; exact Finset.mem_union_right _ hzz)
      simpa only [ha] using runProbes_congr_runReads C xs s' hinv' hk' hrec
    · rfl

/-- Level 3's result is fixed by the reads of its run. -/
theorem indecisiveLStarLevel3_congr_runReads {r r' : FreeMonoid α → ARU} (C : Level3Settings)
    {P : ℕ} (draws : Fin P → FreeMonoid α)
    (hz : ∀ z ∈ runReadsFrom r C initialState (List.ofFn draws), r z = r' z) :
    indecisiveLStarLevel3 C r draws = indecisiveLStarLevel3 C r' draws := by
  have hrun := runProbes_congr_runReads C (List.ofFn draws) initialState (inv_start (read := r))
    (fun _ _ _ h => by simp [initialState, newHypothesis] at h) hz
  simp only [indecisiveLStarLevel3, hrun]

/-- While no step grows the tree past `n + 2` leaves, the run reads at most
`|xs| · (probeSifts L · (n + 2) + (n + 2)·|Σ|·(n + 2))` strings. -/
theorem runReadsFrom_card_le {r : FreeMonoid α → ARU} (C : Level3Settings) {L n : ℕ} :
    ∀ (xs : List (FreeMonoid α)) (s : Level3State α), (∀ x ∈ xs, x.toList.length ≤ L) →
      s.tree.leaves.length ≤ n + 2 → ¬ GrowsRun r C n s xs →
      (runReadsFrom r C s xs).card
        ≤ xs.length * (probeSifts L * (n + 2) + (n + 2) * Fintype.card α * (n + 2))
  | [], _, _, _, _ => by simp [runReadsFrom]
  | x :: xs, s, hL, hs, hg => by
    have hxL : x.toList.length ≤ L := hL x (by simp)
    have hmid : (mids s.tree).card ≤ n + 2 := le_trans (by
      have := mids_card s.tree; omega) hs
    have hprobe : (probeReads r s.tree s.edges C.walkStart x).card ≤ probeSifts L * (n + 2) :=
      (probeReads_card_mids r hxL).trans (by gcongr)
    have hsplit : (splitReads r C s x).card ≤ (n + 2) * Fintype.card α * (n + 2) :=
      (Kappa.splitReads_card_le C r s x).trans (by gcongr)
    have hhead : (probeReads r s.tree s.edges C.walkStart x ∪ splitReads r C s x).card
        ≤ probeSifts L * (n + 2) + (n + 2) * Fintype.card α * (n + 2) :=
      (Finset.card_union_le _ _).trans (Nat.add_le_add hprobe hsplit)
    simp only [runReadsFrom]
    refine (Finset.card_union_le _ _).trans ?_
    rcases ha : applyProbe r C s x with s' | e
    · have hgs : ¬ GrowsRun r C n s' xs := fun h => hg (Or.inr ⟨s', ha, h⟩)
      have hs' : s'.tree.leaves.length ≤ n + 2 := by
        by_contra hc
        exact hg (Or.inl ⟨s', ha, by omega⟩)
      have hrec := runReadsFrom_card_le C xs s' (fun y hy => hL y (by simp [hy])) hs' hgs
      calc (probeReads r s.tree s.edges C.walkStart x ∪ splitReads r C s x).card
            + (runReadsFrom r C s' xs).card
          ≤ (probeSifts L * (n + 2) + (n + 2) * Fintype.card α * (n + 2))
            + xs.length * (probeSifts L * (n + 2) + (n + 2) * Fintype.card α * (n + 2)) :=
            Nat.add_le_add hhead hrec
        _ = (x :: xs).length
            * (probeSifts L * (n + 2) + (n + 2) * Fintype.card α * (n + 2)) := by
            simp only [List.length_cons]; ring
    · simp only [Finset.card_empty, Nat.add_zero]
      calc (probeReads r s.tree s.edges C.walkStart x ∪ splitReads r C s x).card
          ≤ probeSifts L * (n + 2) + (n + 2) * Fintype.card α * (n + 2) := hhead
        _ ≤ (x :: xs).length
            * (probeSifts L * (n + 2) + (n + 2) * Fintype.card α * (n + 2)) := by
            simp only [List.length_cons]; nlinarith [xs.length]

/-- While no step grows the tree past `n + 2` leaves, the run reads at most
`|xs| · (probeSifts L · (n + 2) + (n + 2)·|Σ|·(n + 2))` strings, counted with multiplicity. -/
theorem runReadCount_le {r : FreeMonoid α → ARU} (C : Level3Settings) {L n : ℕ} :
    ∀ (xs : List (FreeMonoid α)) (s : Level3State α), (∀ x ∈ xs, x.toList.length ≤ L) →
      s.tree.leaves.length ≤ n + 2 → ¬ GrowsRun r C n s xs →
      runReadCount r C s xs
        ≤ xs.length * (probeSifts L * (n + 2) + (n + 2) * Fintype.card α * (n + 2))
  | [], _, _, _, _ => by simp [runReadCount]
  | x :: xs, s, hL, hs, hg => by
    set B := probeSifts L * (n + 2) + (n + 2) * Fintype.card α * (n + 2) with hB
    have hxL : x.toList.length ≤ L := hL x (by simp)
    have hmid : (mids s.tree).card ≤ n + 2 :=
      le_trans (by have := mids_card s.tree; omega) hs
    have hprobe : (probeReads r s.tree s.edges C.walkStart x).card ≤ probeSifts L * (n + 2) :=
      (probeReads_card_mids r hxL).trans (by gcongr)
    have hsplit : (splitReads r C s x).card ≤ (n + 2) * Fintype.card α * (n + 2) :=
      (Kappa.splitReads_card_le C r s x).trans (by gcongr)
    have hhead : (probeReads r s.tree s.edges C.walkStart x).card + (splitReads r C s x).card ≤ B :=
      Nat.add_le_add hprobe hsplit
    simp only [runReadCount]
    rcases ha : applyProbe r C s x with s' | e
    · have hgs : ¬ GrowsRun r C n s' xs := fun h => hg (Or.inr ⟨s', ha, h⟩)
      have hs' : s'.tree.leaves.length ≤ n + 2 := by
        by_contra hc; exact hg (Or.inl ⟨s', ha, by omega⟩)
      have hrec := runReadCount_le C xs s' (fun y hy => hL y (by simp [hy])) hs' hgs
      simp only [ha]
      calc (probeReads r s.tree s.edges C.walkStart x).card + (splitReads r C s x).card
            + runReadCount r C s' xs
          ≤ B + xs.length * B := Nat.add_le_add hhead hrec
        _ = (x :: xs).length * B := by simp only [List.length_cons]; ring
    · simp only [ha]
      calc (probeReads r s.tree s.edges C.walkStart x).card + (splitReads r C s x).card + 0
          ≤ B := by omega
        _ ≤ (x :: xs).length * B := by simp only [List.length_cons]; nlinarith [xs.length]

/-- A step never drops a leaf from the tree. -/
theorem applyProbe_leaves_le {r : FreeMonoid α → ARU} {C : Level3Settings} {s s' : Level3State α}
    {x : FreeMonoid α} (h : applyProbe r C s x = .inl s') :
    s.tree.leaves.length ≤ s'.tree.leaves.length := by
  unfold applyProbe at h
  split at h
  · rw [Sum.inl.injEq] at h; subst h; exact le_rfl
  · split at h
    · rw [Sum.inl.injEq] at h; subst h
      exact DTree.length_leaves_le_splitAt _ _ _
    · unfold applyTests at h
      split at h
      · simp at h
      · rw [Sum.inl.injEq] at h; subst h; exact le_rfl
  · unfold applyTests at h
    split at h
    · simp at h
    · rw [Sum.inl.injEq] at h; subst h; exact le_rfl

/-- The run never drops a leaf from the tree it starts at. -/
theorem runProbes_leaves_mono {r : FreeMonoid α → ARU} {C : Level3Settings} :
    ∀ (xs : List (FreeMonoid α)) (s : Level3State α),
      s.tree.leaves.length ≤ (runProbes r C s xs).1.tree.leaves.length
  | [], s => le_rfl
  | x :: xs, s => by
    rw [runProbes]
    rcases ha : applyProbe r C s x with s' | e
    · simp only [ha]; exact (applyProbe_leaves_le ha).trans (runProbes_leaves_mono xs s')
    · simp only [ha, le_refl]

/-- A run whose final tree has at most `n + 2` leaves never grew past them. -/
theorem not_growsRun_of_runProbes_le {r : FreeMonoid α → ARU} {C : Level3Settings} {n : ℕ} :
    ∀ (xs : List (FreeMonoid α)) (s : Level3State α),
      (runProbes r C s xs).1.tree.leaves.length ≤ n + 2 → ¬ GrowsRun r C n s xs
  | [], s, _ => by simp [GrowsRun]
  | x :: xs, s, h => by
    intro hg
    rw [runProbes] at h
    rcases hg with ⟨s', ha, hlt⟩ | ⟨s', ha, hgr⟩
    · simp only [ha] at h
      have hmono := runProbes_leaves_mono (r := r) (C := C) xs s'
      omega
    · simp only [ha] at h
      exact not_growsRun_of_runProbes_le xs s' h hgr

/-- A tree in `classSet n` has at most `n + 2` leaves. -/
theorem leaves_length_le_of_mem_classSet : ∀ (n : ℕ) (T : DTree α),
    T ∈ classSet n → T.leaves.length ≤ n + 2
  | 0, T, h => by
    rw [classSet, Finset.mem_singleton] at h; subst h; simp [DTree.leaves]
  | n + 1, T, h => by
    classical
    rw [classSet, Finset.mem_union] at h
    rcases h with h | h
    · exact (leaves_length_le_of_mem_classSet n T h).trans (by omega)
    · obtain ⟨T', hT', hsplit⟩ := Finset.mem_biUnion.1 h
      obtain ⟨q, -, rfl⟩ := Finset.mem_image.1 hsplit
      have hT'le := leaves_length_le_of_mem_classSet n T' hT'
      have hle := DTree.length_leaves_splitAt_le
        (FreeMonoid.of q.2.1 * T'.midfixAt (commonPrefix q.2.2.1 q.2.2.2)) T' q.1
      omega

end Random

end OrthoDFA
