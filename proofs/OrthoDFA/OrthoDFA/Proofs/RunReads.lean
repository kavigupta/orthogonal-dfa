import OrthoDFA.Proofs.LevelReads
import OrthoDFA.Proofs.KappaReads

/-!
# The strings level 3 reads, counted

A level-3 step adds at most one leaf, so a run's trees stay within `T + 2` leaves whatever is read,
and a run over `T` probes at most `L` long reads at most
`T (probeSifts L (T + 1) + (T + 2)|Σ|(T + 2))` strings counted step by step (`runReadCount_le`);
a probe reads at most `(3 + ⌈log₂ L⌉)|mids T|` (`probeReads_card_clog`).
-/

noncomputable section

namespace OrthoDFA

variable {α : Type*}

/-- Splitting a leaf adds at most one leaf. -/
theorem DTree.length_leaves_splitAt_le (d : FreeMonoid α) :
    ∀ (T : DTree α) (p : List Bool), (T.splitAt d p).leaves.length ≤ T.leaves.length + 1
  | .leaf, [] => by simp [DTree.splitAt, DTree.leaves]
  | .leaf, _ :: _ => by simp [DTree.splitAt, DTree.leaves]
  | .node m r a, [] => by simp [DTree.splitAt, DTree.leaves]
  | .node m r a, false :: p => by
    have ih := DTree.length_leaves_splitAt_le d r p
    simp only [DTree.splitAt, DTree.leaves, List.length_append, List.length_map]
    omega
  | .node m r a, true :: p => by
    have ih := DTree.length_leaves_splitAt_le d a p
    simp only [DTree.splitAt, DTree.leaves, List.length_append, List.length_map]
    omega

variable [Fintype α] [DecidableEq α]

namespace Random

open Eps Kappa Spur

/-! ## Level 3 -/

omit [DecidableEq α] in
/-- A split re-sifts at most `|leaves|·|Σ|·|leaves|` strings. -/
theorem splitReads_card_le (C : Level3Settings) (read : FreeMonoid α → ARU) (s : Level3State α)
    (x : FreeMonoid α) :
    (splitReads read C s x).card
      ≤ s.tree.leaves.length * Fintype.card α * s.tree.leaves.length := by
  unfold splitReads
  rcases probeOrAgree read s.tree s.edges C.walkStart x with _ | ⟨p, c, u, t⟩ | _ | _ | _
  · simp
  · simp only
    split
    · rename_i t₀ w₀ _
      refine (resifts_card s _).trans ?_
      have hmid : (mids (s.tree.splitAt
          (FreeMonoid.of c * s.tree.midfixAt (commonPrefix t t₀)) p)).card
          ≤ s.tree.leaves.length := by
        have h1 := mids_card (s.tree.splitAt
          (FreeMonoid.of c * s.tree.midfixAt (commonPrefix t t₀)) p)
        have h2 := DTree.length_leaves_splitAt_le
          (FreeMonoid.of c * s.tree.midfixAt (commonPrefix t t₀)) s.tree p
        omega
      gcongr
    · simp
  · simp
  · simp
  · simp

/-- A run from `s` over `ys`, with `|leaves(s)| + |ys| ≤ n + 2`, reads at most
`|ys| (probeSifts L (n + 1) + (n + 2)|Σ|(n + 2))` strings. -/
theorem runReadCount_le (read : FreeMonoid α → ARU) (C : Level3Settings) {L : ℕ}
    (X : List (FreeMonoid α)) (n : ℕ) :
    ∀ (ys : List (FreeMonoid α)) (s : Level3State α), (∀ y ∈ ys, y ∈ X) →
      (∀ y ∈ ys, y.toList.length ≤ L) → RunInv read C X s →
      s.tree.leaves.length + ys.length ≤ n + 2 →
      runReadCount read C s ys
        ≤ ys.length * (probeSifts L * (n + 1) + (n + 2) * Fintype.card α * (n + 2))
  | [], _, _, _, _, _ => by simp [runReadCount]
  | x :: ys, s, hX, hL, hs, hn => by
    set B := probeSifts L * (n + 1) + (n + 2) * Fintype.card α * (n + 2) with hB
    rw [List.length_cons] at hn
    have hxL : x.toList.length ≤ L := hL x (by simp)
    have hmid : (mids s.tree).card ≤ n + 1 := by have := mids_card s.tree; omega
    have hprobe : (probeReads read s.tree s.edges C.walkStart x).card ≤ probeSifts L * (n + 1) :=
      (probeReads_card_mids read hxL).trans (Nat.mul_le_mul_left _ hmid)
    have hsplit : (splitReads read C s x).card ≤ (n + 2) * Fintype.card α * (n + 2) :=
      (splitReads_card_le C read s x).trans
        (Nat.mul_le_mul (Nat.mul_le_mul_right _ (by omega)) (by omega))
    have hhead : (probeReads read s.tree s.edges C.walkStart x).card
        + (splitReads read C s x).card ≤ B := Nat.add_le_add hprobe hsplit
    simp only [runReadCount, List.length_cons]
    rcases ha : applyProbe read C s x with s' | e
    · obtain ⟨hs', hlv'⟩ := runInv_step read C (hX x (by simp)) hs ha
      have hrec := runReadCount_le read C X n ys s' (fun y hy => hX y (by simp [hy]))
        (fun y hy => hL y (by simp [hy])) hs' (by omega)
      simp only
      calc _ ≤ B + ys.length * B := Nat.add_le_add hhead hrec
        _ = (ys.length + 1) * B := by ring
    · simp only
      calc _ ≤ B := by omega
        _ ≤ (ys.length + 1) * B := Nat.le_mul_of_pos_left _ (Nat.succ_pos _)

/-- Level 3's run over `T` probes at most `L` long reads at most
`T (probeSifts L (T + 1) + (T + 2)|Σ|(T + 2))` strings. -/
theorem runReadCount_initial_le (read : FreeMonoid α → ARU) (C : Level3Settings) {L T : ℕ}
    (xs : Fin T → FreeMonoid α) (hL : ∀ i, (xs i).toList.length ≤ L) :
    runReadCount read C initialState (List.ofFn xs)
      ≤ T * (probeSifts L * (T + 1) + (T + 2) * Fintype.card α * (T + 2)) := by
  have h0 : (initialState : Level3State α).tree.leaves.length = 2 := by
    simp [initialState, newHypothesis, DTree.leaves]
  have := runReadCount_le read C (L := L) (List.ofFn xs) T (List.ofFn xs) initialState
    (fun _ h => h) (fun y hy => by
      obtain ⟨i, rfl⟩ := List.mem_ofFn.1 hy
      exact hL i) (runInv_start read C _) (by rw [h0, List.length_ofFn]; omega)
  rwa [List.length_ofFn] at this

omit [Fintype α] [DecidableEq α] in
/-- A probe at most `L` long reads at most `(3 + ⌈log₂ L⌉)|mids T|` strings. -/
theorem probeReads_card_clog (read : FreeMonoid α → ARU) (T : DTree α) (E : Edges α) (k : ℕ)
    {x : FreeMonoid α} {L : ℕ} (hx : x.toList.length ≤ L) :
    (probeReads read T E k x).card ≤ (3 + Nat.clog 2 L) * (mids T).card := by
  obtain ⟨ps, hps, hsub⟩ := probeReads_sub (rd := read) (T := T) E k hx
  calc (probeReads read T E k x).card ≤ (posReads read T x ps).card := Finset.card_le_card hsub
    _ ≤ ps.toFinset.card * (mids T).card := posReads_card _ _ _ _
    _ ≤ (3 + Nat.clog 2 L) * (mids T).card := Nat.mul_le_mul_right _ hps

end Random

end OrthoDFA

end
