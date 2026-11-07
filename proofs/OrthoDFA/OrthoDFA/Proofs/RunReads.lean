import OrthoDFA.Proofs.LevelReads
import OrthoDFA.Proofs.KappaReads

/-!
# The strings level 3 reads, counted

A split adds at most one leaf, and re-sifts at most `|leaves|·|Σ|·|leaves|` strings
(`splitReads_card_le`); a probe reads at most `(3 + ⌈log₂ L⌉)|mids T|` (`probeReads_card_clog`).
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

omit [Fintype α] [DecidableEq α] in
/-- A probe at most `L` long reads at most `(3 + ⌈log₂ L⌉)|mids T|` strings. -/
theorem probeReads_card_clog (read : FreeMonoid α → ARU) (T : DTree α) (E : Edges α) (k : ℕ)
    {x : FreeMonoid α} {L : ℕ} (hx : x.toList.length ≤ L) :
    (probeReads read T E k x).card ≤ (3 + Nat.clog 2 L) * (mids T).card := by
  obtain ⟨ps, hps, hsub⟩ := probeReads_sub (rd := read) (T := T) E k hx
  calc (probeReads read T E k x).card ≤ (posReads read T x ps).card := Finset.card_le_card hsub
    _ ≤ ps.card * (mids T).card := posReads_card _ _ _ _
    _ ≤ (3 + Nat.clog 2 L) * (mids T).card := Nat.mul_le_mul_right _ hps

end Random

end OrthoDFA

end
