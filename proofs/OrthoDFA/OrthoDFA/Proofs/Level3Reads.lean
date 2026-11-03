import OrthoDFA.Proofs.SpurCount

/-!
# All a level-3 run reads

`runReads C xs read` is every string a level-3 run over `xs` reads: the union over its steps of
`rounds'`, step `2j` the `j`-th probe's reads and step `2j + 1` its split's re-sifts. A tree with
`ℓ` leaves has at most `ℓ - 1` midfixes (`mids_card`), a probe at most `L` long reads at most
`probeSifts L` sifts, each a tree midfix (`probeReads_card_mids`), and a split re-sifts at most
`ℓ·|Σ|·ℓ` strings (`splitReads_card_le`), so a step over a tree of at most `ℓ` leaves reads at most

    (probeSifts L + |Σ|·ℓ)·ℓ

strings (`rounds'_card_le`), and the whole run at most `2·|xs|` times that (`runReads_card_le`).
-/

namespace OrthoDFA

variable {α : Type*}

/-- Splitting a leaf adds at most one leaf. -/
theorem DTree.length_leaves_splitAt_le (d : FreeMonoid α) :
    ∀ (T : DTree α) (p : List Bool), (T.splitAt d p).leaves.length ≤ T.leaves.length + 1
  | .leaf, [] => by simp [splitAt, leaves]
  | .leaf, _ :: _ => by simp [splitAt, leaves]
  | .node m r a, [] => by simp [splitAt, leaves]
  | .node m r a, false :: p => by
    have ih := DTree.length_leaves_splitAt_le d r p
    simp only [splitAt, leaves, List.length_append, List.length_map]
    omega
  | .node m r a, true :: p => by
    have ih := DTree.length_leaves_splitAt_le d a p
    simp only [splitAt, leaves, List.length_append, List.length_map]
    omega

/-- Splitting a leaf never drops a leaf. -/
theorem DTree.length_leaves_le_splitAt (d : FreeMonoid α) :
    ∀ (T : DTree α) (p : List Bool), T.leaves.length ≤ (T.splitAt d p).leaves.length
  | .leaf, [] => by simp [splitAt, leaves]
  | .leaf, _ :: _ => by simp [splitAt]
  | .node m r a, [] => by simp [splitAt]
  | .node m r a, false :: p => by
    have ih := DTree.length_leaves_le_splitAt d r p
    simp only [splitAt, leaves, List.length_append, List.length_map]
    omega
  | .node m r a, true :: p => by
    have ih := DTree.length_leaves_le_splitAt d a p
    simp only [splitAt, leaves, List.length_append, List.length_map]
    omega

namespace Random

namespace Kappa

open Eps Spur

variable {α : Type*} [Fintype α] [DecidableEq α] (C : Level3Settings) (read : FreeMonoid α → ARU)

/-- A split re-sifts at most `|leaves|·|Σ|·|leaves|` strings. -/
theorem splitReads_card_le (s : Level3State α) (x : FreeMonoid α) :
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

open scoped Classical in
/-- Every string a level-3 run over `xs` reads. -/
noncomputable def runReads (xs : List (FreeMonoid α)) (read : FreeMonoid α → ARU) :
    Finset (FreeMonoid α) :=
  (Finset.range (2 * xs.length)).biUnion fun i => rounds' C xs i read

theorem subset_runReads {xs : List (FreeMonoid α)} {i : ℕ} (hi : i < 2 * xs.length) :
    rounds' C xs i read ⊆ runReads C xs read := fun _z hz =>
  Finset.mem_biUnion.2 ⟨i, Finset.mem_range.2 hi, hz⟩

/-- A step over a tree of at most `ℓ` leaves, on a probe at most `L` long, reads at most
`(probeSifts L + |Σ|·ℓ)·ℓ` strings. -/
theorem rounds'_card_le {xs : List (FreeMonoid α)} {i L ℓ : ℕ}
    (hx : ∀ x ∈ xs, x.toList.length ≤ L)
    (hℓ : ∀ s, stateAt C read initialState xs (i / 2) = some s → s.tree.leaves.length ≤ ℓ) :
    (rounds' C xs i read).card ≤ (probeSifts L + Fintype.card α * ℓ) * ℓ := by
  unfold rounds'
  rcases hs : stateAt C read initialState xs (i / 2) with _ | s
  · simp
  rcases hxi : xs[i / 2]? with _ | x
  · simp
  have hsℓ : s.tree.leaves.length ≤ ℓ := hℓ s hs
  have hxL : x.toList.length ≤ L := hx x (List.mem_of_getElem? hxi)
  have hmidℓ : (mids s.tree).card ≤ ℓ := by have := mids_card s.tree; omega
  simp only
  split_ifs with hpar
  · refine (probeReads_card_mids read hxL).trans ?_
    calc probeSifts L * (mids s.tree).card
        ≤ probeSifts L * ℓ := by gcongr
      _ ≤ (probeSifts L + Fintype.card α * ℓ) * ℓ := Nat.mul_le_mul (by omega) (le_refl ℓ)
  · refine (splitReads_card_le C read s x).trans ?_
    calc s.tree.leaves.length * Fintype.card α * s.tree.leaves.length
        ≤ ℓ * Fintype.card α * ℓ := by gcongr
      _ = (Fintype.card α * ℓ) * ℓ := by ring
      _ ≤ (probeSifts L + Fintype.card α * ℓ) * ℓ := Nat.mul_le_mul (by omega) (le_refl ℓ)

/-- The whole run reads at most `2·|xs|` steps' worth. -/
theorem runReads_card_le {xs : List (FreeMonoid α)} {L ℓ : ℕ}
    (hx : ∀ x ∈ xs, x.toList.length ≤ L)
    (hℓ : ∀ i s, stateAt C read initialState xs i = some s → s.tree.leaves.length ≤ ℓ) :
    (runReads C xs read).card
      ≤ 2 * xs.length * ((probeSifts L + Fintype.card α * ℓ) * ℓ) := by
  refine Finset.card_biUnion_le.trans ?_
  calc ∑ i ∈ Finset.range (2 * xs.length), (rounds' C xs i read).card
      ≤ ∑ _i ∈ Finset.range (2 * xs.length), (probeSifts L + Fintype.card α * ℓ) * ℓ :=
        Finset.sum_le_sum fun i _ => rounds'_card_le C read hx fun s hs => hℓ (i / 2) s hs
    _ = _ := by rw [Finset.sum_const, Finset.card_range, smul_eq_mul]

end Kappa

end Random

end OrthoDFA
