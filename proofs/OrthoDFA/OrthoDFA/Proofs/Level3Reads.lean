import OrthoDFA.Proofs.KappaSteps

/-!
# All a level-3 run reads

`runReads C xs read` is every string a level-3 run over `xs` reads: the union over its steps of
`rounds'`, round `2j` the `j`-th probe's reads and round `2j + 1` its split's re-sifts. A tree with
`ℓ` leaves has at most `ℓ - 1` midfixes (`card_mids_add_one_le`), a probe at most `L` long reads at
most `3 + ⌈log₂ L⌉` sifts (`probeReads_card_le`), and a split re-sifts at most `ℓ·|Σ|` of them
(`splitReads_card_le`), so a step over a tree of at most `ℓ` leaves reads at most

    (3 + ⌈log₂ L⌉ + |Σ|·ℓ)·ℓ

strings (`rounds'_card_le`), and the whole run at most `2·|xs|` times that.
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

/-- A tree's midfixes number below its leaves. -/
theorem card_mids_add_one_le : ∀ T : DTree α, (mids T).card + 1 ≤ T.leaves.length
  | .leaf => by simp [mids, DTree.leaves]
  | .node m r a => by
    classical
    have hr := card_mids_add_one_le r
    have ha := card_mids_add_one_le a
    have hm : (mids (DTree.node m r a)).card ≤ 1 + ((mids r).card + (mids a).card) := by
      simp only [mids]
      refine (Finset.card_insert_le _ _).trans ?_
      have := Finset.card_union_le (mids r) (mids a)
      omega
    have hl : (DTree.node m r a).leaves.length = r.leaves.length + a.leaves.length := by
      simp [DTree.leaves]
    omega

namespace Kappa

open Eps

variable {α : Type*} [Fintype α] [DecidableEq α] (C : Level3Settings)

/-- A probe at most `L` long reads at most `3 + ⌈log₂ L⌉` sifts, each a tree midfix. -/
theorem probeReads_card_le {rd : FreeMonoid α → ARU} {T : DTree α} {E : Edges α} {k : ℕ}
    {x : FreeMonoid α} {L : ℕ} (hx : x.toList.length ≤ L) :
    (probeReads rd T E k x).card ≤ (3 + Nat.clog 2 L) * (mids T).card := by
  obtain ⟨ps, hps, hsub⟩ := probeReads_sub (rd := rd) (T := T) (x := x) E k hx
  calc (probeReads rd T E k x).card
      ≤ (posReads rd T x ps).card := Finset.card_le_card hsub
    _ ≤ ps.toFinset.card * (mids T).card := posReads_card rd T x ps
    _ ≤ (3 + Nat.clog 2 L) * (mids T).card := by gcongr

/-- A split re-sifts at most `|leaves|·|Σ|·|leaves|` strings. -/
theorem splitReads_card_le (rd : FreeMonoid α → ARU) (s : Level3State α) (x : FreeMonoid α) :
    (splitReads rd C s x).card
      ≤ s.tree.leaves.length * Fintype.card α * s.tree.leaves.length := by
  have key : ∀ T' : DTree α, T'.leaves.length ≤ s.tree.leaves.length + 1 →
      (resifts s T').card ≤ s.tree.leaves.length * Fintype.card α * s.tree.leaves.length := by
    intro T' hT'
    refine (resifts_card s T').trans ?_
    have hmid : (mids T').card ≤ s.tree.leaves.length := by
      have := card_mids_add_one_le T'; omega
    gcongr
  unfold splitReads
  split
  · split
    · exact key _ (DTree.length_leaves_splitAt_le _ _ _)
    · simp
  · simp

open scoped Classical in
/-- Every string a level-3 run over `xs` reads. -/
noncomputable def runReads (xs : List (FreeMonoid α)) (read : FreeMonoid α → ARU) :
    Finset (FreeMonoid α) :=
  (Finset.range (2 * xs.length)).biUnion fun i => rounds' C xs i read

theorem subset_runReads {xs : List (FreeMonoid α)} {read : FreeMonoid α → ARU} {i : ℕ}
    (hi : i < 2 * xs.length) : rounds' C xs i read ⊆ runReads C xs read := fun _z hz =>
  Finset.mem_biUnion.2 ⟨i, Finset.mem_range.2 hi, hz⟩

/-- A step over a tree of at most `ℓ` leaves, on a probe at most `L` long, reads at most
`(3 + ⌈log₂ L⌉ + |Σ|·ℓ)·ℓ` strings. -/
theorem rounds'_card_le {xs : List (FreeMonoid α)} {read : FreeMonoid α → ARU} {i L ℓ : ℕ}
    (hx : ∀ x ∈ xs, x.toList.length ≤ L)
    (hℓ : ∀ s, stateAt C read initialState xs (i / 2) = some s → s.tree.leaves.length ≤ ℓ) :
    (rounds' C xs i read).card ≤ (3 + Nat.clog 2 L + Fintype.card α * ℓ) * ℓ := by
  unfold rounds'
  rcases hs : stateAt C read initialState xs (i / 2) with _ | s
  · simp
  rcases hxi : xs[i / 2]? with _ | x
  · simp
  have hsℓ : s.tree.leaves.length ≤ ℓ := hℓ s hs
  have hxL : x.toList.length ≤ L := hx x (List.mem_of_getElem? hxi)
  have hmidℓ : (mids s.tree).card ≤ ℓ := le_trans (by have := card_mids_add_one_le s.tree; omega) hsℓ
  simp only
  split_ifs with hpar
  · refine (probeReads_card_le hxL).trans ?_
    calc (3 + Nat.clog 2 L) * (mids s.tree).card
        ≤ (3 + Nat.clog 2 L) * ℓ := by gcongr
      _ ≤ (3 + Nat.clog 2 L + Fintype.card α * ℓ) * ℓ :=
          Nat.mul_le_mul (by omega) (le_refl ℓ)
  · refine (splitReads_card_le C read s x).trans ?_
    calc s.tree.leaves.length * Fintype.card α * s.tree.leaves.length
        ≤ ℓ * Fintype.card α * ℓ := by gcongr
      _ = (Fintype.card α * ℓ) * ℓ := by ring
      _ ≤ (3 + Nat.clog 2 L + Fintype.card α * ℓ) * ℓ :=
          Nat.mul_le_mul (by omega) (le_refl ℓ)

/-- The whole run reads at most `2·|xs|` steps' worth. -/
theorem runReads_card_le {xs : List (FreeMonoid α)} {read : FreeMonoid α → ARU} {L ℓ : ℕ}
    (hx : ∀ x ∈ xs, x.toList.length ≤ L)
    (hℓ : ∀ i s, stateAt C read initialState xs i = some s → s.tree.leaves.length ≤ ℓ) :
    (runReads C xs read).card ≤ 2 * xs.length * ((3 + Nat.clog 2 L + Fintype.card α * ℓ) * ℓ) := by
  refine (Finset.card_biUnion_le).trans ?_
  calc ∑ i ∈ Finset.range (2 * xs.length), (rounds' C xs i read).card
      ≤ ∑ _i ∈ Finset.range (2 * xs.length), (3 + Nat.clog 2 L + Fintype.card α * ℓ) * ℓ :=
        Finset.sum_le_sum fun i _ => rounds'_card_le C hx fun s hs => hℓ (i / 2) s hs
    _ = _ := by rw [Finset.sum_const, Finset.card_range, smul_eq_mul]

end Kappa

end Random

end OrthoDFA
