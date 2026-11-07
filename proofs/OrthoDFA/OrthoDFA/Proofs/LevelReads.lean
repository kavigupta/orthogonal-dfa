import OrthoDFA.Proofs.SpurRounds
import OrthoDFA.Proofs.SpurCount
import OrthoDFA.Proofs.LoopRuns

/-!
# Level 3's result is fixed by a polynomial operational read-set

The probe run reads, over its first `T` steps, exactly the strings in `levelReads`: the union of
the step rounds `rounds6` for `ρ < T (R + 1)`, with `R = (T + 2)|Σ|`. Reads agreeing on
`levelReads` give the same `indecisiveLStarLevel3` (`level3_congr_levelReads`), and `levelReads`
has at most `T (probeSifts L (T + 1) + (T + 2)|Σ|(T + 2))` strings (`levelReads_card_le`), a
polynomial in `T`, `L`, and `|Σ|`.
-/

namespace OrthoDFA

namespace Random

namespace Spur

open Eps Kappa

variable {α : Type*} [Fintype α] [DecidableEq α]

/-- The operational reads of the first `T` probe steps over the draws `xs`. -/
noncomputable def levelReads {T : ℕ} (C : Level3Settings) (xs : Fin T → FreeMonoid α)
    (r : FreeMonoid α → ARU) : Finset (FreeMonoid α) :=
  (Finset.range (T * ((T + 2) * Fintype.card α + 1))).biUnion
    fun ρ => rounds6 C (List.ofFn xs) T ((T + 2) * Fintype.card α) ρ r

/-! ## The real run keeps its states small -/

/-- Every state the run reaches keeps `RunInv` and has at most `i + 2` leaves after `i` probes. -/
theorem runInv_stateAt (C : Level3Settings) (r : FreeMonoid α → ARU) :
    ∀ (xs : List (FreeMonoid α)) (i : ℕ) (s : Level3State α),
      stateAt C r initialState xs i = some s →
      RunInv r C xs s ∧ s.tree.leaves.length ≤ i + 2
  | xs, 0, s, h => by
    simp only [stateAt, Option.some.injEq] at h
    subst h
    refine ⟨runInv_start r C xs, ?_⟩
    have : (initialState : Level3State α).tree.leaves.length = 2 := by
      simp [initialState, newHypothesis, DTree.leaves]
    omega
  | xs, i + 1, s, h => by
    rw [stateAt_succ] at h
    rcases hs₁ : stateAt C r initialState xs i with _ | s₁ <;> rw [hs₁] at h
    · simp at h
    rcases hx : xs[i]? with _ | x <;> simp only [hx, Option.bind_some, Option.bind_none] at h
    · simp at h
    rcases ha : applyProbe r C s₁ x with s' | e <;> rw [ha] at h <;>
      simp only [Sum.elim_inl, Sum.elim_inr, Option.some.injEq, reduceCtorEq] at h
    subst h
    obtain ⟨h1, h2⟩ := runInv_stateAt C r xs i s₁ hs₁
    obtain ⟨h3, h4⟩ := runInv_step r C (List.mem_of_getElem? hx) h1 ha
    exact ⟨h3, by omega⟩

/-- For `j < T` the first `j` states of the run over `List.ofFn xs` are small (`≤ T + 2`). -/
theorem small_run (C : Level3Settings) (r : FreeMonoid α → ARU) {T : ℕ}
    (xs : Fin T → FreeMonoid α) {j : ℕ} (hj : j < T) :
    Small r C (List.ofFn xs) T j := by
  intro i hi s hs
  have := (runInv_stateAt C r (List.ofFn xs) i s hs).2
  omega

/-! ## Reads agreeing on the operational reads give the same run -/

/-- A run is fixed by the reads its steps make: if `r` and `r'` agree on every step's probe along
the `r`-walk, the runs coincide. -/
theorem runProbes_congr_of_steps (C : Level3Settings) {r r' : FreeMonoid α → ARU} :
    ∀ (ys : List (FreeMonoid α)) (s : Level3State α),
      (∀ j t x, stateAt C r s ys j = some t → ys[j]? = some x →
        applyProbe r C t x = applyProbe r' C t x) →
      runProbes r C s ys = runProbes r' C s ys
  | [], _, _ => rfl
  | x₀ :: ys, s, hstep => by
    have h0 : applyProbe r C s x₀ = applyProbe r' C s x₀ :=
      hstep 0 s x₀ (by simp [stateAt]) (by simp)
    simp only [runProbes_nil, runProbes_cons]
    rw [← h0]
    rcases ha : applyProbe r C s x₀ with s' | e
    · refine runProbes_congr_of_steps C ys s' (fun j t x hjt hjx => ?_)
      refine hstep (j + 1) t x ?_ ?_
      · have heq : stateAt C r s (x₀ :: ys) (j + 1) = stateAt C r s' ys j := by
          simp only [stateAt, ha]
        rw [heq]; exact hjt
      · rw [List.getElem?_cons_succ]; exact hjx
    · rfl

/-- Reads agreeing on `levelReads` give the same level-3 result. -/
theorem level3_congr_levelReads {T : ℕ} (C : Level3Settings) (xs : Fin T → FreeMonoid α)
    {r r' : FreeMonoid α → ARU} (h : ∀ p ∈ levelReads C xs r, r p = r' p) :
    indecisiveLStarLevel3 C r xs = indecisiveLStarLevel3 C r' xs := by
  set R := (T + 2) * Fintype.card α with hRdef
  have hmem : ∀ ρ, ρ < T * (R + 1) → ∀ z,
      z ∈ rounds6 C (List.ofFn xs) T R ρ r → z ∈ levelReads C xs r := by
    intro ρ hρ z hz
    rw [levelReads, Finset.mem_biUnion]
    exact ⟨ρ, Finset.mem_range.2 (by rw [← hRdef]; exact hρ), by rw [← hRdef]; exact hz⟩
  have hrun : runProbes r C initialState (List.ofFn xs)
      = runProbes r' C initialState (List.ofFn xs) := by
    refine runProbes_congr_of_steps C (List.ofFn xs) initialState (fun j t x hst hx => ?_)
    obtain ⟨hjlt, -⟩ := List.getElem?_eq_some_iff.1 hx
    rw [List.length_ofFn] at hjlt
    have hInv : Inv r t := (inv_keys_stateAt r C (List.ofFn xs) j t hst).1
    have hKeys : Keys t := (inv_keys_stateAt r C (List.ofFn xs) j t hst).2
    have hSmall : Small r C (List.ofFn xs) T j := small_run C r xs hjlt
    have hleaf : t.tree.leaves.length ≤ T + 2 := hSmall j le_rfl t hst
    have hdiv : ∀ i, i < R + 1 → (j * (R + 1) + i) / (R + 1) = j := fun i hi => by
      rw [Nat.add_comm, Nat.add_mul_div_right _ _ (by omega), Nat.div_eq_of_lt hi, zero_add]
    have hmod : ∀ i, i < R + 1 → (j * (R + 1) + i) % (R + 1) = i := fun i hi => by
      rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hi]
    have hround : ∀ i, i < R + 1 →
        rounds6 C (List.ofFn xs) T R (j * (R + 1) + i) r = stepRound r C t x i := fun i hi => by
      rw [rounds6_eq (s := t) (x := x) (by rw [hdiv i hi]; exact hSmall)
        (by rw [hdiv i hi]; exact hst) (by rw [hdiv i hi]; exact hx), hmod i hi]
    have hub : j * (R + 1) + (R + 1) ≤ T * (R + 1) := by
      have hle : (j + 1) * (R + 1) ≤ T * (R + 1) := Nat.mul_le_mul_right _ hjlt
      have e : (j + 1) * (R + 1) = j * (R + 1) + (R + 1) := by ring
      omega
    refine applyProbe_congr6 C hKeys (fun z hz => ?_) (fun e he z hz => ?_)
    · refine h z (hmem (j * (R + 1) + 0) (by omega) z ?_)
      rw [hround 0 (by omega)]; simpa [stepRound] using hz
    · obtain ⟨i, hi, hie⟩ := List.getElem_of_mem he
      have hiR : i < R := by
        have hlen : (edgeList t).length = t.tree.leaves.length * Fintype.card α :=
          length_edgeList t
        have hle : (edgeList t).length ≤ R := by
          rw [hlen, hRdef]; exact Nat.mul_le_mul_right _ hleaf
        omega
      refine h z (hmem (j * (R + 1) + (i + 1)) (by omega) z ?_)
      rw [hround (i + 1) (by omega)]
      simp only [stepRound, Nat.add_sub_cancel, if_neg (Nat.succ_ne_zero i),
        List.getElem?_eq_getElem hi, hie]
      exact hz
  simp only [indecisiveLStarLevel3_eq, hrun]

/-! ## The operational read-set is polynomial -/

/-- `levelReads` has at most `T (probeSifts L (T + 1) + (T + 2)|Σ|(T + 2))` strings. -/
theorem levelReads_card_le {T L : ℕ} (C : Level3Settings) (xs : Fin T → FreeMonoid α)
    (r : FreeMonoid α → ARU) (hL : ∀ i, (xs i).toList.length ≤ L) :
    (levelReads C xs r).card
      ≤ T * (probeSifts L * (T + 1) + (T + 2) * Fintype.card α * (T + 2)) := by
  classical
  set R := (T + 2) * Fintype.card α with hRdef
  set P := probeSifts L * (T + 1) + (T + 2) * Fintype.card α * (T + 2) with hPdef
  have hbu : (levelReads C xs r).card
      ≤ ∑ ρ ∈ Finset.range (T * (R + 1)), (rounds6 C (List.ofFn xs) T R ρ r).card := by
    rw [levelReads, ← hRdef]
    exact Finset.card_biUnion_le
  have perStepBound : ∀ m, m < T →
      ∑ i ∈ Finset.range (R + 1), (rounds6 C (List.ofFn xs) T R (m * (R + 1) + i) r).card ≤ P := by
    intro m hmT
    have hdiv : ∀ i, i < R + 1 → (m * (R + 1) + i) / (R + 1) = m := fun i hi => by
      rw [Nat.add_comm, Nat.add_mul_div_right _ _ (by omega), Nat.div_eq_of_lt hi, zero_add]
    have hmod : ∀ i, i < R + 1 → (m * (R + 1) + i) % (R + 1) = i := fun i hi => by
      rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hi]
    rcases hs : stateAt C r initialState (List.ofFn xs) m with _ | s
    · refine le_trans (le_of_eq (Finset.sum_eq_zero fun i hi => ?_)) (Nat.zero_le _)
      have hib := Finset.mem_range.1 hi
      unfold rounds6
      rw [hdiv i hib, hs]
      split_ifs <;> simp
    · have hx : (List.ofFn xs)[m]? = some (xs ⟨m, hmT⟩) := by
        rw [List.getElem?_ofFn, dif_pos hmT]
      have hSmall : Small r C (List.ofFn xs) T m := small_run C r xs hmT
      have hInv : Inv r s := (inv_keys_stateAt r C (List.ofFn xs) m s hs).1
      have hleaf : s.tree.leaves.length ≤ T + 2 := hSmall m le_rfl s hs
      have hround : ∀ i, i < R + 1 →
          rounds6 C (List.ofFn xs) T R (m * (R + 1) + i) r
            = stepRound r C s (xs ⟨m, hmT⟩) i := fun i hi => by
        rw [rounds6_eq (s := s) (x := xs ⟨m, hmT⟩) (by rw [hdiv i hi]; exact hSmall)
          (by rw [hdiv i hi]; exact hs) (by rw [hdiv i hi]; exact hx), hmod i hi]
      rw [Finset.sum_congr rfl fun i hi => by rw [hround i (Finset.mem_range.1 hi)]]
      have hmid : (mids s.tree).card ≤ T + 1 := by have := mids_card s.tree; omega
      have hpr : (probeReads r s.tree s.edges C.walkStart (xs ⟨m, hmT⟩)).card
          ≤ probeSifts L * (T + 1) :=
        (probeReads_card_mids r (hL _)).trans (Nat.mul_le_mul_left _ hmid)
      have hsplit : (if (splitOf r C s (xs ⟨m, hmT⟩)).isSome then
          (edgeList s).length * s.tree.leaves.length else 0)
            ≤ (T + 2) * Fintype.card α * (T + 2) := by
        split_ifs
        · have hle : (edgeList s).length ≤ (T + 2) * Fintype.card α := by
            rw [length_edgeList]; exact Nat.mul_le_mul_right _ hleaf
          exact Nat.mul_le_mul hle hleaf
        · exact Nat.zero_le _
      calc ∑ i ∈ Finset.range (R + 1), (stepRound r C s (xs ⟨m, hmT⟩) i).card
          ≤ (probeReads r s.tree s.edges C.walkStart (xs ⟨m, hmT⟩)).card
            + (if (splitOf r C s (xs ⟨m, hmT⟩)).isSome then
                (edgeList s).length * s.tree.leaves.length else 0) :=
            stepRound_card r C hInv R
        _ ≤ probeSifts L * (T + 1) + (T + 2) * Fintype.card α * (T + 2) :=
            Nat.add_le_add hpr hsplit
        _ = P := hPdef.symm
  have key : ∀ m, m ≤ T →
      ∑ ρ ∈ Finset.range (m * (R + 1)), (rounds6 C (List.ofFn xs) T R ρ r).card ≤ m * P := by
    intro m
    induction m with
    | zero => intro _; simp
    | succ m ih =>
      intro hm
      rw [show (m + 1) * (R + 1) = m * (R + 1) + (R + 1) by ring, Finset.sum_range_add]
      calc ∑ ρ ∈ Finset.range (m * (R + 1)), (rounds6 C (List.ofFn xs) T R ρ r).card
            + ∑ i ∈ Finset.range (R + 1),
                (rounds6 C (List.ofFn xs) T R (m * (R + 1) + i) r).card
          ≤ m * P + P := Nat.add_le_add (ih (by omega)) (perStepBound m (by omega))
        _ = (m + 1) * P := by ring
  exact hbu.trans (key T le_rfl)

end Spur

end Random

end OrthoDFA
