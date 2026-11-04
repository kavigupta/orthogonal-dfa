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
      RunInv r C xs s ∧ s.tree.leaves.length ≤ i + 2 := by
  sorry

/-- For `j < T` the first `j` states of the run over `List.ofFn xs` are small (`≤ T + 2`). -/
theorem small_run (C : Level3Settings) (r : FreeMonoid α → ARU) {T : ℕ}
    (xs : Fin T → FreeMonoid α) {j : ℕ} (hj : j < T) :
    Small r C (List.ofFn xs) T j := by
  sorry

/-! ## Reads agreeing on the operational reads give the same run -/

/-- A run is fixed by the reads its steps make: if `r` and `r'` agree on every step's probe along
the `r`-walk, the runs coincide. -/
theorem runProbes_congr_of_steps (C : Level3Settings) {r r' : FreeMonoid α → ARU} :
    ∀ (ys : List (FreeMonoid α)) (s : Level3State α),
      (∀ j t x, stateAt C r s ys j = some t → ys[j]? = some x →
        applyProbe r C t x = applyProbe r' C t x) →
      runProbes r C s ys = runProbes r' C s ys := by
  sorry

/-- Reads agreeing on `levelReads` give the same level-3 result. -/
theorem level3_congr_levelReads {T : ℕ} (C : Level3Settings) (xs : Fin T → FreeMonoid α)
    {r r' : FreeMonoid α → ARU} (h : ∀ p ∈ levelReads C xs r, r p = r' p) :
    indecisiveLStarLevel3 C r xs = indecisiveLStarLevel3 C r' xs := by
  sorry

/-! ## The operational read-set is polynomial -/

/-- `levelReads` has at most `T (probeSifts L (T + 1) + (T + 2)|Σ|(T + 2))` strings. -/
theorem levelReads_card_le {T L : ℕ} (C : Level3Settings) (xs : Fin T → FreeMonoid α)
    (r : FreeMonoid α → ARU) (hL : ∀ i, (xs i).toList.length ≤ L) :
    (levelReads C xs r).card
      ≤ T * (probeSifts L * (T + 1) + (T + 2) * Fintype.card α * (T + 2)) := by
  sorry

end Spur

end Random

end OrthoDFA
