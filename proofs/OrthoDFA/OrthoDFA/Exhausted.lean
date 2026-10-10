import OrthoDFA.Spurious

/-!
# How often the round's split tests miss, and how often the round ends exhausted

A split test is in the power case at a key `κ` when it counts only strings nothing but `κ`'s own
tests has read, and its two sides' mean answers part by a margin `τ` beyond its threshold.

`RoundStrongPower`: in the round, which takes no key as not splitting, the chance that at some key
the first test in the power case or splitting is in the power case and does not split is at most
`e^{−τ²}` times the chances, summed over the keys, that there is such a test.

`RoundStrongExhausted`: a pass ends after `patience` quiet steps in a row, and a step is quiet
unless it reaches a split test. Where the budget covers `patience + 1` probes for each of
`|Q| + N₁ + N₂` steps that are not quiet, the round ends exhausted with chance at most that power
term, the misread chance `spurRate` and the `BadRoute` mass of every draw over `N₁`, the chance
that `N₂` steps that are not quiet have no read off their route and reach no test in the power
case or splitting, and the chance of a noisy split, which is not bounded.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

variable {α : Type*} [Fintype α] [DecidableEq α]
variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}

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

end Find

section Test

variable (C : StrongCfg α)

end Test

section Trace

variable (C : StrongCfg α) (R : CutReads α)

/-- The steps a pass takes, each with the accumulator before it. -/
noncomputable def passTrace : RoundAcc α → List (FreeMonoid α) → List (RoundAcc α × FreeMonoid α)
  | _, [] => []
  | A, x :: xs =>
    if C.K.patience ≤ A.s.streak ∨ C.budget A.s.tree.paths.length ≤ A.used then []
    else (A, x) :: passTrace (strongStep C R A x) xs

/-- The steps the round takes, each with the accumulator before it. -/
noncomputable def roundTrace :
    (n : ℕ) → ℕ → RoundAcc α → List (FreeMonoid α) → (Fin n → C.Draws)
      → List (RoundAcc α × FreeMonoid α)
  | 0, _, _, _, _ => []
  | n + 1, j, A, first, d =>
    passTrace C R (passStart A) (first ++ List.ofFn (d 0).1) ++
      match strongReading C R j A first (d 0) with
      | (_, .inl _) => []
      | (A', .inr lv) => roundTrace n (j + 1) A' lv (Fin.tail d)

end Trace

end OrthoDFA
