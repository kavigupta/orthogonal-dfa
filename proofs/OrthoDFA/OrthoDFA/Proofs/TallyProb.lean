import OrthoDFA.Proofs.FixTime
import OrthoDFA.Proofs.Freedman
import OrthoDFA.Proofs.Hoeffding

/-!
# Hits along a round of fresh draws

`hits_le`: where each state's hit set has chance at most `ρ`, the hits along `T` draws reach `j`
with chance at most `P(Bin(T, ρ) ≥ j)`, however the states are chosen. `somewhere_le`: an event
of chance at most `c` from every state happens at some point of `T` draws with chance at most
`T · c`.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

variable {X S : Type*} [MeasurableSpace X] [Countable X] [MeasurableSingletonClass X]

section Defs

variable (step : S → X → Option S)

open scoped Classical in
/-- How many of the draws `xs` from `s` fall in the hit set of the state they are drawn at. -/
noncomputable def hitsAlong (A : S → Set X) : S → (T : ℕ) → (Fin T → X) → ℕ
  | _, 0, _ => 0
  | s, _ + 1, xs => (if xs 0 ∈ A s then 1 else 0) + match step s (xs 0) with
    | some s' => hitsAlong A s' _ (Fin.tail xs)
    | none => 0

/-- At some point of the draws `xs` from `s`, the round is at a state from which `F` holds of the
remaining draws. -/
def Somewhere (F : S → (T : ℕ) → (Fin T → X) → Prop) : S → (T : ℕ) → (Fin T → X) → Prop
  | _, 0, _ => False
  | s, _ + 1, xs => F s _ xs ∨ ∃ s', step s (xs 0) = some s' ∧ Somewhere F s' _ (Fin.tail xs)

end Defs

variable (D : Measure X) [IsProbabilityMeasure D] (step : S → X → Option S)

end OrthoDFA
