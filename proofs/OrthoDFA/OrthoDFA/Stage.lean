import OrthoDFA.Clustering
import OrthoDFA.HypothesisDFA

/-!
# Trees and their reads

A family's cut reads a string accept, reject or undecided (`CutReads`).
-/

namespace OrthoDFA

open MeasureTheory
open scoped ENNReal

section

variable {α : Type*} [Fintype α] [DecidableEq α]

instance : MeasurableSpace (FreeMonoid α) := ⊤

open scoped Classical in
/-- How many of `V` accept `x` extended by them, read through `f`. -/
noncomputable def acceptsOn (V : Finset (FreeMonoid α)) (f : FreeMonoid α → ℝ)
    (x : FreeMonoid α) : ℕ :=
  (V.filter fun v => f (x * v) = 1).card

/-- The family `F`, read through `f` and cut at `B`: `x` accepts above `B.hi`, rejects at or
below `B.lo`, and is undecided between. -/
structure CutReads (α : Type*) where
  B : State
  F : Finset (FreeMonoid α)
  f : FreeMonoid α → ℝ

noncomputable def CutReads.cut (R : CutReads α) (x : FreeMonoid α) : Option Bool :=
  if R.B.hi < acceptsOn R.F R.f x then some true
  else if acceptsOn R.F R.f x ≤ R.B.lo then some false else none

end

end OrthoDFA
