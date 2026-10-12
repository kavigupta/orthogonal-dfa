import OrthoDFA.StartAtK
import OrthoDFA.Pass
import Mathlib.Analysis.SpecialFunctions.Log.Base

/-!
# Undecided classes

The classes of undecided outcomes (`Cls`).
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

/-- The undecided outcomes the loop's exits watch. -/
inductive Cls
  | start | stop | triple | pair | edge
  deriving DecidableEq

instance : Fintype Cls :=
  ⟨{.start, .stop, .triple, .pair, .edge}, fun c => by cases c <;> simp⟩

end OrthoDFA
