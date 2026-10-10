import OrthoDFA.StartAtK
import OrthoDFA.Pass
import Mathlib.Analysis.SpecialFunctions.Log.Base

/-!
# The round's trichotomy

The gate runs the hypothesis's DFA from every start on each draw and scores it against the draw's
root read; it passes where its test of the best start's agreement settles above `acc`, and a
gate whose test never settles refuses. On a refusal a fresh sample is read, every draw to one
outcome, until a harvest class's test against the rate incidental indecision alone would give it
fires or an edge still live turns up, else to its end: a class that fires is held, live edges rerun the pass, and
otherwise the limit halves.

`RoundTrichotomy`: outside a set of the oracle's noise of measure at most `β₁`, the gate's batch
and the refusal sample leave, but for chance `β₂`, the reading consistent, or holding a class
that fired, whose bad share is at least one less its incidental rate over its rate, or with live
edges left to rerun, or halving, which below `τ₀` needs every covering start to leave the classes
at most `ν`.
-/

namespace OrthoDFA

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

variable (R : CutReads α)

/-- A harvest class's test on the refusal sample: hits `H` among trials `Tr`, against `θ`. -/
structure ClassTest (α : Type*) where
  hits : FreeMonoid α → Prop
  trials : FreeMonoid α → Prop
  θ : ℝ

end OrthoDFA
