import OrthoDFA.Pass
import OrthoDFA.ClusteringQuality
import OrthoDFA.FamilyRead

/-!
# The round's outcomes

A round reads its family through the oracle, runs the counterexample pass on fresh sampler draws,
and either the DFA and tree it ends with agree, or it leaves a population the next gate must act
on, or the pass halves the indecision limit, or its family reads some state badly, or the
hypothesis takes some visited edge wrongly.

`ν(y)` is how often a uniformly random position of a draw from `D` has the prefix `y`, and `V(q)`
how often it reaches the target state `q`.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

variable {α : Type*} [Fintype α] [DecidableEq α]
variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
variable {Q : Type*}

/-- A node read of `z` at the middle of the band: the family's vote on `z` past the middle. -/
def midRead (O : Oracle μ (FreeMonoid α)) (B : State) (F : Finset (FreeMonoid α)) (ω : Ω)
    (z : FreeMonoid α) : Prop :=
  B.lo + B.hi < 2 * acceptsOn F (fun w => O.mq w ω) z

end OrthoDFA
