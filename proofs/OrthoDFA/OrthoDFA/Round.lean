import OrthoDFA.Pass
import OrthoDFA.ClusteringQuality
import OrthoDFA.FamilyRead

/-!
# A node read at the middle of the band
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
