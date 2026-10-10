import OrthoDFA.Round
import OrthoDFA.Proofs.Replay

/-!
# The edge gap holds or a chain advances

Read at the middle of the band with every state read cleanly, an edge's reading lands off it
either rarely or nearly always, since only a flipped node read on the two paths moves it.  A
badly read state is reached from a visited string, which advances an indecision chain at some
rung of a geometric ladder.  So the round's hypothesis need not be assumed to meet the edge gap.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory
open scoped ENNReal

variable {α : Type*} [Fintype α] [DecidableEq α] {Ω : Type*} [MeasurableSpace Ω]
variable {μ : Measure Ω} {Q : Type*}

noncomputable instance (O : Oracle μ (FreeMonoid α)) (B : State) (F : Finset (FreeMonoid α)) (ω : Ω)
    (z : FreeMonoid α) : Decidable (midRead O B F ω z) :=
  inferInstanceAs (Decidable (B.lo + B.hi < 2 * _))

end OrthoDFA
