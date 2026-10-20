import OrthoDFA.Level3Check
import OrthoDFA.Utils.SeqTest

/-!
# `Level3CheckSound`: a passing check certifies its rate, whatever the reads

The round (`indecisiveLStarCheck`) runs on any fixed reads, each block's probes drawn
independently from `D` and its check's strings from `Dc`. It ends consistent at a hypothesis whose
probes from `0` fail to agree with chance above the check's rate with chance at most `δ/8`:

    P(ending = consistent ∧ rate < Dc(startFails H)) ≤ δ/8

The round ends consistent only where a check passes, at the hypothesis it checked, and the `i`-th
check reads fresh strings at level `δ/(8(i + 1)(i + 2))`.
-/

namespace OrthoDFA

open MeasureTheory Random

/-- The round ends consistent at a hypothesis failing more than the check's rate from `0` with
chance at most `δ/8`, whatever the reads. -/
def Level3CheckSound : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] (read : FreeMonoid α → ARU)
    (D Dc : Measure (FreeMonoid α)) [IsProbabilityMeasure D] [IsProbabilityMeasure Dc]
    (C : Level3Settings) (K : CheckSettings),
    0 < K.δ → 0 < K.rate → K.rate < 1 →
    (Measure.pi fun _ : Fin K.runs =>
        (Measure.pi fun _ : Fin C.probeBudget => D).prod (Measure.pi fun _ : Fin K.size => Dc)).real
        {blocks | let result := indecisiveLStarCheck C K read blocks
          result.ending = .consistent
            ∧ K.rate < Dc.real {x | startFails read result.hypothesis.tree
              result.hypothesis.edges x = true}}
      ≤ K.δ / 8

end OrthoDFA
