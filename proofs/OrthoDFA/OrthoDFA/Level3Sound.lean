import OrthoDFA.Level3Claim

/-!
# `Level3ConsistentSound`: a consistent ending certifies its rate, whatever the reads

Level 3 (`indecisiveLStarLevel3`) runs on any fixed reads, its probes drawn independently from
`D`. It ends consistent at a hypothesis `H` whose probes search or stop undecided at their start or
end sift with chance above the disagreement rate `ε` with chance at most `δ/8`:

    P(ending = consistent ∧ ε < D(searchSet H ∪ endsSet H)) ≤ δ/8

Over a stretch the hypothesis is fixed, so the success test counts fresh draws of that set; the
`j`-th hypothesis's test runs at level `δ/(8(j + 1)(j + 2))`, and these sum to `δ/8`. Nothing is
asked of the reads: neither that they read a string on its state's side, nor that they are
independent, nor that the hypotheses be few.
-/

namespace OrthoDFA

open MeasureTheory Random

/-- Level 3 ends consistent at a hypothesis that disagrees on more than its rate with chance at most
`δ/8`, whatever the reads. -/
def Level3ConsistentSound : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] (read : FreeMonoid α → ARU)
    (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (C : Level3Settings),
    0 < C.δ → 0 < C.disagreementRate → C.disagreementRate < 1 →
    (Measure.pi fun _ : Fin C.probeBudget => D).real
        {draws | let result := indecisiveLStarLevel3 C read draws
          result.ending = .consistent
            ∧ C.disagreementRate < D.real (searchSet read C.walkStart result.hypothesis
              ∪ endsSet read C.walkStart result.hypothesis)}
      ≤ C.δ / 8

end OrthoDFA
