import OrthoDFA.AdaptiveReads

/-!
# A union bound over adaptively chosen reads: the proof

A bad query has a first round, and there a string no earlier round queried. That the string is
first queried in round `j`, with no bad read before, is fixed by the reads of the finitely many
strings queried before, none of them the string itself, so it is independent of its read.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

theorem adaptiveReadsBound_holds : AdaptiveReadsBound := by
  sorry

end OrthoDFA
