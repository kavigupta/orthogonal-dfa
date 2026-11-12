import Mathlib.Probability.Independence.Basic
import RLStar.Code.Query

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

/-- With reads independent across strings, a string the program does not ask reads independently
of what the program returns:

    P(z unasked, p returns in A, read z ∈ B) = P(z unasked, p returns in A) · P(read z ∈ B). -/
def UnaskedFresh : Prop :=
  ∀ {Z Ω R β : Type*} [Countable Z] [DecidableEq Z] [MeasurableSpace Ω] [MeasurableSpace R]
    [Countable R] [MeasurableSingletonClass R] (μ : Measure Ω) [IsProbabilityMeasure μ]
    (read : Z → Ω → R) (p : Query Z R β) (z : Z) (A : Set β) (B : Set R),
    (∀ z, Measurable (read z)) → iIndepFun read μ →
    μ {ω | z ∉ p.reads (read · ω) ∧ p.eval (read · ω) ∈ A ∧ read z ω ∈ B}
      = μ {ω | z ∉ p.reads (read · ω) ∧ p.eval (read · ω) ∈ A} * μ {ω | read z ω ∈ B}

end OrthoDFA
