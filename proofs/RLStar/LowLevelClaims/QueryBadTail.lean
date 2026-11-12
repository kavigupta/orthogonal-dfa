import Mathlib.Probability.Independence.Basic
import Mathlib.Analysis.SpecialFunctions.Exp
import RLStar.Code.Query

/-!
# Bad reads among the strings a program asks

Every string's read is drawn once, independently across strings, and a program picks each
question from the answers to the earlier ones. The bad reads among the distinct strings it asks
have a Poisson-like tail.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

open scoped Classical in
/-- Reads `read z`, independent across the countably many strings `z`, each in its bad set `B z`
with chance at most `u`. A program asking at most `N` strings under any answers finds at least `f`
of them bad with chance at most

    exp(t' u N) / (1 + t)^f,    for t, t' ≥ 0 with t u ≤ 1 − exp(−t' u). -/
def QueryBadTail : Prop :=
  ∀ {Z Ω R β : Type*} [Countable Z] [DecidableEq Z] [MeasurableSpace Ω] [MeasurableSpace R]
    [Countable R] [MeasurableSingletonClass R] (μ : Measure Ω) [IsProbabilityMeasure μ]
    (read : Z → Ω → R) (B : Z → Set R) (u : ℝ) (p : Query Z R β) (N f : ℕ) (t t' : ℝ),
    (∀ z, Measurable (read z)) → iIndepFun read μ → 0 ≤ u →
    (∀ z, μ.real {ω | read z ω ∈ B z} ≤ u) →
    (∀ r : Z → R, (p.reads r).card ≤ N) →
    0 ≤ t → 0 ≤ t' → t * u ≤ 1 - Real.exp (-(t' * u)) →
    μ {ω | f ≤ ((p.reads fun z => read z ω).filter fun z => read z ω ∈ B z).card}
      ≤ ENNReal.ofReal (Real.exp (t' * (u * N)) / (1 + t) ^ f)

end OrthoDFA
