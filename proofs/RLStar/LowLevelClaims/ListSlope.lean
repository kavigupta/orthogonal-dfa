import Mathlib.Probability.Independence.Basic
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import RLStar.Code.FamilyRead

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

/-- With reads independent across strings, a program asking at most `N` strings lists strings it
read undecided, each once. Write `U q` for a bound on the chance a string at state `q` reads
undecided. For every `T ≥ 0` at once, the listed strings at states with `U q ≤ T` number at most

    (2 |σ| N T + |σ| log(|σ|/δ)) / log 2,

but for a chance at most `δ`. -/
def ListSlope : Prop :=
  ∀ {Z Ω σ β : Type*} [Countable Z] [DecidableEq Z] [MeasurableSpace Ω] [Fintype σ]
    (μ : Measure Ω) [IsProbabilityMeasure μ] (read : Z → Ω → ARU) (st : Z → σ) (U : σ → ℝ)
    (p : Query Z ARU β) (ℓ : β → List Z) (N : ℕ) (δ : ℝ),
    (∀ z, Measurable (read z)) → iIndepFun read μ →
    (∀ z, μ.real {ω | read z ω = .undecided} ≤ U (st z)) →
    (∀ r : Z → ARU, (p.reads r).card ≤ N) →
    (∀ r : Z → ARU, (ℓ (p.eval r)).Nodup ∧ ∀ w ∈ ℓ (p.eval r), w ∈ p.reads r ∧ r w = .undecided) →
    0 < δ →
    μ.real {ω | ∃ T, 0 ≤ T ∧
        (2 * Fintype.card σ * N * T + Fintype.card σ * Real.log (Fintype.card σ / δ)) / Real.log 2
          < ((ℓ (p.eval fun z => read z ω)).filter fun w => U (st w) ≤ T).length}
      ≤ δ

end OrthoDFA
