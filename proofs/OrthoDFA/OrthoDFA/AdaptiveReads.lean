import Mathlib.Probability.Independence.Basic

/-!
# A union bound over adaptively chosen reads

Every string's read is drawn once, independently across strings, so reading a string again
returns the same value. Queries come in rounds: round `j` reads the strings `Q j r`, chosen from
what the earlier rounds read. A string's first query is independent of its read, so each string
queried reads bad with chance at most `ε`, however the queries were chosen.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

/-- Reads `read z`, independent across the countably many strings `z`, each in its bad set `B z`
with chance at most `ε`. Round `j` queries the strings `Q j r`, `r` the reads of every string:
two reads agreeing on what the rounds before `j` queried have round `j` query the same strings.
While no earlier query read bad, round `j` queries at most `b j` strings. Then some query of the
first `n` rounds reads bad with chance at most `ε (b 0 + ⋯ + b (n − 1))`. -/
def AdaptiveReadsBound : Prop :=
  ∀ {Z Ω R : Type*} [Countable Z] [MeasurableSpace Ω] [MeasurableSpace R] [Countable R]
    [MeasurableSingletonClass R] (μ : Measure Ω) [IsProbabilityMeasure μ] (read : Z → Ω → R)
    (B : Z → Set R) (ε : ℝ) (Q : ℕ → (Z → R) → Finset Z) (b : ℕ → ℕ) (n : ℕ),
    (∀ z, Measurable (read z)) → iIndepFun read μ →
    (∀ z, μ.real {ω | read z ω ∈ B z} ≤ ε) →
    (∀ j r r', (∀ i < j, ∀ z ∈ Q i r, r z = r' z) → Q j r = Q j r') →
    (∀ j r, (∀ i < j, ∀ z ∈ Q i r, r z ∉ B z) → (Q j r).card ≤ b j) →
    μ {ω | ∃ j < n, ∃ z ∈ Q j (read · ω), read z ω ∈ B z}
      ≤ ENNReal.ofReal (ε * ∑ j ∈ Finset.range n, (b j : ℝ))

end OrthoDFA
