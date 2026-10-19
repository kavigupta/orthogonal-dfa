import Mathlib.Probability.Independence.Basic

/-!
# Bad reads against undecided ones, over adaptively chosen reads

Every string's read is drawn once, independently across strings. Each string reads bad with
chance at most `ε` plus `κ` times its chance of reading undecided. Queries come in rounds, each
chosen from what the rounds before it and its own queries read, so a string's first query is
independent of its read: the bad reads the queries meet number at most `ε` per string queried
plus `κ` per string queried undecided, on average.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

open scoped Classical in
/-- Reads `read z`, independent across the countably many strings `z`, each in its bad set `B z`
with chance at most `ε + κ` times its chance of being in `V z`. Round `j` queries the strings
`Q j r`, `r` the reads of every string, the same under any reads agreeing with `r` on what
rounds `0, …, j` query, while no earlier round's query reads bad. Over the rounds before `n`
that no bad read precedes, the queries number at most `b`, those reading in `V` at most `m`.
Then some query of the first `n` rounds reads bad with chance at most `ε b + κ m`. -/
def RaceReadsBound : Prop :=
  ∀ {Z Ω R : Type*} [Countable Z] [MeasurableSpace Ω] [MeasurableSpace R] [Countable R]
    [MeasurableSingletonClass R] (μ : Measure Ω) [IsProbabilityMeasure μ] (read : Z → Ω → R)
    (B V : Z → Set R) (ε κ : ℝ) (Q : ℕ → (Z → R) → Finset Z) (b m : ℝ) (n : ℕ),
    (∀ z, Measurable (read z)) → iIndepFun read μ → 0 ≤ ε → 0 ≤ κ →
    (∀ z, μ.real {ω | read z ω ∈ B z} ≤ ε + κ * μ.real {ω | read z ω ∈ V z}) →
    (∀ j r r', (∀ i < j, ∀ z ∈ Q i r, r z ∉ B z) → (∀ i ≤ j, ∀ z ∈ Q i r, r z = r' z) →
      Q j r = Q j r') →
    (∀ r, ∑ j ∈ (Finset.range n).filter (fun j => ∀ i < j, ∀ z ∈ Q i r, r z ∉ B z),
      ((Q j r).card : ℝ) ≤ b) →
    (∀ r, ∑ j ∈ (Finset.range n).filter (fun j => ∀ i < j, ∀ z ∈ Q i r, r z ∉ B z),
      (((Q j r).filter fun z => r z ∈ V z).card : ℝ) ≤ m) →
    μ {ω | ∃ j < n, ∃ z ∈ Q j (read · ω), read z ω ∈ B z} ≤ ENNReal.ofReal (ε * b + κ * m)

end OrthoDFA
