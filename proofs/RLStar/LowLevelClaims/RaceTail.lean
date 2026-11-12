import Mathlib.Probability.Independence.Basic
import Mathlib.Analysis.SpecialFunctions.Exp

/-!
# How many rounds meet a fresh bad read, against undecided ones

Every string's read is drawn once, independently across strings. Each string reads bad with
chance at most `ε` plus `κ` times its chance of reading undecided. Queries come in rounds, each
fixed by what it and the rounds before it read. A round is fresh-bad when some string it queries,
and no earlier round did, reads bad. The fresh-bad rounds have a Poisson-like tail: each round
costs `ε` per new string and `κ` per new string read undecided, and at most `f` of them come
from rounds that cost at most `b` and `m` in all.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

namespace RaceTail

variable {Z R : Type*} (Q : ℕ → (Z → R) → Finset Z) (B : Z → Set R)

open scoped Classical in
/-- The strings round `j` queries and no earlier round does. -/
noncomputable def newQ (j : ℕ) (r : Z → R) : Finset Z :=
  (Q j r).filter fun z => ∀ i < j, z ∉ Q i r

/-- Round `j` reads bad a string no earlier round queried. -/
def FreshBad (j : ℕ) (r : Z → R) : Prop := ∃ z ∈ newQ Q j r, r z ∈ B z

open scoped Classical in
/-- The fresh-bad rounds before `j`. -/
noncomputable def badCount (j : ℕ) (r : Z → R) : ℕ :=
  ((Finset.range j).filter fun i => FreshBad Q B i r).card

end RaceTail

open RaceTail

open scoped Classical in
/-- Reads `read z`, independent across the countably many strings `z`, each in its bad set `B z`
with chance at most `ε + κ` times its chance of being in `V z`. Round `j` queries the strings
`Q j r`, the same under any reads agreeing with `r` on what rounds `0, …, j` query, while fewer
than `f` rounds before it are fresh-bad. Over the rounds before `n` that fewer than `f` fresh-bad
rounds precede, the new strings number at most `b`, those reading in `V` at most `m`, and each
round's `ε` per new string and `κ` per new string in `V` come to at most `w`. Then at least `f`
of the first `n` rounds are fresh-bad with chance at most

    exp(t' (ε b + κ m)) / (1 + t)^f,    for t, t' ≥ 0 with t w ≤ 1 - exp(-t' w). -/
def RaceTailBound : Prop :=
  ∀ {Z Ω R : Type*} [Countable Z] [MeasurableSpace Ω] [MeasurableSpace R] [Countable R]
    [MeasurableSingletonClass R] (μ : Measure Ω) [IsProbabilityMeasure μ] (read : Z → Ω → R)
    (B V : Z → Set R) (ε κ : ℝ) (Q : ℕ → (Z → R) → Finset Z) (f n : ℕ) (b m w t t' : ℝ),
    (∀ z, Measurable (read z)) → iIndepFun read μ → 0 ≤ ε → 0 ≤ κ →
    (∀ z, μ.real {ω | read z ω ∈ B z} ≤ ε + κ * μ.real {ω | read z ω ∈ V z}) →
    (∀ j r r', badCount Q B j r < f → (∀ i ≤ j, ∀ z ∈ Q i r, r z = r' z) → Q j r = Q j r') →
    (∀ r, ∑ j ∈ (Finset.range n).filter (fun j => badCount Q B j r < f),
      ((newQ Q j r).card : ℝ) ≤ b) →
    (∀ r, ∑ j ∈ (Finset.range n).filter (fun j => badCount Q B j r < f),
      (((newQ Q j r).filter fun z => r z ∈ V z).card : ℝ) ≤ m) →
    (∀ r j, badCount Q B j r < f →
      ε * (newQ Q j r).card + κ * ((newQ Q j r).filter fun z => r z ∈ V z).card ≤ w) →
    0 ≤ w → 0 ≤ t → 0 ≤ t' → t * w ≤ 1 - Real.exp (-(t' * w)) →
    μ {ω | f ≤ badCount Q B n (read · ω)}
      ≤ ENNReal.ofReal (Real.exp (t' * (ε * b + κ * m)) / (1 + t) ^ f)

end OrthoDFA
