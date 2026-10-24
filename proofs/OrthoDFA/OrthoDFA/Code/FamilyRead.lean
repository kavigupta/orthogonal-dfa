import OrthoDFA.Code.Clustering
import Mathlib.Computability.DFA

/-!
# The family's read of a string

`X(w)` counts the members `v` of the suffix family `F` whose query `w·v` answers 1.  The read
accepts at `X(w) ≥ kh`, rejects at `X(w) ≤ kl`, and is undecided in between.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

/-- Accept, reject or undecided. -/
inductive ARU
  | accept
  | reject
  | undecided
  deriving DecidableEq

instance : MeasurableSpace ARU := ⊤

def readOf (kl kh x : ℕ) : ARU :=
  if kh ≤ x then .accept else if x ≤ kl then .reject else .undecided

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
variable {S : Type*} [Stringlike S]

noncomputable def familyRead (mq : S → Ω → ℝ) (F : Finset S) (kl kh : ℕ) (w : S) (ω : Ω) :
    ARU :=
  readOf kl kh (voteCount mq F w ω)

/-! ## Strings -/

/-- Scoped, so that it never meets another `MeasurableSpace (FreeMonoid α)`.  On a countable
type the discrete σ-algebra is the only one with measurable singletons. -/
noncomputable scoped instance {α : Type*} [Countable α] : Stringlike (FreeMonoid α) where
  toMeasurableSpace := ⊤
  measurable_const_mul _ := fun _ _ => trivial
  measurable_mul_const _ := fun _ _ => trivial
  exists_injective_nat' := (inferInstance : Countable (List α)).exists_injective_nat'
  measurableSet_singleton _ := trivial
  decEq := Classical.decEq _

end OrthoDFA
