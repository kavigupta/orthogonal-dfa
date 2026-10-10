import OrthoDFA.Stage
import OrthoDFA.Automaton

/-!
# The counterexample pass

`TransitionResolver.counterexample_pass` over a round's cut: probes walked through the hypothesis
from where the gate places `ε` and re-sifted, leaves split where the evidence confirms a
disagreement, and what the cut cannot place harvested: into the bisection population when the
probe disagrees, else into the boundary population.
-/

namespace OrthoDFA

variable {α : Type*} [Fintype α] [DecidableEq α]

instance (priority := high) : Stringlike (FreeMonoid α) where
  measurableSet_singleton _ := trivial
  measurable_const_mul _ _ _ := trivial
  measurable_mul_const _ _ _ := trivial
  exists_injective_nat' := Countable.exists_injective_nat (List α)
  decEq := fun a b => (inferInstance : DecidableEq (List α)) a b

namespace DTree

/-- `t` with the leaf at the end of `path` split on `m`. -/
def splitAt (m : FreeMonoid α) : DTree α → List Bool → DTree α
  | .leaf, [] => .node m .leaf .leaf
  | .node n r a, false :: p => .node n (r.splitAt m p) a
  | .node n r a, true :: p => .node n r (a.splitAt m p)
  | t, _ => t

end DTree

/-- The gate's read of `y`: past the middle of the band. -/
noncomputable def CutReads.mid (R : CutReads α) (y : FreeMonoid α) : Bool :=
  decide (R.B.lo + R.B.hi < 2 * acceptsOn R.F R.f y)

/-- `SplitEvidence`'s verdicts. -/
inductive Verdict
  | split
  | noSplit
  | undecided

/-- The pass's knobs.  `train F` is the family's training half; `block F` the suffixes held out
from it, which only the split test reads. -/
structure StageKnobs (α : Type*) where
  train : Finset (FreeMonoid α) → Finset (FreeMonoid α)
  block : Finset (FreeMonoid α) → Finset (FreeMonoid α)
  memberLimit : ℕ
  patience : ℕ
  splitFpr : ℝ
  minSplit : ℝ
  missRate : ℝ
  /-- `BoundarySource` keeps only boundaries of prefixes at least this long. -/
  longEnough : ℕ
  /-- The leaves and distinguishers whose split tests are taken as not splitting, untested: none
  in the round itself. -/
  forced : Set (List Bool × FreeMonoid α)

/-- The suffixes read after a string: the family, its training half and the held-out block. -/
def StageKnobs.suffixes (K : StageKnobs α) (F : Finset (FreeMonoid α)) : Finset (FreeMonoid α) :=
  F ∪ K.train F ∪ K.block F

/-- What the pass carries from probe to probe: the tree, the population in the order its strings
arrived, each edge's target and witness, the probes since the last split or undecided evidence,
how many of those a read the cut could not place kept from being checked, the strings the cut
could not place, and the bisection population. -/
structure PassState (α : Type*) where
  tree : DTree α
  pool : List (FreeMonoid α)
  edges : List Bool → α → Option (List Bool × FreeMonoid α)
  streak : ℕ
  unchecked : ℕ
  /-- The node reads the probes since the last split made. -/
  reads : ℕ
  boundary : List (FreeMonoid α)
  /-- What disagreeing probes could not place: the first read the search for the disagreeing
  edge meets undecided, the prefix before an edge the hypothesis does not hold that it lands on,
  or every read in the band on the gate's reading of a probe the cut cannot place that it sends
  off the walk; and every read in the band on the gate's reading of `ε`. -/
  bisected : List (FreeMonoid α)

variable (K : StageKnobs α) (R : CutReads α)

end OrthoDFA
