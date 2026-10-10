import OrthoDFA.Clustering

/-!
# The round's outcome

A round of the L\* stage ends with a hypothesis: `TransitionResolver`'s discrimination tree over
the family's cut, and the edges between its leaves.  The DFA/DT agreement gate
(`estimate_agreement_rate`) reads the tree at the middle of the cut's band and walks the
hypothesis from where that reading sends `ε`.  The round's harvest is replayed as `Walked` does:
from the first prefix at least as long as an earliest anchor drawn uniformly below `L` that the
tree places, walk the hypothesis, sift the draw, and bisect to the first edge where walk and sift
part, keeping every string the cut cannot place and, with every read decided, the draw up to that
edge.

`RoundOutcome`: for any tree and DFA a round ends with, if the DFA/DT agreement check fails on a
share `d` of sampler strings, the harvest sampler finds at least one string on at least a `d/L`
share of attempts, harvests no single string `t` on more than a `B(t)` share of them, and reads no
single string `t` on more than a `B(t)` share of them, where

    B(t) = ∑_{i ≤ |t|} min(i + 1, L) / L · D(the draw's first i letters are t's).

Known modelling gap.  This is `Walked` as #398 (the uniform anchor) and #403 (one harvest, which
keeps the disagreement's prefix) have it; on main it walks from the start and keeps no prefix.

Known modelling gap.  `HarvestSource` counts an attempt only when it turns up a string it has not
served, and retires a source whose yield falls below `(1 - acc_threshold) / 2`, while
`RoundOutcome` guarantees a refused round's replay only `(1 - acc_threshold) / L`, so a source
between the two can be retired that the argument relies on.
-/

namespace OrthoDFA

open MeasureTheory
open scoped ENNReal

variable {α : Type*} [Fintype α] [DecidableEq α]

instance : MeasurableSpace (FreeMonoid α) := ⊤

/-- The first `i` letters of `w`. -/
def prefixOf (w : FreeMonoid α) (i : ℕ) : FreeMonoid α := FreeMonoid.ofList (w.toList.take i)

open scoped Classical in
/-- How many of `V` accept `x` extended by them, read through `f`. -/
noncomputable def acceptsOn (V : Finset (FreeMonoid α)) (f : FreeMonoid α → ℝ)
    (x : FreeMonoid α) : ℕ :=
  (V.filter fun v => f (x * v) = 1).card

/-- The family `F`, read through `f` and cut at `B`: `x` accepts above `B.hi`, rejects at or
below `B.lo`, and is undecided between. -/
structure CutReads (α : Type*) where
  B : State
  F : Finset (FreeMonoid α)
  f : FreeMonoid α → ℝ

noncomputable def CutReads.cut (R : CutReads α) (x : FreeMonoid α) : Option Bool :=
  if R.B.hi < acceptsOn R.F R.f x then some true
  else if acceptsOn R.F R.f x ≤ R.B.lo then some false else none

/-- A discrimination tree.  A node's accepting side is `acc`, its rejecting side `rej`. -/
inductive DTree (α : Type*)
  | leaf
  | node (midfix : FreeMonoid α) (rej acc : DTree α)

namespace DTree

/-- `x`'s way down the tree by `cut`: the strings it asks the cut about, root first (`x` followed
by each node's midfix), and the path to its leaf (`true` for an accepting side) or the string a
node could not place. -/
def route (cut : FreeMonoid α → Option Bool) :
    DTree α → FreeMonoid α → List (FreeMonoid α) × (List Bool ⊕ FreeMonoid α)
  | .leaf, _ => ([], .inl [])
  | .node m r a, x =>
    match cut (x * m) with
    | none => ([x * m], .inr (x * m))
    | some true => (x * m :: (a.route cut x).1, (a.route cut x).2.map (true :: ·) id)
    | some false => (x * m :: (r.route cut x).1, (r.route cut x).2.map (false :: ·) id)

def sift (cut : FreeMonoid α → Option Bool) (t : DTree α) (x : FreeMonoid α) :
    List Bool ⊕ FreeMonoid α :=
  (t.route cut x).2

/-- The leaves' paths. -/
def paths : DTree α → List (List Bool)
  | .leaf => [[]]
  | .node _ r a => r.paths.map (false :: ·) ++ a.paths.map (true :: ·)

end DTree

/-- The hypothesis: the tree, and each leaf's edges by letter, open where `none`. -/
structure Hypothesis (α : Type*) where
  tree : DTree α
  edges : List Bool → α → Option (List Bool)

/-- The totalised step: the edge's target when it is a leaf, else stay put. -/
def Hypothesis.step (H : Hypothesis α) (path : List Bool) (c : α) : List Bool :=
  match H.edges path c with
  | some p => if p ∈ H.tree.paths then p else path
  | none => path

variable (R : CutReads α)

/-- The earliest anchor's law: uniform on `{0, …, L - 1}`. -/
noncomputable def anchorLaw (L : ℕ) : Measure ℕ :=
  (L : ℝ≥0∞)⁻¹ • ∑ k ∈ Finset.range L, Measure.dirac k

end OrthoDFA
