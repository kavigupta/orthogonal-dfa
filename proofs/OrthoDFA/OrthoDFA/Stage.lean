import OrthoDFA.Clustering
import OrthoDFA.Hypothesis

/-!
# Trees and their reads

A family's cut reads a string accept, reject or undecided (`CutReads`). A discrimination tree
places a string by reading it followed by each node's midfix (`route`, `siftCut`), and a hypothesis
sets each leaf's edges by letter (`Hypothesis`).
-/

namespace OrthoDFA

open MeasureTheory
open scoped ENNReal

section

variable {α : Type*} [Fintype α] [DecidableEq α]

instance : MeasurableSpace (FreeMonoid α) := ⊤

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

end

namespace DTree

variable {α : Type*}

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

def siftCut (cut : FreeMonoid α → Option Bool) (t : DTree α) (x : FreeMonoid α) :
    List Bool ⊕ FreeMonoid α :=
  (t.route cut x).2

end DTree

end OrthoDFA
