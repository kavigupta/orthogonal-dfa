import OrthoDFA.StartAtK
import OrthoDFA.Pass
import Mathlib.Analysis.SpecialFunctions.Log.Base

/-!
# The loop's bookkeeping

The classes the loop's undecided exits count (`Cls`), the outcomes that disagree, and the edge
updates a split makes (`retarget`, `DTree.midAt`).
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

/-- The undecided outcomes the loop's exits watch. -/
inductive Cls
  | start | stop | triple | pair | edge
  deriving DecidableEq

instance : Fintype Cls :=
  ⟨{.start, .stop, .triple, .pair, .edge}, fun c => by cases c <;> simp⟩

/-- The class an outcome counts toward. -/
def Outcome.cls : Outcome α → Option Cls
  | .startUndecided _ => some .start
  | .endUndecided _ => some .stop
  | .triple _ => some .triple
  | .pair _ => some .pair
  | .edge _ _ => some .edge
  | _ => none

/-- The outcomes that disagree: an edge, a pair or a triple. -/
def Outcome.Disagrees (o : Outcome α) : Prop :=
  o.cls = some .edge ∨ o.cls = some .pair ∨ o.cls = some .triple

/-- What the loop carries: the hypothesis, each edge's records this stretch, whether it has been
redirected since it was learned, the strings read so far, and the stretch's counts. -/
structure LoopState (α : Type*) where
  tree : DTree α
  edges : Edges α
  recs : List Bool → α → List (FreeMonoid α × List Bool)
  moved : List Bool → α → Bool
  log : Set (FreeMonoid α)
  n : ℕ
  hits : Cls → ℕ

/-- `s` with a fresh stretch. -/
def LoopState.fresh (s : LoopState α) : LoopState α :=
  { s with recs := fun _ _ => [], n := 0, hits := fun _ => 0 }

/-- The stretch's disagreeing probes. -/
def LoopState.dis (s : LoopState α) : ℕ := s.hits .edge + s.hits .pair + s.hits .triple

/-- The longest common prefix. -/
def lcp : List Bool → List Bool → List Bool
  | a :: as, b :: bs => if a = b then a :: lcp as bs else []
  | _, _ => []

/-- The midfix at the node `p` leads to, `ε` past a leaf. -/
def DTree.midAt : DTree α → List Bool → FreeMonoid α
  | .leaf, _ => 1
  | .node m _ _, [] => m
  | .node _ r _, false :: p => midAt r p
  | .node _ _ a, true :: p => midAt a p

/-- After the leaf `p` splits into `t'`, an edge into `p` sifts its witness's successor again. -/
noncomputable def retarget (R : CutReads α) (t' : DTree α) (p : List Bool) (edges : Edges α) :
    Edges α := fun q c =>
  match edges q c with
  | some (s, w) =>
    if s = p then (t'.sift R.cut (w * FreeMonoid.of c)).elim (fun s' => some (s', w)) fun _ => none
    else some (s, w)
  | none => none

/-- `s` with the edge `(p, c)` set to `e`. -/
def LoopState.setEdge (s : LoopState α) (p : List Bool) (c : α)
    (e : Option (List Bool × FreeMonoid α)) (moved : Bool) : LoopState α :=
  { s with edges := Function.update s.edges p (Function.update (s.edges p) c e),
           moved := Function.update s.moved p (Function.update (s.moved p) c moved) }

/-- `s` with a record of `(sp, t)` at the edge `(p, c)`. -/
def LoopState.record (s : LoopState α) (p : List Bool) (c : α) (sp : FreeMonoid α)
    (t : List Bool) : LoopState α :=
  let r := Function.update (s.recs p) c (s.recs p c ++ [(sp, t)])
  { s with recs := Function.update s.recs p r }

end OrthoDFA
