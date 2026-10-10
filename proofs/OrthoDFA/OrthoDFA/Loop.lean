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

end OrthoDFA
