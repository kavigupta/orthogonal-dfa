import OrthoDFA.Stage
import OrthoDFA.Automaton

/-!
# Splitting a leaf

`splitAt` splits a leaf of a tree on a midfix; `CutReads.mid` is a read past the middle of the
band.
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

variable (R : CutReads α)

end OrthoDFA
