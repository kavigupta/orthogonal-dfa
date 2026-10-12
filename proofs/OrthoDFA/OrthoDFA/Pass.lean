import OrthoDFA.Stage
import OrthoDFA.Automaton

/-!
# A read past the middle of the band

`CutReads.mid` reads past the middle of the band.
-/

/-- Outside `OrthoDFA`, so that `FamilyRead`'s scoped, noncomputable instance does not supply its
fields. -/
instance (priority := high) {α : Type*} [Fintype α] [DecidableEq α] :
    OrthoDFA.Stringlike (FreeMonoid α) where
  measurableSet_singleton _ := trivial
  measurable_const_mul _ _ _ := trivial
  measurable_mul_const _ _ _ := trivial
  exists_injective_nat' := Countable.exists_injective_nat (List α)
  decEq := fun a b => (inferInstance : DecidableEq (List α)) a b

namespace OrthoDFA

variable {α : Type*} [Fintype α] [DecidableEq α]

/-- The gate's read of `y`: past the middle of the band. -/
noncomputable def CutReads.mid (R : CutReads α) (y : FreeMonoid α) : Bool :=
  decide (R.B.lo + R.B.hi < 2 * acceptsOn R.F R.f y)

variable (R : CutReads α)

end OrthoDFA
