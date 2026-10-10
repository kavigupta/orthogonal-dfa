import OrthoDFA.FamilyRead

/-!
# How a round ends well

Every level of the round's proof claims the same ending: consistent, with the hypothesis and the
tree parting on at most `ε` of the probes, or a harvest more than half of whose strings are at bad
states. What makes a state bad is each level's own.
-/

namespace OrthoDFA

open MeasureTheory

variable {α : Type*}

inductive RoundEnd (α : Type*)
  | consistent
  | harvest (zs : List (FreeMonoid α))
  | failed

open scoped Classical in
/-- `disagree` is where the hypothesis and the tree part. -/
def EndsWell [Countable α] {σ : Type*} (M : DFA α σ) (bad : σ → Prop)
    (D : Measure (FreeMonoid α)) (ε : ℝ) (disagree : Set (FreeMonoid α)) : RoundEnd α → Prop
  | .consistent => D.real disagree ≤ ε
  | .harvest zs => zs.length < 2 * (zs.filter fun z => bad (M.eval z.toList)).length
  | .failed => False

end OrthoDFA
