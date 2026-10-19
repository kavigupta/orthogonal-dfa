import OrthoDFA.FamilyRead

/-!
# How a round ends well

Every level of the round's proof claims the same ending: consistent, with the hypothesis and the
tree parting on at most `ε` of the probes, or a harvest. A harvest is a sampler, and `law h` the
law of what it finds: more than half of that law is at bad states, it is spread and it yields.
What makes a state bad is each level's own.

The sampler is spread from `k` when it finds no string more often than draws share its first `k`
letters, and yields `y` when it finds `y` in all.
-/

namespace OrthoDFA

open MeasureTheory

variable {α : Type*}

inductive RoundEnd (H : Type*)
  | consistent
  | harvest (h : H)
  | failed

/-- No string has more of `ν` than draws of `D` share its first `k` letters. -/
def Spread [Countable α] (D : Measure (FreeMonoid α)) (k : ℕ) (ν : Measure (FreeMonoid α)) :
    Prop :=
  ∀ t : FreeMonoid α, ν.real {t} ≤ D.real {p | p.toList.take k = t.toList.take k}

/-- `ν` is a measure, not a distribution: its total mass is the strings one attempt finds, on
average. -/
def Yields [Countable α] (ν : Measure (FreeMonoid α)) (y : ℝ) : Prop := y ≤ ν.real Set.univ

/-- What a round's ending is judged against. -/
structure EndSpec (α σ H : Type*) [Countable α] where
  target : DFA α σ
  /-- The states the reads handle poorly; a harvest must be mostly at them. -/
  badStates : σ → Prop
  probes : Measure (FreeMonoid α)
  disagreeRate : ℝ
  /-- Where the hypothesis and the tree part. -/
  disagree : Set (FreeMonoid α)
  /-- The prefix length a harvest must be spread from. -/
  walkStart : ℕ
  /-- What the harvest `h`'s sampler finds. -/
  harvestLaw : H → Measure (FreeMonoid α)
  yield : ℝ

open scoped Classical in
def EndsWell [Countable α] {σ H : Type*} (s : EndSpec α σ H) : RoundEnd H → Prop
  | .consistent => s.probes.real s.disagree ≤ s.disagreeRate
  | .harvest h =>
    (s.harvestLaw h).real Set.univ
        < 2 * (s.harvestLaw h).real {z | s.badStates (s.target.eval z.toList)}
      ∧ Spread s.probes s.walkStart (s.harvestLaw h) ∧ Yields (s.harvestLaw h) s.yield
  | .failed => False

end OrthoDFA
