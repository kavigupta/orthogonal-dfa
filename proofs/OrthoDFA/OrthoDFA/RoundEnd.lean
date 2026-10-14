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

open scoped Classical in
/-- `disagree` is where the hypothesis and the tree part; `law h` is what the harvest `h`'s
sampler finds, spread from `k`, and `y` what it must yield. -/
def EndsWell [Countable α] {σ H : Type*} (M : DFA α σ) (bad : σ → Prop)
    (D : Measure (FreeMonoid α)) (ε : ℝ) (disagree : Set (FreeMonoid α)) (k : ℕ)
    (law : H → Measure (FreeMonoid α)) (y : ℝ) : RoundEnd H → Prop
  | .consistent => D.real disagree ≤ ε
  | .harvest h => (law h).real Set.univ < 2 * (law h).real {z | bad (M.eval z.toList)}
      ∧ Spread D k (law h) ∧ Yields (law h) y
  | .failed => False

end OrthoDFA
