import OrthoDFA.Code.RoundEnd

/-!
# How a round ends well

Every level of the round's proof claims the same ending: consistent, with the hypothesis and the
tree parting on at most `ε` of the probes, or a harvest. A harvest is a sampler, and `law h` the
law of what it finds: it puts little on states the reads rarely leave undecided, it is spread and
it yields. Each level says what a state's chance of reading undecided is.

The sampler is spread from `k` when it finds no string more often than draws share its first `k`
letters, and yields `y` when it finds `y` in all.
-/

namespace OrthoDFA

open MeasureTheory

variable {α : Type*}

/-- No string has more of `ν` than draws of `D` share its first `k` letters. -/
def Spread [Countable α] (D : Measure (FreeMonoid α)) (k : ℕ) (ν : Measure (FreeMonoid α)) :
    Prop :=
  ∀ t : FreeMonoid α, ν.real {t} ≤ D.real {p | p.toList.take k = t.toList.take k}

/-- `ν` is a measure, not a distribution: its total mass is the strings one attempt finds, on
average. -/
def Yields [Countable α] (ν : Measure (FreeMonoid α)) (y : ℝ) : Prop := y ≤ ν.real Set.univ

/-- A round's ending, with what it is judged against. -/
structure RoundOutcome (α σ H : Type*) [Countable α] where
  target : DFA α σ
  /-- Each state's chance of reading undecided. -/
  undecided : σ → ℝ
  /-- A harvest puts at most `harvestSlope * T + harvestSlack` of its law on the states read
  undecided at most `T` of the time. -/
  harvestSlope : ℝ
  harvestSlack : ℝ
  probes : Measure (FreeMonoid α)
  disagreeRate : ℝ
  /-- Where the hypothesis and the tree part. -/
  disagree : Set (FreeMonoid α)
  /-- The prefix length a harvest must be spread from. -/
  walkStart : ℕ
  /-- What the harvest `h`'s sampler finds. -/
  harvestLaw : H → Measure (FreeMonoid α)
  yield : ℝ
  ending : RoundEnd H

open scoped Classical in
def EndsWell [Countable α] {σ H : Type*} (s : RoundOutcome α σ H) : Prop :=
  match s.ending with
  | .consistent => s.probes.real s.disagree ≤ s.disagreeRate
  | .harvest h =>
    (∀ T, 0 ≤ T → (s.harvestLaw h).real {z | s.undecided (s.target.eval z.toList) ≤ T}
        ≤ (s.harvestSlope * T + s.harvestSlack) * (s.harvestLaw h).real Set.univ)
      ∧ Spread s.probes s.walkStart (s.harvestLaw h) ∧ Yields (s.harvestLaw h) s.yield
  | .failed => False

end OrthoDFA
