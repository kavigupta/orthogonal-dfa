import OrthoDFA.ProbeDistribution

/-!
# How a round ends well

Every level of the round's proof claims the same ending: consistent, with the hypothesis and the
tree parting on at most `ε` of the probes, or a harvest. A harvest is a sampler, and `law h` the
law of what it finds: more than half of that law is at bad states, it is spread and it yields.
What makes a state bad is each level's own.

The sampler is spread from `k` when it finds no string more often than draws share its first `k`
letters, and yields `y` when it finds `y` in all.

Level 2's harvest is still a list, replayed from an anchor drawn uniformly from `s, …, L` on the
final hypothesis (`ReplayEndsWell`). Its replay is spread when it takes no string `t` more often
than

    B(t) = ∑_{s ≤ i ≤ |t|} (min(i, L) + 1 − s) / (L + 1 − s)
             · D(the draw's first i letters are t's),

and yields `y` when it finds a string on at least `y` of its draws and anchors. With
`s = ⌈L/2⌉`, #415's `preconditions.start_length`, the anchors are the positions from which
`covered_states` counts a visit.
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

/-- `replay w e` is what replaying a draw `w` from the anchor `e` harvests. -/
def ReplaySpread [Countable α] (D : Measure (FreeMonoid α)) (s L : ℕ)
    (replay : FreeMonoid α → ℕ → List (FreeMonoid α)) : Prop :=
  ∀ t : FreeMonoid α, (D.prod (uniformAnchor s L)).real {we | t ∈ replay we.1 we.2}
    ≤ ∑ i ∈ Finset.Icc s t.toList.length,
        ((min i L + 1 - s : ℕ) : ℝ) / (L + 1 - s : ℕ)
          * D.real {p | p.toList.take i = t.toList.take i}

/-- The replay finds something on at least `y` of its draws and anchors. -/
def ReplayYields [Countable α] (D : Measure (FreeMonoid α)) (s L : ℕ)
    (replay : FreeMonoid α → ℕ → List (FreeMonoid α)) (y : ℝ) : Prop :=
  y ≤ (D.prod (uniformAnchor s L)).real {we | replay we.1 we.2 ≠ []}

open scoped Classical in
/-- Level 2's ending: its harvest is a list, more than half of whose strings are at bad states,
and its replay is spread and yields. -/
def ReplayEndsWell [Countable α] {σ : Type*} (M : DFA α σ) (bad : σ → Prop)
    (D : Measure (FreeMonoid α)) (ε : ℝ) (disagree : Set (FreeMonoid α)) (s L : ℕ)
    (replay : FreeMonoid α → ℕ → List (FreeMonoid α)) (y : ℝ) :
    RoundEnd (List (FreeMonoid α)) → Prop
  | .consistent => D.real disagree ≤ ε
  | .harvest zs => zs.length < 2 * (zs.filter fun z => bad (M.eval z.toList)).length
      ∧ ReplaySpread D s L replay ∧ ReplayYields D s L replay y
  | .failed => False

end OrthoDFA
