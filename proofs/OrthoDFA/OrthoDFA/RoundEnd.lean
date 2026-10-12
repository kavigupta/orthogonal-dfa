import OrthoDFA.ProbeDistribution

/-!
# How a round ends well

Every level of the round's proof claims the same ending: consistent, with the hypothesis and the
tree parting on at most `ε` of the probes, or a harvest more than half of whose strings are at bad
states, whose sampler is spread and yields. What makes a state bad is each level's own.

The harvest's sampler is a replay of the final hypothesis on a fresh draw, from an anchor drawn
uniformly from `s, …, L`. It is spread when it takes no string `t` more often than

    B(t) = ∑_{s ≤ i ≤ |t|} (min(i, L) + 1 − s) / (L + 1 − s)
             · D(the draw's first i letters are t's),

and yields `y` when it finds a string on at least `y` of its draws and anchors. Each level
supplies its replay and its `y`. With `s = ⌈L/2⌉`, #415's `preconditions.start_length`, the anchors
are the positions from which `covered_states` counts a visit.
-/

namespace OrthoDFA

open MeasureTheory

variable {α : Type*}

inductive RoundEnd (α : Type*)
  | consistent
  | harvest (zs : List (FreeMonoid α))
  | failed

/-- `replay w e` is what replaying a draw `w` from the anchor `e` harvests. -/
def Spread [Countable α] (D : Measure (FreeMonoid α)) (s L : ℕ)
    (replay : FreeMonoid α → ℕ → List (FreeMonoid α)) : Prop :=
  ∀ t : FreeMonoid α, (D.prod (uniformAnchor s L)).real {we | t ∈ replay we.1 we.2}
    ≤ ∑ i ∈ Finset.Icc s t.toList.length,
        ((min i L + 1 - s : ℕ) : ℝ) / (L + 1 - s : ℕ)
          * D.real {p | p.toList.take i = t.toList.take i}

/-- The replay finds something on at least `y` of its draws and anchors. -/
def Yields [Countable α] (D : Measure (FreeMonoid α)) (s L : ℕ)
    (replay : FreeMonoid α → ℕ → List (FreeMonoid α)) (y : ℝ) : Prop :=
  y ≤ (D.prod (uniformAnchor s L)).real {we | replay we.1 we.2 ≠ []}

open scoped Classical in
/-- `disagree` is where the hypothesis and the tree part; `replay` is the harvest's, from anchors
`s, …, L`, and `y` what it must yield. -/
def EndsWell [Countable α] {σ : Type*} (M : DFA α σ) (bad : σ → Prop)
    (D : Measure (FreeMonoid α)) (ε : ℝ) (disagree : Set (FreeMonoid α)) (s L : ℕ)
    (replay : FreeMonoid α → ℕ → List (FreeMonoid α)) (y : ℝ) : RoundEnd α → Prop
  | .consistent => D.real disagree ≤ ε
  | .harvest zs => zs.length < 2 * (zs.filter fun z => bad (M.eval z.toList)).length
      ∧ Spread D s L replay ∧ Yields D s L replay y
  | .failed => False

end OrthoDFA
