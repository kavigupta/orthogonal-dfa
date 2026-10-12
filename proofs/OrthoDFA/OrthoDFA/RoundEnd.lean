import OrthoDFA.FamilyRead

/-!
# How a round ends well

Every level of the round's proof claims the same ending: consistent, with the hypothesis and the
tree parting on at most `ε` of the probes, or a harvest more than half of whose strings are at bad
states, whose sampler is spread and yields. What makes a state bad is each level's own.

The harvest's sampler is a replay of the final hypothesis on a fresh draw, from an anchor drawn
uniformly from `⌈L/2⌉, …, L`, the positions from which #415's `covered_states` counts a visit
(`preconditions.start_length`). It is spread when it takes no string `t` more often than

    B(t) = ∑_{⌈L/2⌉ ≤ i ≤ |t|} (min(i, L) + 1 − ⌈L/2⌉) / (L + 1 − ⌈L/2⌉)
             · D(the draw's first i letters are t's),

and yields `y` when it finds a string on at least `y` of its draws and anchors. Each level
supplies its replay and its `y`.
-/

namespace OrthoDFA

open MeasureTheory
open scoped ENNReal

variable {α : Type*}

inductive RoundEnd (α : Type*)
  | consistent
  | harvest (zs : List (FreeMonoid α))
  | failed

/-- `⌈L/2⌉`, as `preconditions.start_length`. -/
def startLength (L : ℕ) : ℕ := (L + 1) / 2

/-- The probes' law: draws of length at most `L`, none starting more than `q` of the time with
any one string past `⌈L/2⌉`. -/
structure PrefixDistribution (α : Type*) [Countable α] where
  D : Measure (FreeMonoid α)
  prob : IsProbabilityMeasure D
  L : ℕ
  short : ∀ᵐ x ∂D, x.toList.length ≤ L
  q : ℝ
  q_pos : 0 < q
  prefix_le : ∀ u : FreeMonoid α, startLength L ≤ u.toList.length → u.toList.length ≤ L →
    D.real {x | u.toList <+: x.toList} ≤ q

instance [Countable α] (P : PrefixDistribution α) : IsProbabilityMeasure P.D := P.prob

/-- The replay's anchor: uniform on `{⌈L/2⌉, …, L}`. -/
noncomputable def anchorLaw (L : ℕ) : Measure ℕ :=
  ((L + 1 - startLength L : ℕ) : ℝ≥0∞)⁻¹ • ∑ k ∈ Finset.Icc (startLength L) L, Measure.dirac k

/-- `rp w e` is what the replay of a draw `w` from the anchor `e` harvests. -/
def Spread [Countable α] (D : Measure (FreeMonoid α)) (L : ℕ)
    (rp : FreeMonoid α → ℕ → List (FreeMonoid α)) : Prop :=
  ∀ t : FreeMonoid α, (D.prod (anchorLaw L)).real {q | t ∈ rp q.1 q.2}
    ≤ ∑ i ∈ Finset.Icc (startLength L) t.toList.length,
        ((min i L + 1 - startLength L : ℕ) : ℝ) / (L + 1 - startLength L : ℕ)
          * D.real {p | p.toList.take i = t.toList.take i}

/-- The replay finds something on at least `y` of its draws and anchors. -/
def Yields [Countable α] (D : Measure (FreeMonoid α)) (L : ℕ)
    (rp : FreeMonoid α → ℕ → List (FreeMonoid α)) (y : ℝ) : Prop :=
  y ≤ (D.prod (anchorLaw L)).real {q | rp q.1 q.2 ≠ []}

/-- The share of draws that visit a state satisfying `P` at some position from `⌈L/2⌉` to `L`,
as #415's `covered_states` counts a state's visits: a state `q` is covered there when
`coverage D L M (· = q)` is at least `min_coverage`. -/
noncomputable def coverage [Countable α] {σ : Type*} (D : Measure (FreeMonoid α)) (L : ℕ)
    (M : DFA α σ) (P : σ → Prop) : ℝ :=
  D.real {x | ∃ i, startLength L ≤ i ∧ i ≤ L ∧ i ≤ x.toList.length ∧
    P (M.eval (x.toList.take i))}

open scoped Classical in
/-- `disagree` is where the hypothesis and the tree part; `rp` is the harvest's replay, and `y`
what it must yield. -/
def EndsWell [Countable α] {σ : Type*} (M : DFA α σ) (bad : σ → Prop)
    (D : Measure (FreeMonoid α)) (ε : ℝ) (disagree : Set (FreeMonoid α)) (L : ℕ)
    (rp : FreeMonoid α → ℕ → List (FreeMonoid α)) (y : ℝ) : RoundEnd α → Prop
  | .consistent => D.real disagree ≤ ε
  | .harvest zs => zs.length < 2 * (zs.filter fun z => bad (M.eval z.toList)).length
      ∧ Spread D L rp ∧ Yields D L rp y
  | .failed => False

end OrthoDFA
