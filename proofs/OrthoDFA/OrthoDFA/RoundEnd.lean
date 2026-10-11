import OrthoDFA.FamilyRead

/-!
# How a round ends well

Every level of the round's proof claims the same ending: consistent, with the hypothesis and the
tree parting on at most `ε` of the probes, or a harvest more than half of whose strings are at bad
states and whose sampler is spread. What makes a state bad is each level's own.

The harvest's sampler is a replay of the final hypothesis on a fresh draw, from an anchor drawn
uniformly below `L`. It is spread when it takes no string `t` more often than

    B(t) = ∑_{i ≤ |t|} min(i + 1, L) / L · D(the draw's first i letters are t's),

about `4 / L` for uniform binary draws. Each level supplies its replay.
-/

namespace OrthoDFA

open MeasureTheory
open scoped ENNReal

variable {α : Type*}

inductive RoundEnd (α : Type*)
  | consistent
  | harvest (zs : List (FreeMonoid α))
  | failed

/-- The replay's anchor: uniform on `{0, …, L - 1}`. -/
noncomputable def anchorLaw (L : ℕ) : Measure ℕ :=
  (L : ℝ≥0∞)⁻¹ • ∑ k ∈ Finset.range L, Measure.dirac k

/-- `rp w e` is what the replay of a draw `w` from the anchor `e` harvests. -/
def Spread [Countable α] (D : Measure (FreeMonoid α)) (L : ℕ)
    (rp : FreeMonoid α → ℕ → List (FreeMonoid α)) : Prop :=
  ∀ t : FreeMonoid α, (D.prod (anchorLaw L)).real {q | t ∈ rp q.1 q.2}
    ≤ ∑ i ∈ Finset.range (t.toList.length + 1),
        ((min (i + 1) L : ℕ) : ℝ) / L * D.real {p | p.toList.take i = t.toList.take i}

open scoped Classical in
/-- `disagree` is where the hypothesis and the tree part; `rp` is the harvest's replay. -/
def EndsWell [Countable α] {σ : Type*} (M : DFA α σ) (bad : σ → Prop)
    (D : Measure (FreeMonoid α)) (ε : ℝ) (disagree : Set (FreeMonoid α)) (L : ℕ)
    (rp : FreeMonoid α → ℕ → List (FreeMonoid α)) : RoundEnd α → Prop
  | .consistent => D.real disagree ≤ ε
  | .harvest zs => zs.length < 2 * (zs.filter fun z => bad (M.eval z.toList)).length
      ∧ Spread D L rp
  | .failed => False

end OrthoDFA
