import OrthoDFA.FamilyRead

/-! How probes are drawn, where a replay of one is anchored, and which states the draws visit. -/

namespace OrthoDFA

open MeasureTheory
open scoped ENNReal

variable {α : Type*}

/-- Draws of length at most `L`, the replay's anchors running from `firstAnchor`, and no draw
starting more than `prefixBound` of the time with any one string at least `firstAnchor` long. -/
structure PrefixDistribution (α : Type*) [Countable α] where
  D : Measure (FreeMonoid α)
  isProbability : IsProbabilityMeasure D
  L : ℕ
  firstAnchor : ℕ
  firstAnchor_le : firstAnchor ≤ L
  length_le : ∀ᵐ x ∂D, x.toList.length ≤ L
  prefixBound : ℝ
  prefixBound_pos : 0 < prefixBound
  prefix_mass_le : ∀ u : FreeMonoid α, firstAnchor ≤ u.toList.length → u.toList.length ≤ L →
    D.real {x | u.toList <+: x.toList} ≤ prefixBound

instance [Countable α] (P : PrefixDistribution α) : IsProbabilityMeasure P.D := P.isProbability

/-- The replay's anchor: uniform on `{s, …, L}`. -/
noncomputable def uniformAnchor (s L : ℕ) : Measure ℕ :=
  ((L + 1 - s : ℕ) : ℝ≥0∞)⁻¹ • ∑ k ∈ Finset.Icc s L, Measure.dirac k

/-- The share of draws that visit a state satisfying `P` at some position from `s` to `L`. At
`s = ⌈L/2⌉` this counts visits as #415's `covered_states` does: a state `q` is covered there when
`coverage D s L M (· = q)` is at least `min_coverage`. -/
noncomputable def coverage [Countable α] {σ : Type*} (D : Measure (FreeMonoid α)) (s L : ℕ)
    (M : DFA α σ) (P : σ → Prop) : ℝ :=
  D.real {x | ∃ i, s ≤ i ∧ i ≤ L ∧ i ≤ x.toList.length ∧ P (M.eval (x.toList.take i))}

end OrthoDFA
