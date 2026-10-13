import OrthoDFA.FamilyRead

/-! How probes are drawn. -/

namespace OrthoDFA

open MeasureTheory

variable {α : Type*}

/-- Draws of length at most `L`. -/
structure PrefixDistribution (α : Type*) [Countable α] where
  D : Measure (FreeMonoid α)
  isProbability : IsProbabilityMeasure D
  L : ℕ
  length_le : ∀ᵐ x ∂D, x.toList.length ≤ L

instance [Countable α] (P : PrefixDistribution α) : IsProbabilityMeasure P.D := P.isProbability

end OrthoDFA
