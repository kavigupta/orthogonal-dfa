import RLStar.Code.Level3Trie

namespace OrthoDFA

open MeasureTheory

/-- Distinct strings shorter than `K` hold at most `K` of `D` in their cylinders:

    Σ_{u ∈ X} D{x : u ⊑ x} ≤ K. -/
def TrieMassBound : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    (X : Finset (FreeMonoid α)) (K : ℕ), (∀ u ∈ X, u.toList.length < K) →
    ∑ u ∈ X, D.real {x | prefixOf x u.toList.length = u} ≤ K

/-- Expanding `X` to `Y` moves the walk start only of strings through an added prefix:

    D{x : ws_X(x) ≠ ws_Y(x)} ≤ Σ_{u ∈ Y \ X} D{x : u ⊑ x}. -/
def WalkStartMoves : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    (X Y : Finset (FreeMonoid α)), X ⊆ Y →
    D.real {x | Random.walkStartIn X x ≠ Random.walkStartIn Y x}
      ≤ ∑ u ∈ Y \ X, D.real {x | prefixOf x u.toList.length = u}

end OrthoDFA
