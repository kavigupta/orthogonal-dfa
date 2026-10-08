import OrthoDFA.Round

/-!
# The walk's ends

The walk from `k` reads two strings at the root: a draw's first `k` letters and the whole draw.
The FNR gate holds the family to its limit on both, on the uniform population for the whole draw
and on the sampler's draws cut to length `k` for the start.
-/

namespace OrthoDFA

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

/-- The sampler's draws cut to their first `k` letters. -/
noncomputable def startPopulation (D : Measure (FreeMonoid α)) (k : ℕ) : Measure (FreeMonoid α) :=
  D.map (prefixOf · k)

/-- `StartRootCovered`: where the family leaves the start population undecided at most `b` of
the time, as the clustering's gate certifies of each population it reads, the cut leaves the
root read of a draw's first `k` letters undecided at most `b` of the time. -/
def StartRootCovered : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
    [IsProbabilityMeasure μ] (O : Oracle μ (FreeMonoid α)) (B : State) (F : Finset (FreeMonoid α))
    (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (k : ℕ) (b : ℝ),
    ∫ p, undecidedProb O B.lo (B.hi + 1) F p ∂(startPopulation D k) ≤ b →
    ∫ x, μ.real {ω | (readsAt O B F ω).cut (prefixOf x k * 1) = none} ∂D ≤ b

end OrthoDFA
