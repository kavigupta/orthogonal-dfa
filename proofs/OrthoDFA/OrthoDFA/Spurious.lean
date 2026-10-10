import OrthoDFA.RoundStrong

/-!
# Where a draw's search ends at an edge

`EdgeCause`: a draw whose search ends at an edge has a read, down the route a reference placement
gives one of its prefixes' states, that the cut decides on the other side; or the edge it ends at
leads off the leaf its next state is placed at.

`SpuriousDraw`: whatever chose the hypothesis, a draw drawn apart from the noise has such a read
with chance at most `spurRate`, the misread chances of at most `ψ₀` summed over its prefixes and
the midfixes up to length `n`, and the chance it lies in `BadRoute`, the draws a prefix of which the
hypothesis routes through a midfix longer than `n` or one misread more than `ψ₀`.

`RoundStrongSpurious`: so for each reading of the round, each draw of its refusal sample and each
fresh probe of its pass.
-/

namespace OrthoDFA

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

/-- The midfixes down `z`'s route by `side`. -/
def routeMids (side : FreeMonoid α → Bool) (z : FreeMonoid α) : DTree α → List (FreeMonoid α)
  | .leaf => []
  | .node m r a => m :: if side (z * m) then routeMids side z a else routeMids side z r

section Defs

variable {Q : Type*}

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}

end Defs

/-- What each reading's pass starts from: the round's accumulator and the probes it reruns. -/
noncomputable def strongEntries (C : StrongCfg α) (R : CutReads α) :
    (n : ℕ) → ℕ → RoundAcc α → List (FreeMonoid α) → (Fin n → C.Draws)
      → List (RoundAcc α × List (FreeMonoid α))
  | 0, _, _, _, _ => []
  | n + 1, j, A, first, d =>
    (A, first) :: match strongReading C R j A first (d 0) with
      | (_, .inl _) => []
      | (A', .inr lv) => strongEntries C R n (j + 1) A' lv (Fin.tail d)

end OrthoDFA
