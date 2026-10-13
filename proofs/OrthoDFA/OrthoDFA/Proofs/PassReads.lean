import OrthoDFA.Pass
import OrthoDFA.ClusteringQuality
import OrthoDFA.FamilyRead
import OrthoDFA.Proofs.Replay
import Mathlib.Data.List.Basic
import OrthoDFA.Proofs.DTree
import OrthoDFA.Proofs.HypothesisDFA

/-!
# A family's cut as reads
-/

namespace OrthoDFA

variable {α : Type*} [Fintype α] [DecidableEq α]

namespace DTree

end DTree

section Congr

variable (B : State) (F : Finset (FreeMonoid α)) (f₁ f₂ : FreeMonoid α → ℝ)

/-- The family `F` cut at `B`, read through `f`. -/
abbrev rd (f : FreeMonoid α → ℝ) : CutReads α := ⟨B, F, f⟩

/-- `e` as a string: empty, or the letter. -/
def ext : Option α → FreeMonoid α
  | none => 1
  | some c => FreeMonoid.of c

end Congr

end OrthoDFA
