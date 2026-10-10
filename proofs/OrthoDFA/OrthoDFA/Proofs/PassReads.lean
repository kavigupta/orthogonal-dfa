import OrthoDFA.Pass
import OrthoDFA.ClusteringQuality
import OrthoDFA.FamilyRead
import OrthoDFA.Proofs.Replay
import Mathlib.Data.List.Basic

/-!
# What the pass reads

Every read the counterexample pass makes asks the family about a string `b·e₁·e₂·m`: `b` a seed
string, a witness or a probe's prefix, `e₁` and `e₂` each empty or a letter, and `m` a midfix of
the tree the pass holds before the probe.  So two oracles that agree there take the pass through
the same probe (`probeStep_congr`), and the state after `k` probes is decided by the oracle on
those strings, against the tree after `k − 1` (`phase_congr`).
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

theorem prefixOf_length (w : FreeMonoid α) : prefixOf w w.toList.length = w := by
  simp [prefixOf]

end Congr

end OrthoDFA
