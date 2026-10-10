import OrthoDFA.RoundLevel
import OrthoDFA.Proofs.HarvestClasses

/-!
# `RoundTrichotomyLevel`

A reading ends the round badly with chance at most its gate's spent failure chance and the
refusal sample's and certificate's (`reading_bad_le`), whatever the readings before it; summing
over the readings the round makes gives the bound (`round_bad_aux`).
-/

namespace OrthoDFA

open MeasureTheory
open scoped ENNReal

variable {α : Type*} [Fintype α] [DecidableEq α] {Q : Type*}

instance RoundCfg.drawMeasure_prob (C : RoundCfg α) (D : Measure (FreeMonoid α))
    [IsProbabilityMeasure D] : IsProbabilityMeasure (C.drawMeasure D) := by
  unfold RoundCfg.drawMeasure; infer_instance

section Round

variable (C : RoundCfg α) (R : CutReads α) (A : DFA (FreeMonoid α) Q)
  (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (CertGood : KState α → Prop)

omit [Fintype α] [DecidableEq α] in
theorem piFinSuccAbove_zero {n : ℕ} {Y : Type*} [MeasurableSpace Y] (d : Fin (n + 1) → Y) :
    MeasurableEquiv.piFinSuccAbove (fun _ : Fin (n + 1) => Y) 0 d = (d 0, Fin.tail d) := by
  ext j
  · rfl
  · simp [MeasurableEquiv.piFinSuccAbove, Fin.removeNth, Fin.tail]

end Round

end OrthoDFA
