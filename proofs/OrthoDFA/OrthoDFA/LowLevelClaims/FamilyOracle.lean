import OrthoDFA.LowLevelClaims.Level5Claim
import OrthoDFA.LowLevelClaims.SuffixFamily

/-!
# Level 5's oracle, read by a suffix family

`FamilyKappaOracle`: a suffix family's reads are a `KappaReadOracle`, each state's side the one it
reads more often (`readSide`), its undecided chance the state's chance of reading undecided, at
`εw = 0` and the family's `κ`.
-/

noncomputable section

namespace OrthoDFA

/-- The side a read distribution reads on more often. -/
def readSide (d : ARU → ℝ) : Bool := decide (d .reject < d .accept)

open MeasureTheory in
def FamilyKappaOracle : Prop :=
  ∀ {α : Type*} [Fintype α] {σ : Type*} {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
    [IsProbabilityMeasure μ] (fam : SuffixFamily μ α σ),
    ∃ R : Random.Kappa.KappaReadOracle μ fam.M (fun q => readSide (fam.readDist q))
      (fun q => fam.readDist q .undecided) 0 fam.κ,
      R.read = fun z ω => fam.read ω z

end OrthoDFA

end
