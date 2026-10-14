import OrthoDFA.Proofs.SuffixFamily

/-!
# Level 5's claim, read by a suffix family

`Level5Correct` with the reads a suffix family's: its oracle (`SuffixFamily.oracle`) is the
family's reads and their distributions, at the band's `κ` and `ε`.
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

/-- The family's reads, as an oracle with their distributions and the band's `κ` and `ε`. -/
def SuffixFamily.oracle {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
    [IsProbabilityMeasure μ] {σ : Type*} (fam : SuffixFamily μ α σ) : TriOracle μ fam.M :=
  ⟨fun z ω => fam.read ω z, fam.read_measurable, fam.read_indep, fam.readDist,
    fam.reads_by_state, fam.κ, fam.ε, fam.readDist_nonneg, fam.readDist_rare⟩

/-- `Level5Guarantee` at the family's oracle, its `κ` and `ε` the band's. -/
def Level5FamilyCorrect : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] [Nonempty α] {σ : Type*} [Fintype σ] {Ω : Type*}
    [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ] (fam : SuffixFamily μ α σ)
    (θ εd : ℝ) (P : PrefixDistribution α) (δ : ℝ) (k : ℕ),
    Level5Guarantee fam.M fam.oracle θ εd P δ k

end OrthoDFA

end
