import OrthoDFA.Proofs.SuffixFamily

/-!
# One round of the tally loop, read by a suffix family

`TallyRoundCorrect` with the reads a suffix family's: its oracle (`SuffixFamily.oracle`) is the
family's reads and laws, at the band's `κ` and `ε`.
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

/-- The family's reads, as an oracle with its laws and the band's `κ` and `ε`. -/
def SuffixFamily.oracle {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
    [IsProbabilityMeasure μ] {σ : Type*} (fam : SuffixFamily μ α σ) : TriOracle μ fam.M :=
  ⟨fun z ω => fam.read ω z, fam.read_measurable, fam.read_indep, fam.law, fam.read_law, fam.κ,
    fam.ε, fam.law_nonneg, fam.law_rare⟩

/-- `TallyRoundFamily`: `TallyRoundCorrect` at the family's oracle, its `κ` and `ε` the band's. -/
def TallyRoundFamily : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] [Nonempty α] {σ : Type*} [Fintype σ] {Ω : Type*}
    [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ] (fam : SuffixFamily μ α σ)
    (θ εd : ℝ) (P : PrefixDistribution α) (δ : ℝ) (S k : ℕ),
    TallyRoundGuarantee fam.M fam.oracle θ εd P δ S k

end OrthoDFA

end
