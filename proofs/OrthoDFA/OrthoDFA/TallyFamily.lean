import OrthoDFA.Proofs.SuffixFamily

/-!
# One round of the tally loop, read by a suffix family

`TallyRound` with the reads a suffix family's: its read model is the family's laws
(`SuffixFamily.readModel`), at the band's `κ` and `ε` and the goodness level `θ`.
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

/-- The family's read model: its target, its laws, the band's `κ` and `ε`, and a read-state good
where it reads undecided at most `1.5θ` of the time. -/
def SuffixFamily.readModel {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
    [IsProbabilityMeasure μ] {σ : Type*} (fam : SuffixFamily μ α σ) (θ : ℝ) : ReadModel α σ :=
  ⟨fam.M, fam.law, θ, fam.κ, fam.ε, fam.law_nonneg, fam.law_rare⟩

/-- The family's reads, as an oracle with its read model's laws. -/
def SuffixFamily.readOracle {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
    [IsProbabilityMeasure μ] {σ : Type*} (fam : SuffixFamily μ α σ) (θ : ℝ) :
    ReadOracle μ (fam.readModel θ) :=
  ⟨fun z ω => fam.read ω z, fam.read_measurable, fam.read_indep, fam.read_law⟩

/-- `TallyRoundFamily`: read by a suffix family, the round over `C.budget` probes drawn from `D`,
of length at most `L`, fails to end well with chance at most `roundRisk` of the family's read
model, at any records coefficient `κs ≥ 2κ`. -/
def TallyRoundFamily : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] {σ : Type*} [Fintype σ] {Ω : Type*}
    [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ] (fam : SuffixFamily μ α σ)
    (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (L : ℕ),
    (∀ᵐ x ∂D, x.toList.length ≤ L) → ∀ (C : TallyCfg) (θ κs : ℝ), 2 * fam.κ ≤ κs →
    ∫⁻ ω, (Measure.pi fun _ : Fin C.budget => D)
        {xs | ¬ RunEnds (tallyStep C (fam.cut ω)) (TallyEndsWell (fam.readModel θ) D C (fam.read ω))
          tallyStart (List.ofFn xs)} ∂μ
      ≤ roundRisk (fam.readModel θ) D L C κs

end OrthoDFA

end
