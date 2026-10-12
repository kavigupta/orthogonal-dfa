import OrthoDFA.Proofs.SuffixFamily

/-!
# One round of the tally loop, read by a suffix family

`TallyRound` with the reads a suffix family's: its oracle (`SuffixFamily.oracle`) is the family's
reads and laws, at the band's `κ` and `ε`.
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

/-- `TallyRoundFamily`: `TallyRound` at the family's oracle. -/
def TallyRoundFamily : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] {σ : Type*} [Fintype σ] {Ω : Type*}
    [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ] (fam : SuffixFamily μ α σ)
    (θ : ℝ) (P : PrefixDistribution α) (C : TallyCfg) (κs : ℝ), 2 * fam.oracle.κ ≤ κs →
    ∫⁻ ω, (Measure.pi fun _ : Fin C.budget => P.D)
        {xs | ¬ RunEnds (tallyStep C fun z => (fam.oracle.read z ω).cut)
          (TallyEndsWell fam.M (fun q => ¬ fam.oracle.Good θ q) P.D C P.start P.L
            (3 / 4 * θ * coverage P.D P.start P.L fam.M (fun q => ¬ fam.oracle.Good θ q)
              / (P.L + 1 - P.start : ℕ)) (fam.oracle.read · ω))
          tallyStart (List.ofFn xs)} ∂μ
      ≤ roundRisk σ fam.oracle.κ fam.oracle.ε θ P.D P.L C κs
        + ENNReal.ofReal (Real.exp
          (-(3 / 2 * θ * coverage P.D P.start P.L fam.M (fun q => ¬ fam.oracle.Good θ q))
            / (16 * P.q)))

end OrthoDFA

end
