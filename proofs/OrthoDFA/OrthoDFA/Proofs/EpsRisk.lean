import OrthoDFA.Level4Claim
import OrthoDFA.Proofs.RandomRisk

/-!
# Level 4's risk at the proof's free parameters

`epsRoundBound` is `roundBound`, plus `wrongRisk`, the chance that the probes that can read wrong a
prefix of length at least `walkStart` followed by a string of length at most `|Q|` exceed `Gw` of
the probes, plus `probeBudget · Gw`, the chance that some probe of the round is one.
`EpsRoundRiskBound` bounds the round by the least of it over the admissible knobs;
`EpsRoundCorrect` picks knobs.
-/

namespace OrthoDFA

namespace Random

namespace Eps

open MeasureTheory

variable {α : Type*}

/-- The chance that the probes that can read wrong, with chance at most `εw` each, one of the
strings a prefix of length at least `k` followed by a string of length at most `nQ` reaches, are
more than `Gw` of the probes, with probes of length at most `L` and no length-`k` prefix above
`p₀`. -/
noncomputable def wrongRisk (nQ nα L : ℕ) (εw p₀ Gw : ℝ) : ℝ :=
  Real.exp (-(Gw - 2 * εw * (L + 1) * ∑ j ∈ Finset.range (nQ + 1), (nα : ℝ) ^ j) / p₀)

/-- Level 3's free parameters and the probes that can read a string wrong, as a share of the
probes. -/
structure EpsKnobs extends Knobs where
  Gw : ℝ

/-- Level 3's `Knobs.Admissible`, and `Gw` nonnegative. -/
structure EpsKnobs.Admissible {σ : Type*} [Fintype σ] [Fintype α] (θ : ℝ) (C : Level3Settings)
    (ε y : ℝ) (kn : EpsKnobs) : Prop where
  base : kn.toKnobs.Admissible (σ := σ) (α := α) θ C ε y
  Gw_nonneg : 0 ≤ kn.Gw

/-- `roundBound`, plus `wrongRisk` and the chance that some probe of the round can read a string
wrong. -/
noncomputable def epsRoundBound (σ : Type*) [Fintype σ] [Fintype α] (θ εw : ℝ)
    (D : Measure (FreeMonoid α)) (L : ℕ) (C : Level3Settings) (y : ℝ) (kn : EpsKnobs) : ℝ :=
  roundBound (α := α) σ θ D L C y kn.toKnobs
    + wrongRisk (Fintype.card σ) (Fintype.card α) L εw (prefixMax D C.walkStart) kn.Gw
    + C.probeBudget * kn.Gw

/-- The least of `epsRoundBound` over the admissible knobs. -/
noncomputable def epsRoundRisk (σ : Type*) [Fintype σ] [Fintype α] (θ εw : ℝ)
    (D : Measure (FreeMonoid α)) (L : ℕ) (C : Level3Settings) (ε y : ℝ) : ENNReal :=
  ⨅ (kn : EpsKnobs) (_ : kn.Admissible (σ := σ) (α := α) θ C ε y),
    ENNReal.ofReal (epsRoundBound (α := α) σ θ εw D L C y kn)

/-- Level 3's algorithm, with each read wrong with chance at most `εw`, fails to end well with
chance at most `epsRoundRisk`. -/
def EpsRoundRiskBound : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] {σ : Type*} [Fintype σ] (M : DFA α σ)
    (side : σ → Bool) (U : σ → ℝ) (θ εw : ℝ) {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω)
    [IsProbabilityMeasure μ] (R : EpsReadOracle μ M side U εw) (P : PrefixDistribution α)
    (C : Level3Settings) (ε y : ℝ),
    ∫⁻ ω, (Measure.pi fun _ : Fin C.probeBudget => P.D)
        {draws | let result := indecisiveLStarLevel3 C (R.read · ω) draws
          ¬ EndsWell M (BadAt U θ) P.D ε
            {x | Disagrees (R.read · ω) result.hypothesis C.walkStart x}
            C.walkStart (harvestLaw (R.read · ω) P.D C.walkStart result.hypothesis) y
            result.ending} ∂μ
      ≤ epsRoundRisk (α := α) σ θ εw P.D P.L C ε y

end Eps

end Random

end OrthoDFA
