import OrthoDFA.Level3Claim

/-!
# `EpsRoundCorrect`: level 3 with rare wrong reads ends well but for a small chance

Level 3 (`indecisiveLStarLevel3`) runs with every read on its state's wrong side with chance at
most `εw`. A probe whose reads, a prefix of length at least `walkStart` followed by a string of
length at most `|Q|`, are all right splits and learns only on real targets, and level 3's claim
holds while every probe is such. A probe that can read one of those strings wrong is at most `Gw`
of the probes but for `wrongRisk`, so some probe of the round is one with chance at most
`probeBudget · Gw`.

`EpsRoundCorrect`: the chance of not ending well is at most `epsRoundRisk`, the least over the
admissible knobs of `epsRoundBound`: `roundBound`, plus `wrongRisk` and `probeBudget · Gw`.
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
    (L : ℕ) (ε y : ℝ) (kn : EpsKnobs) : Prop where
  base : kn.toKnobs.Admissible (σ := σ) (α := α) θ C L ε y
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
  ⨅ (kn : EpsKnobs) (_ : kn.Admissible (σ := σ) (α := α) θ C L ε y),
    ENNReal.ofReal (epsRoundBound (α := α) σ θ εw D L C y kn)

/-- `ReadOracle` with each read on its state's wrong side with chance at most `εw`. -/
structure EpsReadOracle {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) {σ : Type*}
    (M : DFA α σ) (side : σ → Bool) (U : σ → ℝ) (εw : ℝ) where
  read : FreeMonoid α → Ω → ARU
  measurable : ∀ z, Measurable (read z)
  indep : ProbabilityTheory.iIndepFun read μ
  undecided : ∀ z, μ.real {ω | read z ω = .undecided} = U (M.eval z.toList)
  wrong : ∀ z, μ.real {ω | read z ω = if side (M.eval z.toList) then .reject else .accept} ≤ εw

/-- `RandomRoundCorrect`, its yield included, for level 3's algorithm with each read wrong with
chance at most `εw`. -/
def EpsRoundCorrect : Prop :=
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
