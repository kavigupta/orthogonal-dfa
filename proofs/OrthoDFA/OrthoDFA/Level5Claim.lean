import OrthoDFA.Utils.Allowance

/-!
# `Level5RoundCorrect`: level 3 with wrong reads that scale with indecision, a few tolerated

Level 3 (`indecisiveLStarLevel3`) runs with each read on its state's wrong side with chance at
most `κ U q`, `U q` its state's chance of reading undecided (`KappaReadOracle`), otherwise as
`ReadOracle` says. The round may read up to `f + 1` rounds wrong and still end well, its tree
within `|Q| + f + 2` leaves. A wrong read adds at most one spurious leaf, so with level 3's premises
taken at `|Q| + f` states (`Level3AllowanceCorrect`) the round fails only where `f + 2` rounds of
reads meet a fresh wrong read (`RaceTailBound`), or two probes share their first `walkStart`
letters, so that one probe's wrong read could recur in another's.

A step reads in rounds: its probe's reads and one re-sift per edge of a split, each reading at most
one string undecided, so costing at most `κ`. Over the round the strings read undecided number at
most `m`.
-/

namespace OrthoDFA

namespace Random

namespace Kappa

open MeasureTheory

variable {α : Type*}

/-- `ReadOracle` with each read on its state's wrong side with chance at most `κ U q`. -/
structure KappaReadOracle {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) {σ : Type*}
    (M : DFA α σ) (side : σ → Bool) (U : σ → ℝ) (κ : ℝ) where
  read : FreeMonoid α → Ω → ARU
  measurable : ∀ z, Measurable (read z)
  indep : ProbabilityTheory.iIndepFun read μ
  undecided : ∀ z, μ.real {ω | read z ω = .undecided} = U (M.eval z.toList)
  wrong : ∀ z, μ.real {ω | read z ω = if side (M.eval z.toList) then .reject else .accept}
    ≤ κ * U (M.eval z.toList)

end Kappa

end Random

open MeasureTheory Random Random.Kappa

/-- Level 3's algorithm, with wrong reads that may scale with a state's indecision, fails to end
well within `|Q| + f + 2` leaves with chance at most `δ`, a few wrong reads tolerated. -/
def Level5RoundCorrect : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] {σ : Type*} [Fintype σ] (M : DFA α σ)
    (side : σ → Bool) (U : σ → ℝ) (κ : ℝ) {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω)
    [IsProbabilityMeasure μ] (R : KappaReadOracle μ M side U κ) (P : PrefixDistribution α)
    (C : Level3Settings) (p₀ : ℝ) (f : ℕ) (t t' : ℝ),
    let y := C.harvestRate
    let ε := C.disagreementRate
    let δ := C.δ
    let nQ := Fintype.card σ + f
    let nα := Fintype.card α
    -- As at level 3, at `|Q| + f` states.
    let S : ℕ := nQ * ((nQ + 2) * nα + 1) + 2 * nα + 1
    let K : ℕ := (1 + (nQ + 1) ^ 3 * nα) ^ nQ * (nQ + 3) ^ ((nQ + 2) * nα)
    let A : ℝ := (nQ + 1) * (6 / y + 16 * (2 + Nat.clog 2 P.L))
    let η := p₀ * Real.log (16 * (nQ + 1) * K / δ)
    -- The round's reads undecided.
    let m : ℝ := 16 / 5 * S * Real.log (8 * S * (S + 1) / δ) + 1 + (nQ + 2) * nα * (nQ + 1)
    -- No length-`walkStart` prefix holds more than `p₀` of the probes.
    (∀ u, P.D.real {x | prefixOf x C.walkStart = u} ≤ p₀) → 0 < δ → δ < 1 → 0 < y →
    y ≤ ε / 48 → ε < 1 →
    36 * S / ε * Real.log (16 * S * (S + 1) / δ) + S ≤ C.probeBudget →
    0 ≤ κ → 0 ≤ t → 0 ≤ t' → t * κ ≤ 1 - Real.exp (-(t' * κ)) →
    -- Two probes share their first `walkStart` letters, or `f + 2` rounds read wrong, with chance
    -- at most `δ/8`.
    C.probeBudget * (C.probeBudget - 1 : ℝ) / 2 * p₀
      + Real.exp (t' * (κ * m)) / (1 + t) ^ (f + 2) ≤ δ / 8 →
    ∫⁻ ω, (Measure.pi fun _ : Fin C.probeBudget => P.D)
        {draws | let result := indecisiveLStarLevel3 C (R.read · ω) draws
          ¬ EndsWell
            { target := M,
              undecided := U,
              harvestSlope := A,
              harvestSlack := η / y,
              probes := P.D,
              disagreeRate := ε,
              disagree := searchSet (R.read · ω) C.walkStart result.hypothesis
                ∪ endsSet (R.read · ω) C.walkStart result.hypothesis,
              walkStart := C.walkStart,
              harvestLaw := harvestLaw (R.read · ω) P.D C.walkStart result.hypothesis,
              yield := y,
              ending := result.ending }
          ∨ nQ + 2 < result.hypothesis.tree.leaves.length} ∂μ
      ≤ ENNReal.ofReal δ

end OrthoDFA
