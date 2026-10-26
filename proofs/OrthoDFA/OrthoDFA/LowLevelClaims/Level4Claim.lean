import OrthoDFA.LowLevelClaims.Level3Claim

/-!
# `Level4RoundCorrect`: level 3 with rare wrong reads, a few tolerated

Level 3 (`indecisiveLStarLevel3`) runs with each read on its state's wrong side with chance at
most `εw` (`EpsReadOracle`), otherwise as `ReadOracle` says. It is level 5 at `κ = 0`: up to
`f + 1` rounds of reads may meet a wrong read, the round still ending well within `|Q| + f + 2`
leaves, under level 3's premises at `|Q| + f` states.
-/

namespace OrthoDFA

namespace Random

namespace Eps

open MeasureTheory

variable {α : Type*}

/-- `ReadOracle` with each read on its state's wrong side with chance at most `εw`. -/
structure EpsReadOracle {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) {σ : Type*}
    (M : DFA α σ) (side : σ → Bool) (U : σ → ℝ) (εw : ℝ) where
  read : FreeMonoid α → Ω → ARU
  measurable : ∀ z, Measurable (read z)
  indep : ProbabilityTheory.iIndepFun read μ
  undecided : ∀ z, μ.real {ω | read z ω = .undecided} = U (M.eval z.toList)
  wrong : ∀ z, μ.real {ω | read z ω = if side (M.eval z.toList) then .reject else .accept} ≤ εw

end Eps

end Random

open MeasureTheory Random Random.Eps

/-- Level 3's algorithm, with reads sometimes wrong, fails to end well within `|Q| + f + 2` leaves
with chance at most `δ`, a few wrong reads tolerated. -/
def Level4RoundCorrect : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] {σ : Type*} [Fintype σ] (M : DFA α σ)
    (side : σ → Bool) (U : σ → ℝ) (εw : ℝ) {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω)
    [IsProbabilityMeasure μ] (R : EpsReadOracle μ M side U εw) (P : PrefixDistribution α)
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
    -- A probe round's reads, a round's cost, and the round's reads.
    let w : ℕ := min (P.L + 1) (3 + Nat.clog 2 P.L) * (nQ + 1)
    let c : ℝ := εw * max w (nQ + 2)
    let b : ℝ := C.probeBudget * w + (nQ + 2) * nα * (nQ + 2) * (nQ + 1)
    -- No length-`walkStart` prefix holds more than `p₀` of the probes.
    (∀ u, P.D.real {x | prefixOf x C.walkStart = u} ≤ p₀) → 0 < δ → δ < 1 → 0 < y →
    y ≤ ε / 48 → ε < 1 →
    36 * S / ε * Real.log (16 * S * (S + 1) / δ) + S ≤ C.probeBudget →
    0 ≤ t → 0 ≤ t' → t * c ≤ 1 - Real.exp (-(t' * c)) →
    -- Two probes share their first `walkStart` letters, or `f + 2` rounds read wrong, with chance
    -- at most `δ/8`.
    C.probeBudget * (C.probeBudget - 1 : ℝ) / 2 * p₀
      + Real.exp (t' * (εw * b)) / (1 + t) ^ (f + 2) ≤ δ / 8 →
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
