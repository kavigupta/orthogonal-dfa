import OrthoDFA.Level3Claim

/-!
# `Level4RoundCorrect`: level 3 with rare wrong reads ends well but for a small chance

Level 3 (`indecisiveLStarLevel3`) runs with each read on its state's wrong side with chance at
most `εw` (`EpsReadOracle`), otherwise as `ReadOracle` says. A step that reads no string wrong
splits and learns only on real targets, and level 3's claim holds while every step is such.

`Level4RoundCorrect` asks level 3's premises, the sampler's bounds holding but for chance `δ/16` at
each site rather than `δ/8`, and that wrong reads are rare enough that no string the round reads is
read wrong but for chance `δ/8`.
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

/-- Level 3's algorithm, with reads sometimes wrong, fails to end well with chance at most `δ`. -/
def Level4RoundCorrect : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] {σ : Type*} [Fintype σ] (M : DFA α σ)
    (side : σ → Bool) (U : σ → ℝ) (εw : ℝ) {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω)
    [IsProbabilityMeasure μ] (R : EpsReadOracle μ M side U εw) (P : PrefixDistribution α)
    (C : Level3Settings) (p₀ : ℝ),
    let y := C.harvestRate
    let ε := C.disagreementRate
    let δ := C.δ
    let nQ := Fintype.card σ
    let nα := Fintype.card α
    -- Bounds the hypotheses: at most `|Q| + 2` leaves, each leaf's edges learned once between splits.
    let S : ℕ := nQ * ((nQ + 2) * nα + 1) + 2 * nα + 1
    -- Counts the trees level 3 can grow, with an edge map into each one's leaves.
    let K : ℕ := (1 + (nQ + 1) ^ 3 * nα) ^ nQ * (nQ + 3) ^ ((nQ + 2) * nα)
    -- As at level 3.
    let A : ℝ := (nQ + 1) * (6 / y + 16 * (2 + Nat.clog 2 P.L))
    -- As at level 3, at chance `δ/16` at each site.
    let η := p₀ * Real.log (16 * (nQ + 1) * K / δ)
    -- No length-`walkStart` prefix holds more than `p₀` of the probes.
    (∀ u, P.D.real {x | prefixOf x C.walkStart = u} ≤ p₀) → 0 < δ → 0 < y →
    -- The middles test's yield fits under the success test's rate.
    y ≤ ε / 48 → ε < 1 →
    36 * S / ε * Real.log (16 * S * (S + 1) / δ) + S ≤ C.probeBudget →
    -- A probe sifts at most `min(L + 1, 3 + ⌈log₂ L⌉)` of its prefixes, each reading at most `|Q|`
    -- strings, and at most `|Q|` splits re-sift `(|Q| + 1)²|Σ|` each, so the round reads some
    -- string wrong with chance at most `δ/8`.
    εw * (C.probeBudget * min (P.L + 1) (3 + Nat.clog 2 P.L) * nQ + nQ * (nQ + 1) ^ 2 * nα)
      ≤ δ / 8 →
    ∫⁻ ω, (Measure.pi fun _ : Fin C.probeBudget => P.D)
        {draws | let result := indecisiveLStarLevel3 C (R.read · ω) draws
          -- As at level 3.
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
              ending := result.ending }} ∂μ
      ≤ ENNReal.ofReal δ

end OrthoDFA
