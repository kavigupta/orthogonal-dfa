import OrthoDFA.Level4KappaClaim
import OrthoDFA.AllowanceClaim

/-!
# `Level5RoundCorrect`: level 3 with wrong reads that scale with indecision, a few tolerated

Level 3 (`indecisiveLStarLevel3`) runs on level 5's reads (`KappaReadOracle`). `Level4KappaRoundCorrect` charges
the round its chance of reading anything wrong; here the round may read up to `f + 1` rounds
wrong and still end well. A wrong read adds at most one spurious leaf, so with level 3's premises
taken at `|Q| + f` states (`Level3AllowanceCorrect`) the round fails only where `f + 2` rounds of
reads meet a fresh wrong read (`RaceTailBound`), or two probes share their first `walkStart`
letters, so that one probe's wrong read could recur in another's.

A step reads in rounds: its probe's reads, at most `w = min(L + 1, 3 + ⌈log₂ L⌉)(|Q| + f + 1)`
strings and one undecided, and one re-sift per edge of a split, at most `|Q| + f + 2` strings and
one undecided. A round costs at most `c = εw max(w, |Q| + f + 2) + κ`. Over the round the strings
read number at most `b` and those read undecided at most `m`.
-/

namespace OrthoDFA

open MeasureTheory Random Random.Kappa

/-- Level 3's algorithm, with wrong reads that may scale with a state's indecision, fails to end
well with chance at most `δ`, a few wrong reads tolerated. -/
def Level5RoundCorrect : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] {σ : Type*} [Fintype σ] (M : DFA α σ)
    (side : σ → Bool) (U : σ → ℝ) (εw κ : ℝ) {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω)
    [IsProbabilityMeasure μ] (R : KappaReadOracle μ M side U εw κ) (P : PrefixDistribution α)
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
    -- A probe round's reads, a round's cost, and the round's reads, in all and undecided.
    let w : ℕ := min (P.L + 1) (3 + Nat.clog 2 P.L) * (nQ + 1)
    let c : ℝ := εw * max w (nQ + 2) + κ
    let b : ℝ := C.probeBudget * w + (nQ + 2) * nα * (nQ + 2) * (nQ + 1)
    let m : ℝ := 16 / 5 * S * Real.log (8 * S * (S + 1) / δ) + 1 + (nQ + 2) * nα * (nQ + 1)
    -- No length-`walkStart` prefix holds more than `p₀` of the probes.
    (∀ u, P.D.real {x | prefixOf x C.walkStart = u} ≤ p₀) → 0 < δ → δ < 1 → 0 < y →
    y ≤ ε / 48 → ε < 1 →
    36 * S / ε * Real.log (16 * S * (S + 1) / δ) + S ≤ C.probeBudget →
    0 ≤ εw → 0 ≤ κ → 0 ≤ t → 0 ≤ t' → t * c ≤ 1 - Real.exp (-(t' * c)) →
    -- Two probes share their first `walkStart` letters, or `f + 2` rounds read wrong, with chance
    -- at most `δ/8`.
    C.probeBudget * (C.probeBudget - 1 : ℝ) / 2 * p₀
      + Real.exp (t' * (εw * b + κ * m)) / (1 + t) ^ (f + 2) ≤ δ / 8 →
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
              ending := result.ending }} ∂μ
      ≤ ENNReal.ofReal δ

end OrthoDFA
