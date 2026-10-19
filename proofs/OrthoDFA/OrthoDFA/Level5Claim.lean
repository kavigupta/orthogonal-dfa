import OrthoDFA.Level4Claim

/-!
# `KappaRoundCorrect`: level 3 with wrong reads that scale with indecision

Level 3 (`indecisiveLStarLevel3`) runs with each read on its state's wrong side with chance at
most `εw + κ U q`, `U q` its state's chance of reading undecided (`KappaReadOracle`), otherwise as
`ReadOracle` says. A wrong read can steer the round only once the round reads its string, and a
string is read wrong at most `κ` times as often as undecided, beyond `εw`. A probe's sifts stop at their
first undecided read, so each probe reads at most one string undecided, and the tests count it:
the tests bound the undecided reads a round makes, whatever the budget, and with them the wrong
ones.

`KappaRoundCorrect` asks level 3's premises, the good states' bounds holding but for chance `δ/16`
each rather than `δ/8`, and that the strings the round reads, and those it reads undecided, are few
enough against `εw` and `κ` that no string the round reads is read wrong but for chance `δ/8`.
-/

namespace OrthoDFA

namespace Random

namespace Kappa

open MeasureTheory

variable {α : Type*}

/-- `ReadOracle` with each read on its state's wrong side with chance at most `εw + κ U q`. -/
structure KappaReadOracle {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) {σ : Type*}
    (M : DFA α σ) (side : σ → Bool) (U : σ → ℝ) (εw κ : ℝ) where
  read : FreeMonoid α → Ω → ARU
  measurable : ∀ z, Measurable (read z)
  indep : ProbabilityTheory.iIndepFun read μ
  undecided : ∀ z, μ.real {ω | read z ω = .undecided} = U (M.eval z.toList)
  wrong : ∀ z, μ.real {ω | read z ω = if side (M.eval z.toList) then .reject else .accept}
    ≤ εw + κ * U (M.eval z.toList)

/-- `RandomRoundCorrect` for level 3's algorithm with each read wrong with chance at most
`εw + κ U q`, the good states' bounds holding but for chance `δ/16` each. A probe reads at most
`(L + 1)|Q|` strings and a split re-sifts at most `(|Q| + 1)²|Σ|`, at most `|Q|` times. Under the
`j`-th hypothesis, until a test fires, the probes stopping undecided number at most
`5 log(1/a_j)`, `a_j` the tests' level, whatever the stretch's length; so the round reads at most
`5 S log(8S(S + 1)/δ) + 1 + |Q|(|Q| + 1)²|Σ|` strings undecided, and the last premise keeps the
chance that it reads some string wrong within `δ/8`. -/
def KappaRoundCorrect : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] {σ : Type*} [Fintype σ] (M : DFA α σ)
    (side : σ → Bool) (U : σ → ℝ) (θ εw κ : ℝ) {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω)
    [IsProbabilityMeasure μ] (R : KappaReadOracle μ M side U εw κ) (P : PrefixDistribution α)
    (C : Level3Settings) (p₀ : ℝ),
    let y := C.harvestRate
    let ε := C.disagreementRate
    let δ := C.δ
    let nQ := Fintype.card σ
    let nα := Fintype.card α
    let S : ℕ := nQ * ((nQ + 2) * nα + 1) + 2 * nα + 1
    let K : ℕ := (1 + (nQ + 1) ^ 3 * nα) ^ nQ * (nQ + 3) ^ ((nQ + 2) * nα)
    let G := 9 * θ * (nQ + 1) + 2 * p₀ * Real.log (16 * K / δ)
    (∀ u, P.D.real {x | prefixOf x C.walkStart = u} ≤ p₀) → 0 ≤ θ → 0 < δ →
    48 * (P.L + 1) * (nQ + 1) * θ ≤ 1 → 2 * G ≤ y → y ≤ ε / 48 → ε < 1 →
    36 * S / ε * Real.log (16 * S * (S + 1) / δ) + S ≤ C.probeBudget →
    0 ≤ εw → 0 ≤ κ →
    εw * (C.probeBudget * (P.L + 1) * nQ + nQ * (nQ + 1) ^ 2 * nα)
      + κ * (5 * S * Real.log (8 * S * (S + 1) / δ) + 1 + nQ * (nQ + 1) ^ 2 * nα) ≤ δ / 8 →
    ∫⁻ ω, (Measure.pi fun _ : Fin C.probeBudget => P.D)
        {draws | let result := indecisiveLStarLevel3 C (R.read · ω) draws
          ¬ EndsWell M (BadAt U θ) P.D ε
            (searchSet (R.read · ω) C.walkStart result.hypothesis
              ∪ endsSet (R.read · ω) C.walkStart result.hypothesis)
            C.walkStart (harvestLaw (R.read · ω) P.D C.walkStart result.hypothesis) y
            result.ending} ∂μ
      ≤ ENNReal.ofReal δ

end Kappa

end Random

end OrthoDFA
