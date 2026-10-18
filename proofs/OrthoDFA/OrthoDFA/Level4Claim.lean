import OrthoDFA.Level3Claim

/-!
# `EpsRoundCorrect`: level 3 with rare wrong reads ends well but for a small chance

Level 3 (`indecisiveLStarLevel3`) runs with each read on its state's wrong side with chance at
most `εw` (`EpsReadOracle`), otherwise as `ReadOracle` says. A step that reads no string wrong
splits and learns only on real targets, and level 3's claim holds while every step is such.

`EpsRoundCorrect` asks level 3's premises, `G` holding but for chance `δ/8` rather than `δ/4`, and
that wrong reads are rare enough that no string the round reads is read wrong but for chance
`δ/8`.
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

/-- `RandomRoundCorrect` for level 3's algorithm with each read wrong with chance at most `εw`, `G`
holding but for chance `δ/8`. A probe reads at most `(L + 1)|Q|` strings and a split re-sifts at
most `(|Q| + 1)²|Σ|`, each read wrong with chance at most `εw` when first read, so the last premise
keeps the chance that the round reads some string wrong within `δ/8`. -/
def EpsRoundCorrect : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] {σ : Type*} [Fintype σ] (M : DFA α σ)
    (side : σ → Bool) (U : σ → ℝ) (θ εw : ℝ) {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω)
    [IsProbabilityMeasure μ] (R : EpsReadOracle μ M side U εw) (P : PrefixDistribution α)
    (C : Level3Settings) (p₀ : ℝ),
    let y := C.harvestRate
    let ε := C.disagreementRate
    let δ := C.δ
    let nQ := Fintype.card σ
    let nα := Fintype.card α
    let S : ℕ := nQ * ((nQ + 2) * nα + 1) + 2 * nα + 1
    let G := 3 * θ * (P.L + 1) * (nQ + 1) + p₀ * Real.log (8 * (1 + (nQ + 1) ^ 3 * nα) ^ nQ / δ)
    (∀ u, P.D.real {x | prefixOf x C.walkStart = u} ≤ p₀) → 0 ≤ θ → 0 < δ →
    2 * G ≤ y → y ≤ ε / 48 → ε < 1 →
    36 * S / ε * Real.log (16 * S * (S + 1) / δ) + S ≤ C.probeBudget →
    εw * C.probeBudget * ((P.L + 1) * nQ + (nQ + 1) ^ 2 * nα) ≤ δ / 8 →
    ∫⁻ ω, (Measure.pi fun _ : Fin C.probeBudget => P.D)
        {draws | let result := indecisiveLStarLevel3 C (R.read · ω) draws
          ¬ EndsWell M (BadAt U θ) P.D ε
            (searchSet (R.read · ω) C.walkStart result.hypothesis
              ∪ endsSet (R.read · ω) C.walkStart result.hypothesis)
            C.walkStart (harvestLaw (R.read · ω) P.D C.walkStart result.hypothesis) y
            result.ending} ∂μ
      ≤ ENNReal.ofReal δ

end Eps

end Random

end OrthoDFA
