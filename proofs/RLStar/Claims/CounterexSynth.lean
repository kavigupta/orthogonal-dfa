import Mathlib.Analysis.Asymptotics.Defs
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import RLStar.LowLevelClaims.Level5Claim

/-!
# One L\* round's costs, asymptotically
-/

namespace OrthoDFA

open MeasureTheory Asymptotics Filter Random Random.Kappa

/-- `Q` states, `A` letters, the disagreement rate `ε`, and `L` the probes' longest length. -/
structure CounterexSynthInputs where
  Q : ℕ
  A : ℕ
  ε : ℝ
  δ : ℝ
  L : ℕ

def counterexSynthRegion : Set CounterexSynthInputs :=
  {x | 1 ≤ x.Q ∧ 1 ≤ x.A ∧ 0 < x.δ ∧ x.δ ≤ 1 / 2 ∧ 0 < x.ε ∧ x.ε < 1 ∧ 2 ≤ x.L}

open Real in
def CounterexSynthClaim.{u, v, w} : Prop :=
  ∃ (nprobes : CounterexSynthInputs → ℕ)
    (slope slackRate p₀Tol κTol : CounterexSynthInputs → ℝ),
    (fun x => (nprobes x : ℝ)) =O[𝓟 counterexSynthRegion] (fun x =>
      x.Q ^ 2 * x.A / x.ε * log (x.Q * x.A / x.δ)) ∧
    slope =O[𝓟 counterexSynthRegion] (fun x => x.Q * (1 / x.ε + log x.L)) ∧
    slackRate =O[𝓟 counterexSynthRegion] (fun x =>
      (x.Q * x.A * log (x.Q + 1) + log (1 / x.δ)) / x.ε) ∧
    (fun x => 1 / p₀Tol x) =O[𝓟 counterexSynthRegion] (fun x =>
      (x.Q ^ 2 * x.A / x.ε * log (x.Q * x.A / x.δ)) ^ 2) ∧
    (fun x => 1 / κTol x) =O[𝓟 counterexSynthRegion] (fun x =>
      x.Q ^ 2 * x.A * log (x.Q * x.A / x.δ) / x.δ) ∧
    (∀ x, 0 < p₀Tol x ∧ 0 < κTol x) ∧
    (∀ {A : Type u} [Fintype A] [DecidableEq A] {Q : Type v} [Fintype Q]
      (M : DFA A Q)
      (side : Q → Bool) (U : Q → ℝ) (κ : ℝ) {Ω : Type w} [MeasurableSpace Ω] (μ : Measure Ω)
      [IsProbabilityMeasure μ] (R : KappaReadOracle μ M side U κ) (P : PrefixDistribution A)
      (C : Level3Settings) (p₀ : ℝ),
      let y := C.harvestRate
      let ε := C.disagreementRate
      let δ := C.δ
      let x : CounterexSynthInputs := { Q := Fintype.card Q, A := Fintype.card A, ε, δ, L := P.L }
      x ∈ counterexSynthRegion → C.probeBudget = nprobes x →
      (∀ u, P.D.real {w | prefixOf w C.walkStart = u} ≤ p₀) → p₀ ≤ p₀Tol x →
      0 ≤ κ → κ ≤ κTol x →
      ∫⁻ ω, (Measure.pi fun _ : Fin C.probeBudget => P.D)
          {draws | let result := indecisiveLStarLevel3 C (R.read · ω) draws
            ¬ EndsWell
              { target := M,
                undecided := U,
                harvestSlope := slope x,
                harvestSlack := p₀ * slackRate x,
                probes := P.D,
                disagreeRate := ε,
                disagree := searchSet (R.read · ω) C.walkStart result.hypothesis
                  ∪ endsSet (R.read · ω) C.walkStart result.hypothesis,
                walkStart := C.walkStart,
                harvestLaw := harvestLaw (R.read · ω) P.D C.walkStart result.hypothesis,
                yield := y,
                ending := result.ending }
            ∨ Fintype.card Q + 2 < result.hypothesis.tree.leaves.length} ∂μ
        ≤ ENNReal.ofReal δ)

end OrthoDFA
