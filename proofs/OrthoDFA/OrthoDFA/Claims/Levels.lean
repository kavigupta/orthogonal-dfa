import Mathlib.Analysis.Asymptotics.Defs
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import OrthoDFA.LowLevelClaims.Level2Claim
import OrthoDFA.LowLevelClaims.Level4Claim
import OrthoDFA.LowLevelClaims.Level5Claim

/-!
# One L\* round's costs, asymptotically

`LevelsClaim`: each level's round ends well, but for chance `δ`, at a probe budget within the
big-O shapes below. Levels 3 to 5 share one budget, and their harvests' slope and slack are
bounded the same way. Levels 4 and 5 also need the probes' walk starts spread, every one holding
at most `p₀Tol` of the probes' law, and their reads rarely wrong: at level 4 at most `εwTol` of
the time, at level 5 at most `κTol` times the state's indecision. How small those must be is
bounded below the same way. Both run at no leaves beyond level 3's `|Q| + 2`.
-/

namespace OrthoDFA

open MeasureTheory Asymptotics Filter Random Random.Eps Random.Kappa

/-- `(|Q|, |Σ|, 1/ε, 1/δ, 1/y, L)`, `L` the probes' longest length. -/
abbrev LevelVec := ℝ × ℝ × ℝ × ℝ × ℝ × ℝ

/-- `(|Q|, |Σ|, 1/ε, 1/δ, 1/(ε − y))`. -/
abbrev Level2Vec := ℝ × ℝ × ℝ × ℝ × ℝ

open Real in
def LevelsClaim : Prop :=
  ∃ (nprobes : LevelVec → ℕ) (nprobes₂ : Level2Vec → ℕ)
    (slope slackRate p₀Tol κTol εwTol : LevelVec → ℝ),
    (fun v => (nprobes v : ℝ)) =O[atTop] (fun (Q, A, iε, iδ, _, _) =>
      Q ^ 2 * A * iε * log (Q ^ 2 * A * iδ)) ∧
    (fun v => (nprobes₂ v : ℝ)) =O[atTop] (fun (Q, A, iε, iδ, ig) =>
      Q ^ 2 * A * ig ^ 2 / iε * log (Q ^ 2 * A * iδ)) ∧
    slope =O[atTop] (fun (Q, _, _, _, iy, L) => Q * (iy + log L)) ∧
    slackRate =O[atTop] (fun (Q, A, _, iδ, iy, _) => (Q * A * log (Q * A) + log iδ) * iy) ∧
    (fun v => 1 / p₀Tol v) =O[atTop] (fun (Q, A, iε, iδ, _, _) =>
      (Q ^ 2 * A * iε * log (Q ^ 2 * A * iδ)) ^ 2 * iδ) ∧
    (fun v => 1 / κTol v) =O[atTop] (fun (Q, A, _, iδ, _, _) =>
      Q ^ 2 * A * log (Q * A * iδ) * sqrt iδ) ∧
    (fun v => 1 / εwTol v) =O[atTop] (fun (Q, A, iε, iδ, _, L) =>
      Q ^ 3 * A * iε * log (Q ^ 2 * A * iδ) * log L * sqrt iδ) ∧
    (∀ v, 0 < p₀Tol v ∧ 0 < κTol v ∧ 0 < εwTol v) ∧
    -- Level 2: every state's reads all undecided or all on its side.
    (∀ {α : Type*} [Fintype α] [DecidableEq α] {σ : Type*} [Fintype σ] (M : DFA α σ)
      (read : FreeMonoid α → ARU) (side : σ → Bool) (C : Level2.Settings),
      let y := C.harvestRate
      let ε := C.disagreementRate
      let δ := C.δ
      let v : Level2Vec := (Fintype.card σ, Fintype.card α, 1 / ε, 1 / δ, 1 / (ε - y))
      Level2.ReadsByState read M side → 0 < y → y < ε → ε < 1 → 0 < δ → δ < 1 →
      ∀ (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D],
        (Measure.pi fun _ : Fin (nprobes₂ v) => D).real
            {draws | let result := Level2.indecisiveLStarLevel2 C read draws
              ¬ EndsWell
                { target := M, undecided := Level2.undecidedOf read M, harvestSlope := 1,
                  harvestSlack := 0, probes := D, disagreeRate := ε,
                  disagree := {x | Disagrees read result.hypothesis C.walkStart x},
                  walkStart := C.walkStart,
                  harvestLaw := fun _ => Level2.harvestLaw read D C.walkStart result.hypothesis,
                  yield := y, ending := result.ending }}
          ≤ δ) ∧
    -- Level 3: reads undecided at random, never wrong.
    (∀ {α : Type*} [Fintype α] [DecidableEq α] {σ : Type*} [Fintype σ] (M : DFA α σ)
      (side : σ → Bool) (U : σ → ℝ) {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω)
      [IsProbabilityMeasure μ] (R : ReadOracle μ M side U) (P : PrefixDistribution α)
      (C : Level3Settings) (p₀ : ℝ),
      let y := C.harvestRate
      let ε := C.disagreementRate
      let δ := C.δ
      let v : LevelVec := (Fintype.card σ, Fintype.card α, 1 / ε, 1 / δ, 1 / y, P.L)
      C.probeBudget = nprobes v →
      (∀ u, P.D.real {x | prefixOf x C.walkStart = u} ≤ p₀) → 0 < δ → 0 < y →
      y ≤ ε / 48 → ε < 1 →
      ∫⁻ ω, (Measure.pi fun _ : Fin C.probeBudget => P.D)
          {draws | let result := indecisiveLStarLevel3 C (R.read · ω) draws
            ¬ EndsWell
              { target := M,
                undecided := U,
                harvestSlope := slope v,
                harvestSlack := p₀ * slackRate v,
                probes := P.D,
                disagreeRate := ε,
                disagree := searchSet (R.read · ω) C.walkStart result.hypothesis
                  ∪ endsSet (R.read · ω) C.walkStart result.hypothesis,
                walkStart := C.walkStart,
                harvestLaw := harvestLaw (R.read · ω) P.D C.walkStart result.hypothesis,
                yield := y,
                ending := result.ending }} ∂μ
        ≤ ENNReal.ofReal δ) ∧
    -- Level 4: reads wrong at most `εw` of the time.
    (∀ {α : Type*} [Fintype α] [DecidableEq α] {σ : Type*} [Fintype σ] (M : DFA α σ)
      (side : σ → Bool) (U : σ → ℝ) (εw : ℝ) {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω)
      [IsProbabilityMeasure μ] (R : EpsReadOracle μ M side U εw) (P : PrefixDistribution α)
      (C : Level3Settings) (p₀ : ℝ),
      let y := C.harvestRate
      let ε := C.disagreementRate
      let δ := C.δ
      let v : LevelVec := (Fintype.card σ, Fintype.card α, 1 / ε, 1 / δ, 1 / y, P.L)
      C.probeBudget = nprobes v →
      (∀ u, P.D.real {x | prefixOf x C.walkStart = u} ≤ p₀) → p₀ ≤ p₀Tol v →
      0 ≤ εw → εw ≤ εwTol v →
      0 < δ → δ < 1 → 0 < y → y ≤ ε / 48 → ε < 1 →
      ∫⁻ ω, (Measure.pi fun _ : Fin C.probeBudget => P.D)
          {draws | let result := indecisiveLStarLevel3 C (R.read · ω) draws
            ¬ EndsWell
              { target := M,
                undecided := U,
                harvestSlope := slope v,
                harvestSlack := p₀ * slackRate v,
                probes := P.D,
                disagreeRate := ε,
                disagree := searchSet (R.read · ω) C.walkStart result.hypothesis
                  ∪ endsSet (R.read · ω) C.walkStart result.hypothesis,
                walkStart := C.walkStart,
                harvestLaw := harvestLaw (R.read · ω) P.D C.walkStart result.hypothesis,
                yield := y,
                ending := result.ending }
            ∨ Fintype.card σ + 2 < result.hypothesis.tree.leaves.length} ∂μ
        ≤ ENNReal.ofReal δ) ∧
    -- Level 5: reads wrong at most `κ` times as often as undecided.
    (∀ {α : Type*} [Fintype α] [DecidableEq α] {σ : Type*} [Fintype σ] (M : DFA α σ)
      (side : σ → Bool) (U : σ → ℝ) (κ : ℝ) {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω)
      [IsProbabilityMeasure μ] (R : KappaReadOracle μ M side U 0 κ) (P : PrefixDistribution α)
      (C : Level3Settings) (p₀ : ℝ),
      let y := C.harvestRate
      let ε := C.disagreementRate
      let δ := C.δ
      let v : LevelVec := (Fintype.card σ, Fintype.card α, 1 / ε, 1 / δ, 1 / y, P.L)
      C.probeBudget = nprobes v →
      (∀ u, P.D.real {x | prefixOf x C.walkStart = u} ≤ p₀) → p₀ ≤ p₀Tol v →
      0 ≤ κ → κ ≤ κTol v →
      0 < δ → δ < 1 → 0 < y → y ≤ ε / 48 → ε < 1 →
      ∫⁻ ω, (Measure.pi fun _ : Fin C.probeBudget => P.D)
          {draws | let result := indecisiveLStarLevel3 C (R.read · ω) draws
            ¬ EndsWell
              { target := M,
                undecided := U,
                harvestSlope := slope v,
                harvestSlack := p₀ * slackRate v,
                probes := P.D,
                disagreeRate := ε,
                disagree := searchSet (R.read · ω) C.walkStart result.hypothesis
                  ∪ endsSet (R.read · ω) C.walkStart result.hypothesis,
                walkStart := C.walkStart,
                harvestLaw := harvestLaw (R.read · ω) P.D C.walkStart result.hypothesis,
                yield := y,
                ending := result.ending }
            ∨ Fintype.card σ + 2 < result.hypothesis.tree.leaves.length} ∂μ
        ≤ ENNReal.ofReal δ)

end OrthoDFA
