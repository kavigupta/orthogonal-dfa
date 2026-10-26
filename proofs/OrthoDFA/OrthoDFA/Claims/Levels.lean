import Mathlib.Analysis.Asymptotics.Defs
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import OrthoDFA.Level2Claim
import OrthoDFA.Level4Claim
import OrthoDFA.Level5Claim

/-!
# One L\* round's costs, asymptotically

Each level's round ends well, but for chance `δ`, at a probe budget within a big-O shape. A claim
per level, `Level2Claim` to `Level5Claim`, each with its own budget; levels 3 to 5 also bound their
harvest's slope and slack. Levels 4 and 5 also need the probes' walk starts spread, every one
holding at most `p₀Tol` of the probes' law, and their reads rarely wrong: at level 4 at most `εwTol`
of the time, at level 5 at most `κTol` times the state's indecision. How small those must be is
bounded below the same way. Both run at no leaves beyond level 3's `|Q| + 2`. Each bound has one
constant over all of `levelRegion` (`level2Region` for level 2). `Level3ClaimSimple` to
`Level5ClaimSimple` state the same with the slope and slack each a single product.
-/

namespace OrthoDFA

open MeasureTheory Asymptotics Filter Random Random.Eps Random.Kappa

/-- A round's inputs: `Q` states, `A` letters, the disagreement rate `ε`, the failure chance `δ`,
the harvest rate `y`, and `L` the probes' longest length. -/
structure LevelInputs where
  Q : ℕ
  A : ℕ
  ε : ℝ
  δ : ℝ
  y : ℝ
  L : ℕ

def levelRegion : Set LevelInputs :=
  {x | 1 ≤ x.Q ∧ 1 ≤ x.A ∧ 0 < x.δ ∧ x.δ ≤ 1 / 2 ∧ 0 < x.y ∧ x.y ≤ x.ε / 48 ∧ x.ε < 1 ∧ 2 ≤ x.L}

/-- Level 2's inputs, which have no probe length. -/
structure Level2Inputs where
  Q : ℕ
  A : ℕ
  ε : ℝ
  δ : ℝ
  y : ℝ

def level2Region : Set Level2Inputs :=
  {x | 1 ≤ x.Q ∧ 1 ≤ x.A ∧ 0 < x.δ ∧ x.δ ≤ 1 / 2 ∧ 0 < x.y ∧ x.y < x.ε ∧ x.ε < 1}

/-- Level 3, with reads undecided at random and never wrong, ending its round well on every
instance in `levelRegion` at budget `nprobes`, harvest slope `slope` and slack `p₀ * slackRate`. -/
def Level3Within.{u, v, w} (nprobes : LevelInputs → ℕ) (slope slackRate : LevelInputs → ℝ) :
    Prop :=
  (∀ {α : Type u} [Fintype α] [DecidableEq α] {σ : Type v} [Fintype σ] (M : DFA α σ)
    (side : σ → Bool) (U : σ → ℝ) {Ω : Type w} [MeasurableSpace Ω] (μ : Measure Ω)
    [IsProbabilityMeasure μ] (R : ReadOracle μ M side U) (P : PrefixDistribution α)
    (C : Level3Settings) (p₀ : ℝ),
    let y := C.harvestRate
    let ε := C.disagreementRate
    let δ := C.δ
    let x : LevelInputs := { Q := Fintype.card σ, A := Fintype.card α, ε, δ, y, L := P.L }
    x ∈ levelRegion → C.probeBudget = nprobes x →
    (∀ u, P.D.real {w | prefixOf w C.walkStart = u} ≤ p₀) →
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
              ending := result.ending }} ∂μ
      ≤ ENNReal.ofReal δ)

/-- Positive tolerances, and level 4, with reads wrong at most `εw` of the time, ending its round
well on every instance in `levelRegion` as `Level3Within` does. -/
def Level4Within.{u, v, w} (nprobes : LevelInputs → ℕ)
    (slope slackRate p₀Tol εwTol : LevelInputs → ℝ) : Prop :=
  (∀ x, 0 < p₀Tol x ∧ 0 < εwTol x) ∧
  (∀ {α : Type u} [Fintype α] [DecidableEq α] {σ : Type v} [Fintype σ] (M : DFA α σ)
    (side : σ → Bool) (U : σ → ℝ) (εw : ℝ) {Ω : Type w} [MeasurableSpace Ω] (μ : Measure Ω)
    [IsProbabilityMeasure μ] (R : EpsReadOracle μ M side U εw) (P : PrefixDistribution α)
    (C : Level3Settings) (p₀ : ℝ),
    let y := C.harvestRate
    let ε := C.disagreementRate
    let δ := C.δ
    let x : LevelInputs := { Q := Fintype.card σ, A := Fintype.card α, ε, δ, y, L := P.L }
    x ∈ levelRegion → C.probeBudget = nprobes x →
    (∀ u, P.D.real {w | prefixOf w C.walkStart = u} ≤ p₀) → p₀ ≤ p₀Tol x →
    0 ≤ εw → εw ≤ εwTol x →
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
          ∨ Fintype.card σ + 2 < result.hypothesis.tree.leaves.length} ∂μ
      ≤ ENNReal.ofReal δ)

/-- Positive tolerances, and level 5, with reads wrong at most `κ` times as often as undecided,
ending its round well on every instance in `levelRegion` as `Level3Within` does. -/
def Level5Within.{u, v, w} (nprobes : LevelInputs → ℕ)
    (slope slackRate p₀Tol κTol : LevelInputs → ℝ) : Prop :=
  (∀ x, 0 < p₀Tol x ∧ 0 < κTol x) ∧
  (∀ {α : Type u} [Fintype α] [DecidableEq α] {σ : Type v} [Fintype σ] (M : DFA α σ)
    (side : σ → Bool) (U : σ → ℝ) (κ : ℝ) {Ω : Type w} [MeasurableSpace Ω] (μ : Measure Ω)
    [IsProbabilityMeasure μ] (R : KappaReadOracle μ M side U 0 κ) (P : PrefixDistribution α)
    (C : Level3Settings) (p₀ : ℝ),
    let y := C.harvestRate
    let ε := C.disagreementRate
    let δ := C.δ
    let x : LevelInputs := { Q := Fintype.card σ, A := Fintype.card α, ε, δ, y, L := P.L }
    x ∈ levelRegion → C.probeBudget = nprobes x →
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
          ∨ Fintype.card σ + 2 < result.hypothesis.tree.leaves.length} ∂μ
      ≤ ENNReal.ofReal δ)

open Real in
/-- Level 3's budget, harvest slope and slack. -/
def Level3Claim.{u, v, w} : Prop :=
  ∃ (nprobes : LevelInputs → ℕ) (slope slackRate : LevelInputs → ℝ),
    (fun x => (nprobes x : ℝ)) =O[𝓟 levelRegion] (fun x =>
      x.Q ^ 2 * x.A / x.ε * log (x.Q ^ 2 * x.A / x.δ)) ∧
    slope =O[𝓟 levelRegion] (fun x => x.Q * (1 / x.y + log x.L)) ∧
    slackRate =O[𝓟 levelRegion] (fun x =>
      (x.Q * x.A * log (x.Q * x.A) + log (1 / x.δ)) / x.y) ∧
    Level3Within.{u, v, w} nprobes slope slackRate

open Real in
/-- `Level3Claim` with each shape a single product. -/
def Level3ClaimSimple.{u, v, w} : Prop :=
  ∃ (nprobes : LevelInputs → ℕ) (slope slackRate : LevelInputs → ℝ),
    (fun x => (nprobes x : ℝ)) =O[𝓟 levelRegion] (fun x =>
      x.Q ^ 2 * x.A / x.ε * log (x.Q ^ 2 * x.A / x.δ)) ∧
    slope =O[𝓟 levelRegion] (fun x => x.Q * log x.L / x.y) ∧
    slackRate =O[𝓟 levelRegion] (fun x => x.Q * x.A * log (x.Q * x.A / x.δ) / x.y) ∧
    Level3Within.{u, v, w} nprobes slope slackRate

open Real in
/-- Level 4's budget, harvest slope and slack, and how spread and how rarely wrong reads must be. -/
def Level4Claim.{u, v, w} : Prop :=
  ∃ (nprobes : LevelInputs → ℕ) (slope slackRate p₀Tol εwTol : LevelInputs → ℝ),
    (fun x => (nprobes x : ℝ)) =O[𝓟 levelRegion] (fun x =>
      x.Q ^ 2 * x.A / x.ε * log (x.Q ^ 2 * x.A / x.δ)) ∧
    slope =O[𝓟 levelRegion] (fun x => x.Q * (1 / x.y + log x.L)) ∧
    slackRate =O[𝓟 levelRegion] (fun x =>
      (x.Q * x.A * log (x.Q * x.A) + log (1 / x.δ)) / x.y) ∧
    (fun x => 1 / p₀Tol x) =O[𝓟 levelRegion] (fun x =>
      (x.Q ^ 2 * x.A / x.ε * log (x.Q ^ 2 * x.A / x.δ)) ^ 2 / x.δ) ∧
    (fun x => 1 / εwTol x) =O[𝓟 levelRegion] (fun x =>
      x.Q ^ 3 * x.A / x.ε * log (x.Q ^ 2 * x.A / x.δ) * log x.L / sqrt x.δ) ∧
    Level4Within.{u, v, w} nprobes slope slackRate p₀Tol εwTol

open Real in
/-- `Level4Claim` with each shape a single product. -/
def Level4ClaimSimple.{u, v, w} : Prop :=
  ∃ (nprobes : LevelInputs → ℕ) (slope slackRate p₀Tol εwTol : LevelInputs → ℝ),
    (fun x => (nprobes x : ℝ)) =O[𝓟 levelRegion] (fun x =>
      x.Q ^ 2 * x.A / x.ε * log (x.Q ^ 2 * x.A / x.δ)) ∧
    slope =O[𝓟 levelRegion] (fun x => x.Q * log x.L / x.y) ∧
    slackRate =O[𝓟 levelRegion] (fun x => x.Q * x.A * log (x.Q * x.A / x.δ) / x.y) ∧
    (fun x => 1 / p₀Tol x) =O[𝓟 levelRegion] (fun x =>
      (x.Q ^ 2 * x.A / x.ε * log (x.Q ^ 2 * x.A / x.δ)) ^ 2 / x.δ) ∧
    (fun x => 1 / εwTol x) =O[𝓟 levelRegion] (fun x =>
      x.Q ^ 3 * x.A / x.ε * log (x.Q ^ 2 * x.A / x.δ) * log x.L / sqrt x.δ) ∧
    Level4Within.{u, v, w} nprobes slope slackRate p₀Tol εwTol

open Real in
/-- Level 5's budget, harvest slope and slack, and how spread and how rarely wrong reads must be. -/
def Level5Claim.{u, v, w} : Prop :=
  ∃ (nprobes : LevelInputs → ℕ) (slope slackRate p₀Tol κTol : LevelInputs → ℝ),
    (fun x => (nprobes x : ℝ)) =O[𝓟 levelRegion] (fun x =>
      x.Q ^ 2 * x.A / x.ε * log (x.Q ^ 2 * x.A / x.δ)) ∧
    slope =O[𝓟 levelRegion] (fun x => x.Q * (1 / x.y + log x.L)) ∧
    slackRate =O[𝓟 levelRegion] (fun x =>
      (x.Q * x.A * log (x.Q * x.A) + log (1 / x.δ)) / x.y) ∧
    (fun x => 1 / p₀Tol x) =O[𝓟 levelRegion] (fun x =>
      (x.Q ^ 2 * x.A / x.ε * log (x.Q ^ 2 * x.A / x.δ)) ^ 2 / x.δ) ∧
    (fun x => 1 / κTol x) =O[𝓟 levelRegion] (fun x =>
      x.Q ^ 2 * x.A * log (x.Q * x.A / x.δ) / sqrt x.δ) ∧
    Level5Within.{u, v, w} nprobes slope slackRate p₀Tol κTol

open Real in
/-- `Level5Claim` with each shape a single product. -/
def Level5ClaimSimple.{u, v, w} : Prop :=
  ∃ (nprobes : LevelInputs → ℕ) (slope slackRate p₀Tol κTol : LevelInputs → ℝ),
    (fun x => (nprobes x : ℝ)) =O[𝓟 levelRegion] (fun x =>
      x.Q ^ 2 * x.A / x.ε * log (x.Q ^ 2 * x.A / x.δ)) ∧
    slope =O[𝓟 levelRegion] (fun x => x.Q * log x.L / x.y) ∧
    slackRate =O[𝓟 levelRegion] (fun x => x.Q * x.A * log (x.Q * x.A / x.δ) / x.y) ∧
    (fun x => 1 / p₀Tol x) =O[𝓟 levelRegion] (fun x =>
      (x.Q ^ 2 * x.A / x.ε * log (x.Q ^ 2 * x.A / x.δ)) ^ 2 / x.δ) ∧
    (fun x => 1 / κTol x) =O[𝓟 levelRegion] (fun x =>
      x.Q ^ 2 * x.A * log (x.Q * x.A / x.δ) / sqrt x.δ) ∧
    Level5Within.{u, v, w} nprobes slope slackRate p₀Tol κTol

open Real in
/-- Level 2's round ends well, but for chance `δ`, at a probe budget within the big-O shape, with
every state's reads all undecided or all on its side. -/
def Level2Claim.{u, v} : Prop :=
  ∃ nprobes₂ : Level2Inputs → ℕ,
    (fun x => (nprobes₂ x : ℝ)) =O[𝓟 level2Region] (fun x =>
      x.Q ^ 2 * x.A * x.ε / (x.ε - x.y) ^ 2 * log (x.Q ^ 2 * x.A / x.δ)) ∧
  (∀ {α : Type u} [Fintype α] [DecidableEq α] {σ : Type v} [Fintype σ] (M : DFA α σ)
    (read : FreeMonoid α → ARU) (side : σ → Bool) (C : Level2.Settings),
    let y := C.harvestRate
    let ε := C.disagreementRate
    let δ := C.δ
    let x : Level2Inputs := { Q := Fintype.card σ, A := Fintype.card α, ε, δ, y }
    Level2.ReadsByState read M side → x ∈ level2Region →
    ∀ (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D],
      (Measure.pi fun _ : Fin (nprobes₂ x) => D).real
          {draws | let result := Level2.indecisiveLStarLevel2 C read draws
            ¬ EndsWell
              { target := M, undecided := Level2.undecidedOf read M, harvestSlope := 1,
                harvestSlack := 0, probes := D, disagreeRate := ε,
                disagree := {w | Disagrees read result.hypothesis C.walkStart w},
                walkStart := C.walkStart,
                harvestLaw := fun _ => Level2.harvestLaw read D C.walkStart result.hypothesis,
                yield := y, ending := result.ending }}
        ≤ δ)

end OrthoDFA
