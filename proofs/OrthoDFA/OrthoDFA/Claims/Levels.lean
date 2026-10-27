import Mathlib.Analysis.Asymptotics.Defs
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import OrthoDFA.LowLevelClaims.Level2Claim
import OrthoDFA.LowLevelClaims.Level4Claim
import OrthoDFA.LowLevelClaims.Level5Claim

/-!
# One L\* round's costs, asymptotically

A claim per level, `Level2Claim` to `Level5Claim`: the level's round ends well, but for chance
`δ`, at a probe budget within a big-O shape, its harvest rate fixed at `ε/48` (`ε/2` at level 2).
Levels 3 to 5 also bound their harvest's slope and slack, and levels 4 and 5 need the probes' walk
starts spread, every one holding at most `p₀Tol` of the probes' law, and their reads rarely wrong:
at level 4 at most `εwTol` of the time, at level 5 at most `κTol` times the state's indecision. How
small those must be is bounded the same way. Both may grow their tree `spurious = O(log(1/δ))`
leaves past level 3's `|Q| + 2`, their costs growing with `|Q| + log(1/δ)` in place of `|Q|`. Each
bound has one constant over all of `levelRegion` (`level2Region` for level 2).
-/

namespace OrthoDFA

open MeasureTheory Asymptotics Filter Random Random.Eps Random.Kappa

/-- Level 2's inputs, which have no probe length. -/
structure Level2Inputs where
  Q : ℕ
  A : ℕ
  ε : ℝ
  δ : ℝ

def level2Region : Set Level2Inputs :=
  {x | 1 ≤ x.Q ∧ 1 ≤ x.A ∧ 0 < x.δ ∧ x.δ ≤ 1 / 2 ∧ 0 < x.ε ∧ x.ε < 1}

open Real in
/-- Level 2's round ends well, but for chance `δ`, at a probe budget within the big-O shape, with
every state's reads all undecided or all on its side. -/
def Level2Claim.{u, v} : Prop :=
  ∃ nprobes₂ : Level2Inputs → ℕ,
    (fun x => (nprobes₂ x : ℝ)) =O[𝓟 level2Region] (fun x =>
      x.Q ^ 2 * x.A / x.ε * log (x.Q * x.A / x.δ)) ∧
    (∀ {A : Type u} [Fintype A] [DecidableEq A] {Q : Type v} [Fintype Q]
      (M : DFA A Q)
      (read : FreeMonoid A → ARU) (side : Q → Bool) (C : Level2.Settings),
      let y := C.harvestRate
      let ε := C.disagreementRate
      let δ := C.δ
      let x : Level2Inputs := { Q := Fintype.card Q, A := Fintype.card A, ε, δ }
      Level2.ReadsByState read M side → x ∈ level2Region → y = ε / 2 →
      ∀ (D : Measure (FreeMonoid A)) [IsProbabilityMeasure D],
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

/-- A round's inputs: `Q` states, `A` letters, the disagreement rate `ε`, the failure chance `δ`,
and `L` the probes' longest length. -/
structure LevelInputs where
  Q : ℕ
  A : ℕ
  ε : ℝ
  δ : ℝ
  L : ℕ

def levelRegion : Set LevelInputs :=
  {x | 1 ≤ x.Q ∧ 1 ≤ x.A ∧ 0 < x.δ ∧ x.δ ≤ 1 / 2 ∧ 0 < x.ε ∧ x.ε < 1 ∧ 2 ≤ x.L}

open Real in
/-- Level 3, with reads undecided at random and never wrong: its budget, harvest slope and
slack, and its round ending well on every instance in `levelRegion` at them. -/
def Level3Claim.{u, v, w} : Prop :=
  ∃ (nprobes : LevelInputs → ℕ) (slope slackRate : LevelInputs → ℝ),
    (fun x => (nprobes x : ℝ)) =O[𝓟 levelRegion] (fun x =>
      x.Q ^ 2 * x.A / x.ε * log (x.Q * x.A / x.δ)) ∧
    slope =O[𝓟 levelRegion] (fun x => x.Q * (1 / x.ε + log x.L)) ∧
    slackRate =O[𝓟 levelRegion] (fun x =>
      (x.Q * x.A * log (x.Q * x.A) + log (1 / x.δ)) / x.ε) ∧
    (∀ {A : Type u} [Fintype A] [DecidableEq A] {Q : Type v} [Fintype Q]
      (M : DFA A Q)
      (side : Q → Bool) (U : Q → ℝ) {Ω : Type w} [MeasurableSpace Ω] (μ : Measure Ω)
      [IsProbabilityMeasure μ] (R : ReadOracle μ M side U) (P : PrefixDistribution A)
      (C : Level3Settings) (p₀ : ℝ),
      let y := C.harvestRate
      let ε := C.disagreementRate
      let δ := C.δ
      let x : LevelInputs := { Q := Fintype.card Q, A := Fintype.card A, ε, δ, L := P.L }
      x ∈ levelRegion → y = ε / 48 → C.probeBudget = nprobes x →
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

open Real in
/-- Level 4, with reads wrong at most `εw` of the time: its budget, harvest slope and slack, the
largest `p₀` and `εw` it tolerates, and its round ending well at them within `spurious` leaves past
`|Q| + 2`. -/
def Level4Claim.{u, v, w} : Prop :=
  ∃ (nprobes spurious : LevelInputs → ℕ) (slope slackRate p₀Tol εwTol : LevelInputs → ℝ),
    (fun x => (nprobes x : ℝ)) =O[𝓟 levelRegion] (fun x =>
      (x.Q + log (1 / x.δ)) ^ 2 * x.A / x.ε * log (x.Q * x.A / x.δ)) ∧
    slope =O[𝓟 levelRegion] (fun x => (x.Q + log (1 / x.δ)) * (1 / x.ε + log x.L)) ∧
    slackRate =O[𝓟 levelRegion] (fun x =>
      (x.Q + log (1 / x.δ)) * x.A * log ((x.Q + log (1 / x.δ)) * x.A) / x.ε) ∧
    (fun x => 1 / p₀Tol x) =O[𝓟 levelRegion] (fun x =>
      ((x.Q + log (1 / x.δ)) ^ 2 * x.A / x.ε * log (x.Q * x.A / x.δ)) ^ 2 / x.δ) ∧
    (fun x => 1 / εwTol x) =O[𝓟 levelRegion] (fun x =>
      (x.Q + log (1 / x.δ)) ^ 3 * x.A / x.ε * log (x.Q * x.A / x.δ) * log x.L) ∧
    (fun x => (spurious x : ℝ)) =O[𝓟 levelRegion] (fun x => log (1 / x.δ)) ∧
    (∀ x, 0 < p₀Tol x ∧ 0 < εwTol x) ∧
    (∀ {A : Type u} [Fintype A] [DecidableEq A] {Q : Type v} [Fintype Q]
      (M : DFA A Q)
      (side : Q → Bool) (U : Q → ℝ) (εw : ℝ) {Ω : Type w} [MeasurableSpace Ω] (μ : Measure Ω)
      [IsProbabilityMeasure μ] (R : EpsReadOracle μ M side U εw) (P : PrefixDistribution A)
      (C : Level3Settings) (p₀ : ℝ),
      let y := C.harvestRate
      let ε := C.disagreementRate
      let δ := C.δ
      let x : LevelInputs := { Q := Fintype.card Q, A := Fintype.card A, ε, δ, L := P.L }
      x ∈ levelRegion → y = ε / 48 → C.probeBudget = nprobes x →
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
            ∨ Fintype.card Q + spurious x + 2 < result.hypothesis.tree.leaves.length} ∂μ
        ≤ ENNReal.ofReal δ)

open Real in
/-- Level 5, with reads wrong at most `κ` times as often as undecided: its budget, harvest slope
and slack, the largest `p₀` and `κ` it tolerates, and its round ending well at them within
`spurious` leaves past `|Q| + 2`. -/
def Level5Claim.{u, v, w} : Prop :=
  ∃ (nprobes spurious : LevelInputs → ℕ) (slope slackRate p₀Tol κTol : LevelInputs → ℝ),
    (fun x => (nprobes x : ℝ)) =O[𝓟 levelRegion] (fun x =>
      (x.Q + log (1 / x.δ)) ^ 2 * x.A / x.ε * log (x.Q * x.A / x.δ)) ∧
    slope =O[𝓟 levelRegion] (fun x => (x.Q + log (1 / x.δ)) * (1 / x.ε + log x.L)) ∧
    slackRate =O[𝓟 levelRegion] (fun x =>
      (x.Q + log (1 / x.δ)) * x.A * log ((x.Q + log (1 / x.δ)) * x.A) / x.ε) ∧
    (fun x => 1 / p₀Tol x) =O[𝓟 levelRegion] (fun x =>
      ((x.Q + log (1 / x.δ)) ^ 2 * x.A / x.ε * log (x.Q * x.A / x.δ)) ^ 2 / x.δ) ∧
    (fun x => 1 / κTol x) =O[𝓟 levelRegion] (fun x =>
      (x.Q + log (1 / x.δ)) ^ 2 * x.A * log (x.Q * x.A / x.δ)) ∧
    (fun x => (spurious x : ℝ)) =O[𝓟 levelRegion] (fun x => log (1 / x.δ)) ∧
    (∀ x, 0 < p₀Tol x ∧ 0 < κTol x) ∧
    (∀ {A : Type u} [Fintype A] [DecidableEq A] {Q : Type v} [Fintype Q]
      (M : DFA A Q)
      (side : Q → Bool) (U : Q → ℝ) (κ : ℝ) {Ω : Type w} [MeasurableSpace Ω] (μ : Measure Ω)
      [IsProbabilityMeasure μ] (R : KappaReadOracle μ M side U κ) (P : PrefixDistribution A)
      (C : Level3Settings) (p₀ : ℝ),
      let y := C.harvestRate
      let ε := C.disagreementRate
      let δ := C.δ
      let x : LevelInputs := { Q := Fintype.card Q, A := Fintype.card A, ε, δ, L := P.L }
      x ∈ levelRegion → y = ε / 48 → C.probeBudget = nprobes x →
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
            ∨ Fintype.card Q + spurious x + 2 < result.hypothesis.tree.leaves.length} ∂μ
        ≤ ENNReal.ofReal δ)

end OrthoDFA
