import OrthoDFA.Level4Claim

/-!
# `Level5RoundCorrect`: level 3 with wrong reads that scale with indecision

Level 3 (`indecisiveLStarLevel3`) runs with each read on its state's wrong side with chance at
most `εw + κ U q`, `U q` its state's chance of reading undecided (`KappaReadOracle`), otherwise as
`ReadOracle` says. A wrong read can steer the round only once the round reads its string, and a
string is read wrong at most `κ` times as often as undecided, beyond `εw`. A probe's sifts stop at their
first undecided read, so each probe reads at most one string undecided, and the tests count it:
the tests bound the undecided reads a round makes, whatever the budget, and with them the wrong
ones.

`Level5RoundCorrect` asks level 3's premises, the good states' bounds holding but for chance `δ/16`
each rather than `δ/8`, and that the strings the round reads, and those it reads undecided, are few
enough against `εw` and `κ` that no string the round reads is read wrong but for chance `δ/8`.

`KappaRoundCorrectAt` reads the probes' prefix mass at a depth `kd` past the walk's start, as a
walk from `0` needs. A string read undecided at a good state before `kd` can then only cost a
harvest that does not end well, a wasted round, never a consistent ending.
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

end Kappa

end Random

open MeasureTheory Random Random.Kappa

/-- Level 3's algorithm, with wrong reads that may scale with a state's indecision, fails to end
well with chance at most `δ`. -/
def Level5RoundCorrect : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] {σ : Type*} [Fintype σ] (M : DFA α σ)
    (side : σ → Bool) (U : σ → ℝ) (θ εw κ : ℝ) {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω)
    [IsProbabilityMeasure μ] (R : KappaReadOracle μ M side U εw κ) (P : PrefixDistribution α)
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
    -- Bounds, but for chance `δ/16`, the share of probes stopping undecided at their start or end
    -- sift at a good state.
    let G := 9 * θ * (nQ + 1) + 2 * p₀ * Real.log (16 * K / δ)
    -- No length-`walkStart` prefix holds more than `p₀` of the probes.
    (∀ u, P.D.real {x | prefixOf x C.walkStart = u} ≤ p₀) → 0 ≤ θ → 0 < δ →
    -- Good states' undecided middles stay under a quarter of the searches.
    48 * (P.L + 1) * (nQ + 1) * θ ≤ 1 →
    -- Good states' undecided ends stay under the ends test's rate.
    2 * G ≤ y →
    -- The middles test's yield fits under the success test's rate.
    y ≤ ε / 48 → ε < 1 →
    36 * S / ε * Real.log (16 * S * (S + 1) / δ) + S ≤ C.probeBudget →
    0 ≤ εw → 0 ≤ κ →
    -- The round reads some string wrong with chance at most `δ/8`. A probe reads at most
    -- `(L + 1)|Q|` strings and at most `|Q|` splits re-sift `(|Q| + 1)²|Σ|` each; and until a test
    -- fires, a hypothesis's probes stop undecided at most `5 log(1/a_j)` times however long the
    -- stretch, so the `κ` term does not grow with the budget.
    εw * (C.probeBudget * (P.L + 1) * nQ + nQ * (nQ + 1) ^ 2 * nα)
      + κ * (5 * S * Real.log (8 * S * (S + 1) / δ) + 1 + nQ * (nQ + 1) ^ 2 * nα) ≤ δ / 8 →
    ∫⁻ ω, (Measure.pi fun _ : Fin C.probeBudget => P.D)
        {draws | let result := indecisiveLStarLevel3 C (R.read · ω) draws
          -- Bad: undecided more than `1.5θ` of the time. A consistent ending has at most `ε` of
          -- the probes searching, changing the hypothesis or stopping undecided at their start or
          -- end sift; a harvest's sampler yields `y`.
          ¬ EndsWell
            { target := M,
              badStates := BadAt U θ,
              probes := P.D,
              disagreeRate := ε,
              disagree := searchSet (R.read · ω) C.walkStart result.hypothesis
                ∪ endsSet (R.read · ω) C.walkStart result.hypothesis,
              walkStart := C.walkStart,
              harvestLaw := harvestLaw (R.read · ω) P.D C.walkStart result.hypothesis,
              yield := y,
              ending := result.ending }} ∂μ
      ≤ ENNReal.ofReal δ

/-- `KappaRoundCorrect` with the probes' prefix mass read at a depth `kd` rather than at the walk's
start `k`. A probe reads strings `u m`, `u` a prefix of it and `m` a midfix of the tree; those
with `k ≤ |u| < kd` are shared across the length-`kd` prefixes, and one at a good state read
undecided can only feed a harvest. A tree holds at most `|Q| + 1` midfixes and each step's tree
those of the one before, so the trees a round reaches have at most `H (|Q| + 1)` such strings,
`H = Σ_{k ≤ i < kd} |Σ|^i`. The round fails to end well other than by a harvest with chance at
most `δ`, and at all with chance at most

    δ + [k < kd] · 1.5θ (N (L + 1)|Q| + |Q|(|Q| + 1)²|Σ| + H (|Q| + 1)),

`N` the budget, the first two terms bounding what the round reads besides. -/
def KappaRoundCorrectAt : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] {σ : Type*} [Fintype σ] (M : DFA α σ)
    (side : σ → Bool) (U : σ → ℝ) (θ εw κ : ℝ) {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω)
    [IsProbabilityMeasure μ] (R : KappaReadOracle μ M side U εw κ) (P : PrefixDistribution α)
    (C : Level3Settings) (p₀ : ℝ) (kd : ℕ),
    let y := C.harvestRate
    let ε := C.disagreementRate
    let δ := C.δ
    let nQ := Fintype.card σ
    let nα := Fintype.card α
    let S : ℕ := nQ * ((nQ + 2) * nα + 1) + 2 * nα + 1
    let K : ℕ := (1 + (nQ + 1) ^ 3 * nα) ^ nQ * (nQ + 3) ^ ((nQ + 2) * nα)
    let G := 9 * θ * (nQ + 1) + 2 * p₀ * Real.log (16 * K / δ)
    let H : ℝ := ∑ i ∈ Finset.Ico C.walkStart kd, (nα : ℝ) ^ i
    (∀ u, P.D.real {x | prefixOf x kd = u} ≤ p₀) → 0 ≤ θ → 0 < δ →
    48 * (P.L + 1) * (nQ + 1) * θ ≤ 1 → 2 * G ≤ y → y ≤ ε / 48 → ε < 1 →
    36 * S / ε * Real.log (16 * S * (S + 1) / δ) + S ≤ C.probeBudget →
    0 ≤ εw → 0 ≤ κ →
    εw * (C.probeBudget * (P.L + 1) * nQ + nQ * (nQ + 1) ^ 2 * nα)
      + κ * (5 * S * Real.log (8 * S * (S + 1) / δ) + 1 + nQ * (nQ + 1) ^ 2 * nα) ≤ δ / 8 →
    ∫⁻ ω, (Measure.pi fun _ : Fin C.probeBudget => P.D)
        {draws | let result := indecisiveLStarLevel3 C (R.read · ω) draws
          ¬ EndsWell
            { target := M,
              badStates := BadAt U θ,
              probes := P.D,
              disagreeRate := ε,
              disagree := searchSet (R.read · ω) C.walkStart result.hypothesis
                ∪ endsSet (R.read · ω) C.walkStart result.hypothesis,
              walkStart := C.walkStart,
              harvestLaw := harvestLaw (R.read · ω) P.D C.walkStart result.hypothesis,
              yield := y,
              ending := result.ending }
            ∧ ∀ h, result.ending ≠ .harvest h} ∂μ
      ≤ ENNReal.ofReal δ ∧
    ∫⁻ ω, (Measure.pi fun _ : Fin C.probeBudget => P.D)
        {draws | let result := indecisiveLStarLevel3 C (R.read · ω) draws
          ¬ EndsWell
            { target := M,
              badStates := BadAt U θ,
              probes := P.D,
              disagreeRate := ε,
              disagree := searchSet (R.read · ω) C.walkStart result.hypothesis
                ∪ endsSet (R.read · ω) C.walkStart result.hypothesis,
              walkStart := C.walkStart,
              harvestLaw := harvestLaw (R.read · ω) P.D C.walkStart result.hypothesis,
              yield := y,
              ending := result.ending }} ∂μ
      ≤ ENNReal.ofReal (δ + if C.walkStart < kd then 3 / 2 * θ * (C.probeBudget * (P.L + 1) * nQ
        + nQ * (nQ + 1) ^ 2 * nα + H * (nQ + 1)) else 0)

end OrthoDFA
