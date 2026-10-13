import OrthoDFA.Level3Claim
import OrthoDFA.Level4

/-!
# `EpsRoundCorrect`: level 4 ends well but for a small chance

Level 4 (`indecisiveLStarLevel4`) runs with every read on its state's wrong side with chance
at most `εw`. A wrong read can make a record untrue, and `recordsToMove` untrue records since the
last split at one edge and a target that no state at the edge's leaf reaches under its letter
redirect the edge there, or split its leaf where no two of its states part.

`EpsRoundCorrect`: the chance of not ending well is at most `epsRoundRisk`, the least over the
admissible knobs of `epsRoundBound`: `roundBound`, plus `noiseRisk` for the probes that can read a
string wrong reaching `Gw` of the probes, and `fakeRisk`, a binomial tail over the round's probes
for each split.
-/

namespace OrthoDFA

namespace Random

namespace Eps

open MeasureTheory

variable {α : Type*}

/-- Each split, and each edge's learning and redirect between splits, changes the hypothesis
once; the stretches are one more than those changes. -/
def stretches (C : Level4Settings) (nα : ℕ) : ℕ :=
  C.maxLeaves * (2 * C.maxLeaves * nα + 1) + 1

/-- What one stretch risks over its first `Ns` probes: a test firing at one of its looks at a site
whose harvest would not end well (at the start or the middles, whose sampler then finds at most
`p`, a binomial tail; at one of the `maxLeaves |Σ|` edges, the count passing twice its mean by
`edgeExcess`), or a false success; and
the stretch outliving `Ns` probes, though a record recurs at least `θr` of the time, or the
searches are at most `εd'` of the probes yet success has not fired at look `N₁`, or they are more
and the undecided middles at least `θpt'` of them yet their test has not fired. -/
noncomputable def stretchRisk (C : Level4Settings) (nα L Ns N₁ : ℕ) (p θr εd' θpt' : ℝ) : ℝ :=
  ∑ n ∈ Finset.Icc C.firstLook Ns,
      (binomSfGe n p (hFire C.startRate C.testLevel n) + C.testLevel)
    + ∑ n ∈ Finset.Icc 1 Ns,
      ((C.maxLeaves * nα : ℕ) * Real.exp (-C.edgeExcess / (2 * (L + 1)))
        + binomSfGe n p (hFire C.middleRate C.testLevel (max C.firstLook ⌈C.searchShare * n⌉₊)))
    + (1 - binomSfGe Ns θr C.recordsToMove)
    + binE N₁ εd' (fun s =>
        if C.testLevel ≤ 1 - binomSfGe N₁ C.disagreementRate (s + 1)
          ∨ binomSfGe N₁ C.disagreementRate s < C.testLevel then 1 else 0)
    + (1 - binomSfGe Ns εd' (max C.firstLook ⌈C.searchShare * Ns⌉₊))
    + ∑ s ∈ Finset.Icc (max C.firstLook ⌈C.searchShare * Ns⌉₊) Ns,
        binE s θpt' (fun j => if C.testLevel ≤ binomSfGe s C.middleRate j then 1 else 0)

/-- The relations the knobs must meet, at the settings `C`, the probes' length `L`, the ending's
tolerance `ε` and the yield `y`: an edge's least traffic times its rate is at least twice the
yield, and twice what an edge's sampler can find at good read-states, `2 (L + 1) G`. -/
structure Admissible {σ : Type*} [Fintype σ] [Fintype α] (θ : ℝ) (C : Level4Settings)
    (L : ℕ) (ε y : ℝ) (kn : Knobs) : Prop where
  maxLeaves_ge : Fintype.card σ + 2 ≤ C.maxLeaves
  recordsToMove_pos : 1 ≤ C.recordsToMove
  firstLook_pos : 1 ≤ C.firstLook
  firstLook_le : C.firstLook ≤ kn.N₁
  N₁_le : kn.N₁ ≤ kn.Ns
  θ_nonneg : 0 ≤ θ
  G_nonneg : 0 ≤ kn.G
  testLevel_nonneg : 0 ≤ C.testLevel
  startRate_nonneg : 0 ≤ C.startRate
  startRate_le : C.startRate ≤ 1
  edgeRate_nonneg : 0 ≤ C.edgeRate
  middleRate_nonneg : 0 ≤ C.middleRate
  middleRate_le : C.middleRate ≤ 1
  disagreementRate_nonneg : 0 ≤ C.disagreementRate
  disagreementRate_le : C.disagreementRate ≤ 1
  disagreementRate_le_ε : C.disagreementRate ≤ ε
  θr_nonneg : 0 ≤ kn.θr
  θr_le : kn.θr ≤ 1
  εd'_nonneg : 0 ≤ kn.εd'
  εd'_le : kn.εd' ≤ 1
  θpt'_nonneg : 0 ≤ kn.θpt'
  θpt'_le : kn.θpt' ≤ 1
  sep : (C.maxLeaves : ℝ) ^ 2 * Fintype.card α * kn.θr ≤ (1 - kn.θpt') * kn.εd'
  probeBudget_ge : stretches C (Fintype.card α) * kn.Ns ≤ C.probeBudget
  G_le : 2 * kn.G ≤ 1
  yield_le_one : y ≤ 1
  noise_le_traffic : 4 * (L + 1) * kn.G ≤ C.edgeRate * C.edgeReadShare
  yield_le_traffic : 2 * y ≤ C.edgeRate * C.edgeReadShare

/-- The round's failure chance at the knobs `kn`: `noiseRisk`, the probes that can read a good
read-state undecided reaching `G` of the probes for some tree, and `stretches` times
`stretchRisk`, what one stretch risks over its first `Ns` probes. -/
noncomputable def roundBound (σ : Type*) [Fintype σ] [Fintype α] (θ : ℝ)
    (D : Measure (FreeMonoid α)) (L : ℕ) (C : Level4Settings) (y : ℝ) (kn : Knobs) : ℝ :=
  noiseRisk (Fintype.card σ) (Fintype.card α) L (3 / 2 * θ) (prefixMax D C.walkStart) kn.G
    + stretches C (Fintype.card α)
      * stretchRisk C (Fintype.card α) L kn.Ns kn.N₁ (max (2 * kn.G) y) kn.θr kn.εd' kn.θpt'


/-- What the fake fixes risk where reads can be wrong: for each split, `recordsToMove` records
since the last split at a target that is not real, each from a probe that can read a string
wrong, at most `Gw` of the probes, over at most `probeBudget` probes. -/
noncomputable def fakeRisk (C : Level4Settings) (Gw : ℝ) : ℝ :=
  C.maxLeaves * binomSfGe C.probeBudget Gw C.recordsToMove

/-- The proof's free parameters where reads can be wrong: `Knobs`, and the probes that can read a
string wrong, as a share of the probes. -/
structure EpsKnobs extends Knobs where
  Gw : ℝ

/-- `Admissible`, and `Gw` a share. -/
structure EpsKnobs.Admissible {σ : Type*} [Fintype σ] [Fintype α] (θ : ℝ) (C : Level4Settings)
    (L : ℕ) (ε y : ℝ) (kn : EpsKnobs) : Prop where
  base : Eps.Admissible (σ := σ) (α := α) θ C L ε y kn.toKnobs
  Gw_nonneg : 0 ≤ kn.Gw
  Gw_le : kn.Gw ≤ 1

/-- `roundBound`, plus `noiseRisk` for the probes that can read a string wrong reaching `Gw` of the
probes, and `fakeRisk`. -/
noncomputable def epsRoundBound (σ : Type*) [Fintype σ] [Fintype α] (θ εw : ℝ)
    (D : Measure (FreeMonoid α)) (L : ℕ) (C : Level4Settings) (y : ℝ) (kn : EpsKnobs) : ℝ :=
  roundBound (α := α) σ θ D L C y kn.toKnobs
    + noiseRisk (Fintype.card σ) (Fintype.card α) L εw (prefixMax D C.walkStart) kn.Gw
    + fakeRisk C kn.Gw

/-- The least of `epsRoundBound` over the admissible knobs. -/
noncomputable def epsRoundRisk (σ : Type*) [Fintype σ] [Fintype α] (θ εw : ℝ)
    (D : Measure (FreeMonoid α)) (L : ℕ) (C : Level4Settings) (ε y : ℝ) : ENNReal :=
  ⨅ (kn : EpsKnobs) (_ : kn.Admissible (σ := σ) (α := α) θ C L ε y),
    ENNReal.ofReal (epsRoundBound (α := α) σ θ εw D L C y kn)

/-- `ReadOracle` with each read on its state's wrong side with chance at most `εw`. -/
structure EpsReadOracle {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) {σ : Type*}
    (M : DFA α σ) (side : σ → Bool) (U : σ → ℝ) (εw : ℝ) where
  read : FreeMonoid α → Ω → ARU
  measurable : ∀ z, Measurable (read z)
  indep : ProbabilityTheory.iIndepFun read μ
  undecided : ∀ z, μ.real {ω | read z ω = .undecided} = U (M.eval z.toList)
  wrong : ∀ z, μ.real {ω | read z ω = if side (M.eval z.toList) then .reject else .accept} ≤ εw

/-- `RandomRoundCorrect`, its yield included, for level 4: each read wrong with chance at most
`εw`. -/
def EpsRoundCorrect : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] {σ : Type*} [Fintype σ] (M : DFA α σ)
    (side : σ → Bool) (U : σ → ℝ) (θ εw : ℝ) {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω)
    [IsProbabilityMeasure μ] (R : EpsReadOracle μ M side U εw) (P : PrefixDistribution α)
    (C : Level4Settings) (ε y : ℝ),
    ∫⁻ ω, (Measure.pi fun _ : Fin C.probeBudget => P.D)
        {draws | let result := indecisiveLStarLevel4 C (R.read · ω) draws
          ¬ EndsWell M (BadAt U θ) P.D ε
            {x | Disagrees (R.read · ω) result.hypothesis C.walkStart x}
            C.walkStart (harvestLaw (R.read · ω) P.D C.walkStart result.hypothesis) y
            result.ending} ∂μ
      ≤ epsRoundRisk (α := α) σ θ εw P.D P.L C ε y

end Eps

end Random

end OrthoDFA
