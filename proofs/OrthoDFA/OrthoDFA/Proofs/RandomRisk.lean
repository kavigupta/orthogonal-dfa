import OrthoDFA.Level3Claim

/-!
# The round's risk at the proof's free parameters

`roundBound` is the round's failure chance at the proof's free parameters (`Knobs`): `noiseRisk`,
the chance that the probes that can read a good read-state undecided reach `G` of the probes for
some tree level 3 can grow, plus, for each of the at most `S` hypotheses, `stretchRisk` at its
tests' level, what one stretch risks over its first `Ns` probes. `RoundRiskBound` bounds the round
by the least of it over the knobs meeting `Knobs.Admissible`; `RandomRoundCorrect` picks knobs.
-/

namespace OrthoDFA

namespace Random

open MeasureTheory

variable {α : Type*}

/-- `E[f(Bin(n, p))]`. -/
noncomputable def binE (n : ℕ) (p : ℝ) (f : ℕ → ℝ) : ℝ :=
  ∑ j ∈ Finset.range (n + 1), (n.choose j : ℝ) * p ^ j * (1 - p) ^ (n - j) * f j

/-- The chance that the probes that can read a string with a read of chance at most `β` reach
`G` of the probes, for some tree of at most `nQ + 2` leaves grown by splits on a letter and a
midfix where two leaves part, with probes of length at most `L` and no length-`k` prefix above
`p₀`. -/
noncomputable def noiseRisk (nQ nα L : ℕ) (β p₀ G : ℝ) : ℝ :=
  ((1 + (nQ + 1) ^ 3 * nα) ^ nQ : ℕ) * Real.exp (-(G - 2 * β * (L + 1) * (nQ + 1)) / p₀)

/-- What one stretch at tests' level `a` risks over its first `Ns` probes: a test firing at some
look at a site whose harvest would not end well, or a false success, each at most `a`; and the
stretch outliving `Ns` probes, though the probes changing the hypothesis are at least `θr` of
them, or the searches and the probes stopping undecided at their ends are at most `εd'` of the
probes yet success is silent at look `N₁`, or the latter are at least `θe'` yet their test is
silent, or the searches are more than `εd' − θe'` yet fewer than `searchShare` of the `Ns`
probes, or the undecided middles are at least `θpt'` of them yet their test is silent. -/
noncomputable def stretchRisk (C : Level3Settings) (a : ℝ) (Ns N₁ : ℕ) (θr εd' θe' θpt' : ℝ) :
    ℝ :=
  3 * a + (1 - θr) ^ Ns
    + (εd' * (C.disagreementRate / C.disagreementAlternative)
        + (1 - εd') * ((1 - C.disagreementRate) / (1 - C.disagreementAlternative))) ^ N₁ / a
    + (θe' * (C.endsRate / C.endsAlternative)
        + (1 - θe') * ((1 - C.endsRate) / (1 - C.endsAlternative))) ^ Ns / a
    + 2 ^ (⌈C.searchShare * Ns⌉₊ - 1) * (1 - (εd' - θe') / 2) ^ Ns
    + (θpt' * (C.middleRate / C.middleAlternative)
        + (1 - θpt') * ((1 - C.middleRate) / (1 - C.middleAlternative))) ^ ⌈C.searchShare * Ns⌉₊
      / a

/-- The largest mass the probes' distribution puts on one length-`k` prefix. -/
noncomputable def prefixMax [Fintype α] (D : Measure (FreeMonoid α)) (k : ℕ) : ℝ :=
  ⨆ u, D.real {x | prefixOf x k = u}

/-- The proof's free parameters. -/
structure Knobs where
  /-- The probes a stretch gets before its risk is counted. -/
  Ns : ℕ
  /-- The look at which the success test must have fired. -/
  N₁ : ℕ
  /-- The good read-states' undecided reads, as a share of the probes. -/
  G : ℝ
  /-- The rate of probes changing the hypothesis the stretch's risk tests against. -/
  θr : ℝ
  /-- The disagreement rate the stretch's risk tests against. -/
  εd' : ℝ
  /-- The ends' rate the stretch's risk tests against. -/
  θe' : ℝ
  /-- The middles rate the stretch's risk tests against. -/
  θpt' : ℝ

/-- The relations the knobs must meet, at the settings `C`, the ending's tolerance `ε` and the
yield `y`: the ends' rate is at least what its sampler can find, `max (2G) y`, `2G` bounding what
it can find at good read-states; and against that rate of undecided middles per probe, with
searches at least `searchShare` of the probes, the middles' test's likelihood ratio falls on
average, its factor `m/μ` at an undecided middle and `(1 − m)/(1 − μ)` at another search,
`μ = middleRate` and `m = middleAlternative`. -/
structure Knobs.Admissible {σ : Type*} [Fintype σ] [Fintype α] (θ : ℝ) (C : Level3Settings)
    (ε y : ℝ) (kn : Knobs) : Prop where
  N₁_pos : 1 ≤ kn.N₁
  N₁_le : kn.N₁ ≤ kn.Ns
  θ_nonneg : 0 ≤ θ
  G_nonneg : 0 ≤ kn.G
  δ_pos : 0 < C.δ
  endsRate_pos : 0 < C.endsRate
  endsRate_lt : C.endsRate < 1 / 2
  middleRate_pos : 0 < C.middleRate
  middleRate_lt : C.middleRate < 1 / 2
  disagreementRate_pos : 0 < C.disagreementRate
  disagreementRate_lt : C.disagreementRate < 1
  disagreementRate_le_ε : C.disagreementRate ≤ ε
  θr_nonneg : 0 ≤ kn.θr
  θr_le : kn.θr ≤ 1
  εd'_nonneg : 0 ≤ kn.εd'
  εd'_le : kn.εd' ≤ 1
  θe'_nonneg : 0 ≤ kn.θe'
  θe'_le : kn.θe' ≤ kn.εd'
  θpt'_ge : C.middleAlternative ≤ kn.θpt'
  θpt'_le : kn.θpt' ≤ 1
  sep : kn.θr ≤ (1 - kn.θpt') * (kn.εd' - kn.θe')
  probeBudget_ge : (Fintype.card σ * ((Fintype.card σ + 2) * Fintype.card α + 1)
    + 2 * Fintype.card α + 1) * kn.Ns ≤ C.probeBudget
  harvest_le_ends : max (2 * kn.G) y ≤ C.endsRate
  middles_fall : max (2 * kn.G) y * (C.middleAlternative / C.middleRate)
      * ((1 - C.middleAlternative) / (1 - C.middleRate)) ^ (C.searchShare - 1)
    + (1 - max (2 * kn.G) y) * ((1 - C.middleAlternative) / (1 - C.middleRate)) ^ C.searchShare
    ≤ 1

/-- The round's failure chance at the knobs `kn`, with no length-`walkStart` prefix above `p₀`:
`noiseRisk`, and `stretchRisk` at each of the at most `S` hypotheses' level. -/
noncomputable def roundBoundAt (σ : Type*) [Fintype σ] [Fintype α] (θ p₀ : ℝ) (L : ℕ)
    (C : Level3Settings) (kn : Knobs) : ℝ :=
  noiseRisk (Fintype.card σ) (Fintype.card α) L (3 / 2 * θ) p₀ kn.G
    + ∑ j ∈ Finset.range (Fintype.card σ * ((Fintype.card σ + 2) * Fintype.card α + 1)
        + 2 * Fintype.card α + 1),
      stretchRisk C (C.level j) kn.Ns kn.N₁ kn.θr kn.εd' kn.θe' kn.θpt'

/-- `roundBoundAt` at the probes' largest prefix mass. -/
noncomputable def roundBound (σ : Type*) [Fintype σ] [Fintype α] (θ : ℝ)
    (D : Measure (FreeMonoid α)) (L : ℕ) (C : Level3Settings) (_y : ℝ) (kn : Knobs) : ℝ :=
  roundBoundAt (α := α) σ θ (prefixMax D C.walkStart) L C kn

/-- The least of `roundBound` over the admissible knobs. -/
noncomputable def roundRisk (σ : Type*) [Fintype σ] [Fintype α] (θ : ℝ)
    (D : Measure (FreeMonoid α)) (L : ℕ) (C : Level3Settings) (ε y : ℝ) : ENNReal :=
  ⨅ (kn : Knobs) (_ : kn.Admissible (σ := σ) (α := α) θ C ε y),
    ENNReal.ofReal (roundBound (α := α) σ θ D L C y kn)

/-- Over `C.probeBudget` probes drawn from `P.D`, level 3 fails to end well with chance at most
`roundRisk`. -/
def RoundRiskBound : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] {σ : Type*} [Fintype σ] (M : DFA α σ)
    (side : σ → Bool) (U : σ → ℝ) (θ : ℝ) {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω)
    [IsProbabilityMeasure μ] (R : ReadOracle μ M side U) (P : PrefixDistribution α)
    (C : Level3Settings) (ε y : ℝ),
    ∫⁻ ω, (Measure.pi fun _ : Fin C.probeBudget => P.D)
        {draws | let result := indecisiveLStarLevel3 C (R.read · ω) draws
          ¬ EndsWell M (BadAt U θ) P.D ε
            (searchSet (R.read · ω) C.walkStart result.hypothesis
              ∪ endsSet (R.read · ω) C.walkStart result.hypothesis)
            C.walkStart (harvestLaw (R.read · ω) P.D C.walkStart result.hypothesis) y
            result.ending} ∂μ
      ≤ roundRisk (α := α) σ θ P.D P.L C ε y

end Random

end OrthoDFA
