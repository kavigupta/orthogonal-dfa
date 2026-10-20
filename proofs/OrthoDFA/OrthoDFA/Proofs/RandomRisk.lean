import OrthoDFA.Level3Claim

/-!
# The round's risk at the proof's free parameters

`roundBoundAt` is the round's failure chance at the proof's free parameters (`Knobs`): `tailRisk`
at each state's rate and at each site, the chance that for some tree level 3 can grow and some
edge map the probes harvesting at the states read undecided at most that rate exceed their mean
by `η`, plus, for each of the at most `S` hypotheses, `stretchRisk` at its tests' level, what one
stretch risks over its first `Ns` probes. `Level3RoundCorrect` picks knobs meeting
`Knobs.Admissible`.
-/

namespace OrthoDFA

namespace Random

open MeasureTheory

variable {α : Type*}

/-- `E[f(Bin(n, p))]`. -/
noncomputable def binE (n : ℕ) (p : ℝ) (f : ℕ → ℝ) : ℝ :=
  ∑ j ∈ Finset.range (n + 1), (n.choose j : ℝ) * p ^ j * (1 - p) ^ (n - j) * f j

/-- The chance that, for some tree of at most `nQ + 2` leaves grown by splits on a letter and a
midfix where two leaves part, and some edge map into its leaves, a sum over the probes exceeds
its bound by `η`, each pair's tail at scale `p₀`. -/
noncomputable def tailRisk (nQ nα : ℕ) (p₀ η : ℝ) : ℝ :=
  ((1 + (nQ + 1) ^ 3 * nα) ^ nQ * (nQ + 3) ^ ((nQ + 2) * nα) : ℕ) * Real.exp (-(η / p₀))

/-- What one stretch at tests' level `a` risks over its first `Ns` probes: a test firing at some
look at a site whose harvest would not end well, the middles' twice, or a false success, each at
most `a`; and the stretch outliving `Ns` probes, though the probes changing the hypothesis are at
least `θr` of them, or the searches and the probes stopping undecided at their ends are at most
`εd'` of the probes yet success is silent at look `N₁`, or the latter are at least `θe'` yet their
test is silent, or the searches are more than `εd' − θe'` yet fewer than `searchShare` of the
`Ns` probes, or the undecided middles are at least `θpt'` of them yet their test is silent. -/
noncomputable def stretchRisk (C : Level3Settings) (a : ℝ) (Ns N₁ : ℕ) (θr εd' θe' θpt' : ℝ) :
    ℝ :=
  4 * a + (1 - θr) ^ Ns
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
  /-- Half the least yield of an ends harvest that ends well. -/
  Ge : ℝ
  /-- Half the least share of the searches a middles harvest that ends well yields, past `ηm`. -/
  θgpt : ℝ
  ηm : ℝ
  /-- The rate of undecided middles per probe below which the middles' test is checked against
  it rather than against the searches. -/
  pM : ℝ
  /-- The rate of probes changing the hypothesis the stretch's risk tests against. -/
  θr : ℝ
  /-- The disagreement rate the stretch's risk tests against. -/
  εd' : ℝ
  /-- The ends' rate the stretch's risk tests against. -/
  θe' : ℝ
  /-- The middles rate the stretch's risk tests against. -/
  θpt' : ℝ

/-- The relations the knobs must meet, at the settings `C`, the ending's tolerance `ε` and the
yield `y`: the ends' rate is at least `max (2Ge) y`; `θgpt` is at most a quarter of `middleRate`,
and `4ηm ≤ pM`; and against a rate `pM ≥ y` of undecided middles per probe, with searches at least
`searchShare` of the probes, the middles' test's likelihood ratio falls on average, its factor
`m/μ` at an undecided middle and `(1 − m)/(1 − μ)` at another search, `μ = middleRate` and
`m = middleAlternative`. -/
structure Knobs.Admissible {σ : Type*} [Fintype σ] [Fintype α] (C : Level3Settings)
    (ε y : ℝ) (kn : Knobs) : Prop where
  N₁_pos : 1 ≤ kn.N₁
  N₁_le : kn.N₁ ≤ kn.Ns
  Ge_nonneg : 0 ≤ kn.Ge
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
  harvest_le_ends : max (2 * kn.Ge) y ≤ C.endsRate
  θgpt_le : 4 * kn.θgpt ≤ C.middleRate
  pM_nonneg : 0 ≤ kn.pM
  y_le_pM : y ≤ kn.pM
  ηm_le : 4 * kn.ηm ≤ kn.pM
  middles_fall : kn.pM * (C.middleAlternative / C.middleRate)
      * ((1 - C.middleAlternative) / (1 - C.middleRate)) ^ (C.searchShare - 1)
    + (1 - kn.pM) * ((1 - C.middleAlternative) / (1 - C.middleRate)) ^ C.searchShare
    ≤ 1

/-- The round's failure chance at the knobs `kn`, with no length-`walkStart` prefix above `p₀`:
`tailRisk` at each state's rate and at each site, past `η`, and `stretchRisk` at each of the at
most `S` hypotheses' level. -/
noncomputable def roundBoundAt (σ : Type*) [Fintype σ] [Fintype α] (η p₀ : ℝ)
    (C : Level3Settings) (kn : Knobs) : ℝ :=
  2 * (Fintype.card σ + 1) * tailRisk (Fintype.card σ) (Fintype.card α) p₀ η
    + ∑ j ∈ Finset.range (Fintype.card σ * ((Fintype.card σ + 2) * Fintype.card α + 1)
        + 2 * Fintype.card α + 1),
      stretchRisk C (C.level j) kn.Ns kn.N₁ kn.θr kn.εd' kn.θe' kn.θpt'

end Random

end OrthoDFA
