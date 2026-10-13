import OrthoDFA.Level3

/-!
# `RandomRoundCorrect`: level 3 ends well but for a small chance

Every string's read is drawn once, independently across strings, from a distribution fixed by
the string's target state: undecided with chance `U q`, otherwise on the state's side, never on
the other (`ReadOracle`). A read-state is bad where it is undecided more than `1.5θ` of the time.

A harvest is a site of the final hypothesis. Its sampler (`harvestLaw`) draws a fresh probe, walks
it under that hypothesis and finds the probe's undecided strings at the site, as the tests counted
them; the reads are the round's own.

Over `C.probeBudget` probes, level 3 fails to end well (`OrthoDFA.RoundEnd`), a harvest's sampler
yielding `y`, with chance at most `roundRisk`: the least, over the proof's free parameters
(`Knobs`) meeting `Knobs.Admissible`, of `roundBound`, which is `noiseRisk`, the chance that the
probes that can read a good read-state undecided reach `G` of the probes for some tree level 3 can
grow, plus `stretches` times `stretchRisk`, what one stretch risks over its first `Ns` probes.
-/

namespace OrthoDFA

namespace Random

open MeasureTheory
open OrthoDFA.Ideal (probeOrAgree)

variable {α : Type*}

/-- The strings a probe of `x` walked from `k` under `H` reads undecided at the site, as the tests
count them. -/
noncomputable def siteHarvest [Fintype α] [DecidableEq α] (read : FreeMonoid α → ARU) (k : ℕ)
    (H : HypothesisDFA α) : Level3Site α → FreeMonoid α → List (FreeMonoid α)
  | .start, x => startString (probeOrAgree false read H.tree H.edges k x)
  | .edge p c, x => undecidedAtEdge (positionReads read H.tree H.edges k x) p c
  | .middles, x => middleString (probeOrAgree false read H.tree H.edges k x)

/-- The law of what the harvest's sampler finds, each string as often as a probe drawn from `D`
and walked under `H` finds it at the site. -/
noncomputable def harvestLaw [Fintype α] [DecidableEq α] (read : FreeMonoid α → ARU)
    (D : Measure (FreeMonoid α)) (k : ℕ) (H : HypothesisDFA α) (site : Level3Site α) :
    Measure (FreeMonoid α) :=
  D.bind fun x => ((siteHarvest read k H site x).map Measure.dirac).sum

/-- A read-state is bad where it reads undecided more than `1.5θ` of the time. -/
def BadAt {σ : Type*} (U : σ → ℝ) (θ : ℝ) (q : σ) : Prop := 3 / 2 * θ < U q

/-- `E[f(Bin(n, p))]`. -/
noncomputable def binE (n : ℕ) (p : ℝ) (f : ℕ → ℝ) : ℝ :=
  ∑ j ∈ Finset.range (n + 1), (n.choose j : ℝ) * p ^ j * (1 - p) ^ (n - j) * f j

/-- The least count at which `rateSide θ` calls a rate above `θ` after `n` trials. -/
noncomputable def hFire (θ a : ℝ) (n : ℕ) : ℕ := sInf {h | binomSfGe n θ h < a}

/-- The chance that the probes that can read a string with a read of chance at most `β` reach
`G` of the probes, for some tree of at most `nQ + 2` leaves grown by splits on a letter and a
midfix where two leaves part, with probes of length at most `L` and no length-`k` prefix above
`p₀`. -/
noncomputable def noiseRisk (nQ nα L : ℕ) (β p₀ G : ℝ) : ℝ :=
  ((1 + (nQ + 1) ^ 3 * nα) ^ nQ : ℕ) * Real.exp (-(G - 2 * β * (L + 1) * (nQ + 1)) / p₀)

/-- What one stretch risks over its first `Ns` probes: a test firing at one of its looks at a site
whose harvest would not end well (at the start or the middles, whose sampler then finds at most
`p`, a binomial tail; at one of the `(|Q| + 2)|Σ|` edges, the count passing twice its mean by
`edgeExcess`), or a false success; and
the stretch outliving `Ns` probes, though clean disagreements are at least `θr` of them, or the
searches are at most `εd'` of the probes yet success has not fired at look `N₁`, or they are more
and the undecided middles at least `θpt'` of them yet their test has not fired. -/
noncomputable def stretchRisk (C : Level3Settings) (nσ nα L Ns N₁ : ℕ) (p θr εd' θpt' : ℝ) : ℝ :=
  ∑ n ∈ Finset.Icc C.firstLook Ns,
      (binomSfGe n p (hFire C.startRate C.testLevel n) + C.testLevel)
    + ∑ n ∈ Finset.Icc 1 Ns,
      (((nσ + 2) * nα : ℕ) * Real.exp (-C.edgeExcess / (2 * (L + 1)))
        + binomSfGe n p (hFire C.middleRate C.testLevel (max C.firstLook ⌈C.searchShare * n⌉₊)))
    + (1 - binomSfGe Ns θr 1)
    + binE N₁ εd' (fun s =>
        if C.testLevel ≤ 1 - binomSfGe N₁ C.disagreementRate (s + 1)
          ∨ binomSfGe N₁ C.disagreementRate s < C.testLevel then 1 else 0)
    + (1 - binomSfGe Ns εd' (max C.firstLook ⌈C.searchShare * Ns⌉₊))
    + ∑ s ∈ Finset.Icc (max C.firstLook ⌈C.searchShare * Ns⌉₊) Ns,
        binE s θpt' (fun j => if C.testLevel ≤ binomSfGe s C.middleRate j then 1 else 0)

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
  /-- The clean disagreements' rate the stretch's risk tests against. -/
  θr : ℝ
  /-- The disagreement rate the stretch's risk tests against. -/
  εd' : ℝ
  /-- The middles rate the stretch's risk tests against. -/
  θpt' : ℝ

/-- The relations the knobs must meet, at the settings `C`, the probes' length `L`, the ending's
tolerance `ε` and the yield `y`: an edge's least traffic times its rate is at least twice the
yield, and twice what an edge's sampler can find at good read-states, `2 (L + 1) G`. -/
structure Knobs.Admissible {σ : Type*} [Fintype σ] [Fintype α] (θ : ℝ) (C : Level3Settings)
    (L : ℕ) (ε y : ℝ) (kn : Knobs) : Prop where
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
  sep : kn.θr ≤ (1 - kn.θpt') * kn.εd'
  probeBudget_ge : Ideal.stretches (Fintype.card σ) (Fintype.card α) * kn.Ns ≤ C.probeBudget
  G_le : 2 * kn.G ≤ 1
  yield_le_one : y ≤ 1
  noise_le_traffic : 4 * (L + 1) * kn.G ≤ C.edgeRate * C.edgeReadShare
  yield_le_traffic : 2 * y ≤ C.edgeRate * C.edgeReadShare

/-- The round's failure chance at the knobs `kn`: `noiseRisk`, the probes that can read a good
read-state undecided reaching `G` of the probes for some tree, and `stretches` times
`stretchRisk`, what one stretch risks over its first `Ns` probes. -/
noncomputable def roundBound (σ : Type*) [Fintype σ] [Fintype α] (θ : ℝ)
    (D : Measure (FreeMonoid α)) (L : ℕ) (C : Level3Settings) (y : ℝ) (kn : Knobs) : ℝ :=
  noiseRisk (Fintype.card σ) (Fintype.card α) L (3 / 2 * θ) (prefixMax D C.walkStart) kn.G
    + Ideal.stretches (Fintype.card σ) (Fintype.card α)
      * stretchRisk C (Fintype.card σ) (Fintype.card α) L kn.Ns kn.N₁ (max (2 * kn.G) y) kn.θr
        kn.εd' kn.θpt'

/-- The least of `roundBound` over the admissible knobs. -/
noncomputable def roundRisk (σ : Type*) [Fintype σ] [Fintype α] (θ : ℝ)
    (D : Measure (FreeMonoid α)) (L : ℕ) (C : Level3Settings) (ε y : ℝ) : ENNReal :=
  ⨅ (kn : Knobs) (_ : kn.Admissible (σ := σ) (α := α) θ C L ε y),
    ENNReal.ofReal (roundBound (α := α) σ θ D L C y kn)

/-- Reads of every string, independent across strings, each undecided with its state's chance `U`
and never on its state's wrong side. -/
structure ReadOracle {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) {σ : Type*} (M : DFA α σ)
    (side : σ → Bool) (U : σ → ℝ) where
  read : FreeMonoid α → Ω → ARU
  measurable : ∀ z, Measurable (read z)
  indep : ProbabilityTheory.iIndepFun read μ
  undecided : ∀ z, μ.real {ω | read z ω = .undecided} = U (M.eval z.toList)
  wrong : ∀ z, μ {ω | read z ω = if side (M.eval z.toList) then .reject else .accept} = 0

/-- Over `C.probeBudget` probes drawn from `P.D`, level 3 fails to end well, a state being bad
where it is undecided more than `1.5θ` of the time and a harvest's sampler yielding `y`, with
chance at most `roundRisk`. -/
def RandomRoundCorrect : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] {σ : Type*} [Fintype σ] (M : DFA α σ)
    (side : σ → Bool) (U : σ → ℝ) (θ : ℝ) {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω)
    [IsProbabilityMeasure μ] (R : ReadOracle μ M side U) (P : PrefixDistribution α)
    (C : Level3Settings) (ε y : ℝ),
    ∫⁻ ω, (Measure.pi fun _ : Fin C.probeBudget => P.D)
        {draws | let result := indecisiveLStarLevel3 C (R.read · ω) draws
          ¬ EndsWell M (BadAt U θ) P.D ε
            {x | Disagrees (R.read · ω) result.hypothesis C.walkStart x}
            C.walkStart (harvestLaw (R.read · ω) P.D C.walkStart result.hypothesis) y
            result.ending} ∂μ
      ≤ roundRisk (α := α) σ θ P.D P.L C ε y

end Random

end OrthoDFA
