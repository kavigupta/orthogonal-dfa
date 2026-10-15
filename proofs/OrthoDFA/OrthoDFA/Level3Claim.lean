import OrthoDFA.Level3
import OrthoDFA.ProbeDistribution

/-!
# `RandomRoundCorrect`: level 3 ends well but for a small chance

Every string's read is drawn once, independently across strings, from a distribution fixed by
the string's target state: undecided with chance `U q`, otherwise on the state's side, never on
the other (`ReadOracle`). A read-state is bad where it is undecided more than `1.5θ` of the time.

A harvest is a site of the final hypothesis. Its sampler (`harvestLaw`) draws a fresh probe, walks
it under that hypothesis and finds the probe's undecided strings at the site, as the tests counted
them; the reads are the round's own.

Over `C.probeBudget` probes, level 3 fails to end well (`OrthoDFA.RoundEnd`), a harvest's sampler
yielding `y`, with chance at most `δ`. `S = |Q|((|Q| + 2)|Σ| + 1) + 2|Σ| + 1` bounds the
hypotheses: each split adds a leaf, at most `|Q| + 2` of them, and between splits each of a leaf's
edges is learned once.
-/

namespace OrthoDFA

namespace Random

open MeasureTheory

variable {α : Type*}

/-- The strings a probe of `x` walked from `k` under `H` reads undecided at the site, as the tests
count them. -/
noncomputable def siteHarvest [Fintype α] [DecidableEq α] (read : FreeMonoid α → ARU) (k : ℕ)
    (H : HypothesisDFA α) : Level3Site α → FreeMonoid α → List (FreeMonoid α)
  | .start, x => startString (probeOrAgree read H.tree H.edges k x)
  | .edge p c, x => undecidedAtEdge (positionReads read H.tree H.edges k x) p c
  | .middles, x => middleString (probeOrAgree read H.tree H.edges k x)

/-- The law of what the harvest's sampler finds, each string as often as a probe drawn from `D`
and walked under `H` finds it at the site. -/
noncomputable def harvestLaw [Fintype α] [DecidableEq α] (read : FreeMonoid α → ARU)
    (D : Measure (FreeMonoid α)) (k : ℕ) (H : HypothesisDFA α) (site : Level3Site α) :
    Measure (FreeMonoid α) :=
  D.bind fun x => ((siteHarvest read k H site x).map Measure.dirac).sum

/-- A read-state is bad where it reads undecided more than `1.5θ` of the time. -/
def BadAt {σ : Type*} (U : σ → ℝ) (θ : ℝ) (q : σ) : Prop := 3 / 2 * θ < U q

/-- Reads of every string, independent across strings, each undecided with its state's chance `U`
and never on its state's wrong side. -/
structure ReadOracle {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) {σ : Type*} (M : DFA α σ)
    (side : σ → Bool) (U : σ → ℝ) where
  read : FreeMonoid α → Ω → ARU
  measurable : ∀ z, Measurable (read z)
  indep : ProbabilityTheory.iIndepFun read μ
  undecided : ∀ z, μ.real {ω | read z ω = .undecided} = U (M.eval z.toList)
  wrong : ∀ z, μ {ω | read z ω = if side (M.eval z.toList) then .reject else .accept} = 0

/-- With every string's read drawn once as `ReadOracle` says, every length-`walkStart` prefix
of the probes at most `p₀` of them, and `C.probeBudget` at least
`8 S / searchShare · log(16 S (S + 1) / δ) + S`, `S` bounding the hypotheses, level 3 fails to
end well (`OrthoDFA.RoundEnd`), a state being bad where it is undecided more than `1.5θ` of the
time, with chance at most `δ`. A consistent ending has its hypothesis and tree parting on at most
`ε = disagreementRate` of the probes, and a harvest's sampler yields `y`. Each rate the tests count
must clear `p = max(2G, y)`, where `G` bounds what the probes can find undecided at good states but
for chance `δ / 4`. -/
def RandomRoundCorrect : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] {σ : Type*} [Fintype σ] (M : DFA α σ)
    (side : σ → Bool) (U : σ → ℝ) (θ : ℝ) {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω)
    [IsProbabilityMeasure μ] (R : ReadOracle μ M side U) (P : PrefixDistribution α)
    (C : Level3Settings) (y p₀ : ℝ),
    let ε := C.disagreementRate
    let δ := C.δ
    let nQ := Fintype.card σ
    let nα := Fintype.card α
    let S : ℕ := nQ * ((nQ + 2) * nα + 1) + 2 * nα + 1
    let G := 3 * θ * (P.L + 1) * (nQ + 1) + p₀ * Real.log (4 * (1 + (nQ + 1) ^ 3 * nα) ^ nQ / δ)
    let p := max (2 * G) y
    (∀ u, P.D.real {x | prefixOf x C.walkStart = u} ≤ p₀) → 0 ≤ θ → 0 < δ →
    0 < C.searchShare → 8 * C.searchShare ≤ ε → ε < 1 → C.middleRate ≤ 1 / 4 →
    C.startRate < 1 / 2 → 0 ≤ C.edgeRate →
    p ≤ C.startRate → 2 * p ≤ C.edgeRate * C.edgeReadShare → 2 * p ≤ C.searchShare * C.middleRate →
    8 * S / C.searchShare * Real.log (16 * S * (S + 1) / δ) + S ≤ C.probeBudget →
    ∫⁻ ω, (Measure.pi fun _ : Fin C.probeBudget => P.D)
        {draws | let result := indecisiveLStarLevel3 C (R.read · ω) draws
          ¬ EndsWell M (BadAt U θ) P.D ε
            {x | Disagrees (R.read · ω) result.hypothesis C.walkStart x}
            C.walkStart (harvestLaw (R.read · ω) P.D C.walkStart result.hypothesis) y
            result.ending} ∂μ
      ≤ ENNReal.ofReal δ

end Random

end OrthoDFA
