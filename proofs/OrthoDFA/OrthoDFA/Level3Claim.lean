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

Level 3 ends well as `OrthoDFA.RoundEnd` says, the disagreements being the probes that search,
change the hypothesis, or stop undecided at their start or end sift. Over `C.probeBudget` probes,
it fails to, a harvest's sampler yielding `y`, with chance at most `δ`.
`S = |Q|((|Q| + 2)|Σ| + 1) + 2|Σ| + 1` bounds the hypotheses: each split adds a leaf, at most
`|Q| + 2` of them, and between splits each of a leaf's edges is learned once.
-/

namespace OrthoDFA

namespace Random

open MeasureTheory

variable {α : Type*}

/-- The strings a probe of `x` walked from `k` under `H` reads undecided at the site, as the tests
count them. -/
noncomputable def siteHarvest (read : FreeMonoid α → ARU) (k : ℕ) (H : HypothesisDFA α) :
    Level3Site → FreeMonoid α → List (FreeMonoid α)
  | .ends, x => endsString (probeOrAgree read H.tree H.edges k x)
  | .middles, x => middleString (probeOrAgree read H.tree H.edges k x)

/-- The law of what the harvest's sampler finds, each string as often as a probe drawn from `D`
and walked under `H` finds it at the site. -/
noncomputable def harvestLaw [Countable α] (read : FreeMonoid α → ARU) (D : Measure (FreeMonoid α))
    (k : ℕ) (H : HypothesisDFA α) (site : Level3Site) :
    Measure (FreeMonoid α) :=
  D.bind fun x => ((siteHarvest read k H site x).map Measure.dirac).sum

/-- The probes that, walked from `k` under `H`, cross a wrong edge, stop at an unlearned one, or
part from their sift at decided reads. -/
def searchSet (read : FreeMonoid α → ARU) (k : ℕ) (H : HypothesisDFA α) : Set (FreeMonoid α) :=
  {x | match probeOrAgree read H.tree H.edges k x with
    | .wrongEdge .. | .unlearnedEdge .. | .undecided _ false => True
    | _ => False}

/-- The probes whose start or end sift, walked from `k` under `H`, stops undecided. -/
def endsSet (read : FreeMonoid α → ARU) (k : ℕ) (H : HypothesisDFA α) : Set (FreeMonoid α) :=
  {x | endsString (probeOrAgree read H.tree H.edges k x) ≠ []}

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
of the probes at most `p₀` of them, `2G ≤ y ≤ ε/48` (`harvestRate`, `disagreementRate`), `ε < 1`,
and `C.probeBudget` at least `36 S / ε · log(16 S (S + 1) / δ) + S`, `S` bounding the hypotheses,
level 3 fails to end well, a state being bad where it is undecided more than `1.5θ` of the time,
with chance at most `δ`. A consistent ending has at most `ε` of the probes searching, changing
the hypothesis or stopping undecided at their start or end sift, and a harvest's sampler yields
`y`. `G` bounds what the probes can find undecided at good states but
for chance `δ / 4`. -/
def RandomRoundCorrect : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] {σ : Type*} [Fintype σ] (M : DFA α σ)
    (side : σ → Bool) (U : σ → ℝ) (θ : ℝ) {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω)
    [IsProbabilityMeasure μ] (R : ReadOracle μ M side U) (P : PrefixDistribution α)
    (C : Level3Settings) (p₀ : ℝ),
    let y := C.harvestRate
    let ε := C.disagreementRate
    let δ := C.δ
    let nQ := Fintype.card σ
    let nα := Fintype.card α
    let S : ℕ := nQ * ((nQ + 2) * nα + 1) + 2 * nα + 1
    let G := 3 * θ * (P.L + 1) * (nQ + 1) + p₀ * Real.log (4 * (1 + (nQ + 1) ^ 3 * nα) ^ nQ / δ)
    (∀ u, P.D.real {x | prefixOf x C.walkStart = u} ≤ p₀) → 0 ≤ θ → 0 < δ →
    2 * G ≤ y → y ≤ ε / 48 → ε < 1 →
    36 * S / ε * Real.log (16 * S * (S + 1) / δ) + S ≤ C.probeBudget →
    ∫⁻ ω, (Measure.pi fun _ : Fin C.probeBudget => P.D)
        {draws | let result := indecisiveLStarLevel3 C (R.read · ω) draws
          ¬ EndsWell M (BadAt U θ) P.D ε
            (searchSet (R.read · ω) C.walkStart result.hypothesis
              ∪ endsSet (R.read · ω) C.walkStart result.hypothesis)
            C.walkStart (harvestLaw (R.read · ω) P.D C.walkStart result.hypothesis) y
            result.ending} ∂μ
      ≤ ENNReal.ofReal δ

end Random

end OrthoDFA
