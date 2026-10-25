import Mathlib.Analysis.Asymptotics.Defs
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import OrthoDFA.Clustering

/-!
# The clustering's costs, asymptotically

`ClusteringClaim`: with no strings harvested, the clustering stops at a state whose band width
`k`, suffix pool `nsuff` and prefix count `npref` are within the big-O shapes below, some state
already within the smaller `npref₀`, and returns a family as `ClusteringGuarantee` says but for
chance `δ`, wherever the tame populations' atoms are under `atomTol` and the suffix measure's
collision mass under `collTol`; how small those must be is bounded below the same way. Each bound
has one constant over all of `clusterRegion`. `ClusteringClaimSimple` states the same with each
shape a single product. The algorithm runs at `η₀ = 1/2 − s₀`, `s₀` the signal, and at
`α = δ/2`; `ε_u` is the indecision limit and `ε_w` the band's cross limit.
-/

namespace OrthoDFA

open MeasureTheory Asymptotics Filter

/-- The clustering's inputs: `nJ` populations and its rates. -/
structure ClusterInputs where
  nJ : ℕ
  δ : ℝ
  pAP : ℝ
  εcov : ℝ
  ε_u : ℝ
  ε_w : ℝ
  s₀ : ℝ

def clusterRegion : Set ClusterInputs :=
  {x | 1 ≤ x.nJ ∧ 0 < x.δ ∧ x.δ ≤ 1 / 2 ∧ 0 < x.pAP ∧ x.pAP ≤ 1 ∧
    0 < x.εcov ∧ x.εcov ≤ 1 / 2 ∧ 0 < x.ε_u ∧ x.ε_u ≤ 1 / 2 ∧ 0 < x.ε_w ∧ x.ε_w ≤ 1 ∧
    0 < x.s₀ ∧ x.s₀ ≤ 1 / 2}

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
  {S : Type*} [Stringlike S]
  {J : Type*} [Fintype J]

/-- `ClusteringGuarantee`'s conclusion at `states`, its conditions on each state's atoms replaced
by `atomTol` and `collTol`. -/
def ClusteringOutcome (O : Oracle μ S) (populations tame : Finset J) (uni : J)
    (Pre Suf : Set S) (x : ClusterInputs) (atomTol collTol : ℝ)
    (states : Finset State) : Prop :=
  (∀ B ∈ states, ∀ F : Finset S, F.card + 1 ≤ B.k → ∀ p,
      (B.hi < meanVote O F p → μ.real {ω | voteCount O.mq F p ω ≤ B.lo} ≤ x.ε_w)
      ∧ (meanVote O F p ≤ B.lo → μ.real {ω | B.hi < voteCount O.mq F p ω} ≤ x.ε_w)) ∧
  ∀ (D : J → Measure S) (Dsf : Measure S),
    (∀ j, IsProbabilityMeasure (D j)) → IsProbabilityMeasure Dsf →
    (∀ j ∈ populations, D j Preᶜ = 0) →
    Dsf Sufᶜ = 0 →
    x.pAP ≤ Dsf.real {v | ∀ p, p * v ∈ O.L ↔ p ∈ O.L} →
    (∀ j ∈ tame, ∀ t : S, (D j).real {t} ≤ atomTol) →
    collisionMass Dsf ≤ collTol →
    1 - x.δ ≤ (runMeasure μ D Dsf).real
      {r | let R := streams (fun _ : J => (∅ : Finset S)) uni r
        (∃ B : {B : State // B ∈ states},
            R ∈ ret O.mq populations uni (1 / 2 - x.s₀) x.ε_u x.εcov (x.δ / 2) B.val)
        ∧ ∀ B : {B : State // B ∈ states},
          R ∈ ret O.mq populations uni (1 / 2 - x.s₀) x.ε_u x.εcov (x.δ / 2) B.val →
          1 - 513 / 512 * x.εcov
            ≤ (D uni).real {p | cutCorrect O B.val.lo (B.val.hi + 1)
                (familyAt O.mq populations R B.val) p (oracleNoise r.1)}
          ∧ ∀ j ∈ tame, 1 / 2 - x.εcov / 512
            ≤ (D j).real {p | cutCorrect O B.val.lo (B.val.hi + 1)
                (familyAt O.mq populations R B.val) p (oracleNoise r.1)}
            ∧ (D j).real {p | ¬ decided O.mq B.val.lo (B.val.hi + 1)
                (familyAt O.mq populations R B.val) p (oracleNoise r.1)}
              ≤ 2 * x.ε_u}

/-- Positive tolerances, and on every instance in `clusterRegion`, with no strings harvested,
states within `k`, `nsuff` and `npref`, one within `npref₀`, that meet `ClusteringOutcome`. -/
def ClusteringWithin.{u, v, w} (k nsuff npref npref₀ atomTol collTol : ClusterInputs → ℝ) : Prop :=
  (∀ x, 0 < atomTol x ∧ 0 < collTol x) ∧
    ∀ {Ω : Type u} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
      {S : Type v} [Stringlike S] {J : Type w} [Fintype J]
      (x : ClusterInputs) (O : Oracle μ S) (populations tame : Finset J) (uni : J)
      (Pre Suf : Set S),
    x ∈ clusterRegion →
    populations.card = x.nJ →
    O.η ≤ 1 / 2 - x.s₀ →
    uni ∈ tame →
    tame ⊆ populations →
    Flat Pre Suf →
    ∃ states : Finset State,
      (∀ B ∈ states, (B.npref : ℝ) ≤ npref x ∧ (B.k : ℝ) ≤ k x ∧ (B.nsuff : ℝ) ≤ nsuff x) ∧
      (∃ B ∈ states, (B.npref : ℝ) ≤ npref₀ x) ∧
      ClusteringOutcome O populations tame uni Pre Suf x (atomTol x) (collTol x) states

open Real in
def ClusteringClaim.{u, v, w} : Prop :=
  ∃ k nsuff npref npref₀ atomTol collTol : ClusterInputs → ℝ,
    k =O[𝓟 clusterRegion] (fun x => log (1 / (x.εcov * x.ε_u * x.ε_w * x.s₀)) / x.s₀ ^ 2) ∧
    nsuff =O[𝓟 clusterRegion] (fun x =>
      log (1 / (x.εcov * x.ε_u * x.ε_w * x.s₀)) / (x.pAP * x.s₀ ^ 2)
        + log (x.nJ / x.δ) / x.pAP ^ 2) ∧
    npref₀ =O[𝓟 clusterRegion] (fun x =>
      log (x.nJ / x.δ) * (1 / x.ε_u ^ 2 + 1 / (x.ε_u * x.s₀) + 1 / (x.εcov * x.s₀ ^ 3))) ∧
    npref =O[𝓟 clusterRegion] (fun x =>
      x.nJ ^ 2 * log (x.nJ / (x.δ * x.pAP * x.εcov * x.ε_u * x.ε_w * x.s₀))
          * (1 / (x.εcov ^ 2 * x.s₀ ^ 8) + 1 / (x.ε_u ^ 2 * x.s₀ ^ 4))) ∧
    (fun x => 1 / atomTol x) =O[𝓟 clusterRegion] (fun x =>
      x.nJ ^ 3 * log (x.nJ / (x.δ * x.pAP * x.εcov * x.ε_u * x.ε_w * x.s₀))
          * (1 / (x.εcov ^ 2 * x.s₀ ^ 8) + 1 / (x.ε_u ^ 2 * x.s₀ ^ 4))
          * (1 / (x.εcov * x.s₀ ^ 3) + 1 / (x.ε_u * x.s₀))
        + x.nJ / x.δ * log (x.nJ / (x.δ * x.pAP * x.εcov * x.ε_u * x.ε_w * x.s₀))
          * (1 / x.εcov ^ 2 + 1 / x.ε_u ^ 2)) ∧
    (fun x => 1 / collTol x) =O[𝓟 clusterRegion] (fun x =>
      x.nJ / (x.δ * x.pAP) * (log (1 / (x.εcov * x.ε_u * x.ε_w * x.s₀)) / (x.pAP * x.s₀ ^ 2)
        + log (x.nJ / x.δ) / x.pAP ^ 2)) ∧
    ClusteringWithin.{u, v, w} k nsuff npref npref₀ atomTol collTol

open Real in
/-- `ClusteringClaim` with each shape a single product. -/
def ClusteringClaimSimple.{u, v, w} : Prop :=
  ∃ k nsuff npref npref₀ atomTol collTol : ClusterInputs → ℝ,
    k =O[𝓟 clusterRegion] (fun x => log (1 / (x.εcov * x.ε_u * x.ε_w * x.s₀)) / x.s₀ ^ 2) ∧
    nsuff =O[𝓟 clusterRegion] (fun x =>
      log (x.nJ / (x.δ * x.εcov * x.ε_u * x.ε_w * x.s₀)) / (x.pAP ^ 2 * x.s₀ ^ 2)) ∧
    npref₀ =O[𝓟 clusterRegion] (fun x =>
      log (x.nJ / x.δ) / (x.εcov * x.ε_u ^ 2 * x.s₀ ^ 3)) ∧
    npref =O[𝓟 clusterRegion] (fun x =>
      x.nJ ^ 2 * log (x.nJ / (x.δ * x.pAP * x.εcov * x.ε_u * x.ε_w * x.s₀))
        / (x.εcov ^ 2 * x.ε_u ^ 2 * x.s₀ ^ 8)) ∧
    (fun x => 1 / atomTol x) =O[𝓟 clusterRegion] (fun x =>
      x.nJ ^ 3 * log (x.nJ / (x.δ * x.pAP * x.εcov * x.ε_u * x.ε_w * x.s₀))
        / (x.δ * x.εcov ^ 3 * x.ε_u ^ 3 * x.s₀ ^ 11)) ∧
    (fun x => 1 / collTol x) =O[𝓟 clusterRegion] (fun x =>
      x.nJ * log (x.nJ / (x.δ * x.εcov * x.ε_u * x.ε_w * x.s₀)) / (x.δ * x.pAP ^ 3 * x.s₀ ^ 2)) ∧
    ClusteringWithin.{u, v, w} k nsuff npref npref₀ atomTol collTol

end OrthoDFA
