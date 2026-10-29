import Mathlib.Analysis.Asymptotics.Defs
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import OrthoDFA.LowLevelClaims.Clustering

/-!
# The clustering's costs, asymptotically

`ClusteringClaim`: with no strings harvested, the clustering stops at a state whose band width
`k`, suffix pool `nsuff` and prefix count `npref` are within the big-O shapes below, and returns a
family as `ClusteringGuarantee` says but for chance `δ`, wherever the tame populations' atoms are
under `atomTol` and the suffix measure's collision mass under `collTol`; how small those must be is
bounded below the same way. Each bound has one constant over all of `clusterRegion`.
`ClusteringClaimSimple` states the same with each shape a single product. The algorithm runs at
`η₀ = 1/2 − s₀`, `s₀` the signal, at `α = δ/2` and at cross limit `κ⁴`; `ε_s` is the share of the
cut its redraws and hash may cost, `ε_u` the indecision limit, and every state's band reads its
rarer decided side at most `κ` times as often as undecided.
-/

namespace OrthoDFA

open MeasureTheory Asymptotics Filter

/-- The clustering's inputs: `nJ` populations and its rates. -/
structure ClusterInputs where
  nJ : ℕ
  δ : ℝ
  pAP : ℝ
  εcov : ℝ
  ε_s : ℝ
  ε_u : ℝ
  κ : ℝ
  s₀ : ℝ

def clusterRegion : Set ClusterInputs :=
  {x | 1 ≤ x.nJ ∧ 0 < x.δ ∧ x.δ ≤ 1 / 2 ∧ 0 < x.pAP ∧ x.pAP ≤ 1 ∧
    0 < x.εcov ∧ x.εcov ≤ 1 / 2 ∧ 0 < x.ε_s ∧ x.ε_s ≤ 1 / 2 ∧ 0 < x.ε_u ∧ x.ε_u ≤ 1 / 2 ∧
    0 < x.κ ∧ x.κ ≤ 1 ∧ 0 < x.s₀ ∧ x.s₀ ≤ 1 / 2}

open Real in
noncomputable def ClusterInputs.ℓ_conf (x : ClusterInputs) : ℝ :=
  log (max (1 / (x.εcov * x.s₀)) (max (1 / x.ε_u) (1 / x.κ)))

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
  {S : Type*} [Stringlike S]
  {J : Type*} [Fintype J]

/-- `ClusteringGuarantee`'s conclusion at `states`, its conditions on each state's atoms replaced
by `atomTol` and `collTol`. -/
def ClusteringOutcome (O : Oracle μ S) (populations tame : Finset J) (uni : J)
    (Pre Suf : Set S) (x : ClusterInputs) (atomTol collTol : ℝ)
    (states : Finset State) : Prop :=
  (∀ B ∈ states, ∀ F : Finset S, F.card + 1 ≤ B.k → ∀ p,
      (B.hi < meanVote O F p → μ.real {ω | voteCount O.mq F p ω ≤ B.lo} ≤ x.κ ^ 4)
      ∧ (meanVote O F p ≤ B.lo → μ.real {ω | B.hi < voteCount O.mq F p ω} ≤ x.κ ^ 4)) ∧
  (∀ B ∈ states, ∀ N, N + 1 ≤ B.k → ∀ p r : ℝ, 0 ≤ p → p ≤ 1 → 0 ≤ r → r ≤ 1 →
      RarerBounded N B.lo (B.hi + 1) p r x.κ) ∧
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
          1 - x.εcov - x.ε_s
            ≤ (D uni).real {p | cutCorrect O B.val.lo (B.val.hi + 1)
                (familyAt O.mq populations R B.val) p (oracleNoise r.1)}
          ∧ ∀ j ∈ tame, 1 / 2 - x.ε_s
            ≤ (D j).real {p | cutCorrect O B.val.lo (B.val.hi + 1)
                (familyAt O.mq populations R B.val) p (oracleNoise r.1)}
            ∧ (D j).real {p | ¬ decided O.mq B.val.lo (B.val.hi + 1)
                (familyAt O.mq populations R B.val) p (oracleNoise r.1)}
              ≤ 2 * x.ε_u}

/-- Positive tolerances, and on every instance in `clusterRegion`, with no strings harvested,
states within `k`, `nsuff` and `npref` that meet `ClusteringOutcome`. -/
def ClusteringWithin.{u, v, w} (k nsuff npref atomTol collTol : ClusterInputs → ℝ) : Prop :=
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
      ClusteringOutcome O populations tame uni Pre Suf x (atomTol x) (collTol x) states

open Real in
def ClusteringClaim.{u, v, w} : Prop :=
  ∃ k nsuff npref atomTol collTol : ClusterInputs → ℝ,
    k =O[𝓟 clusterRegion] (fun x => x.ℓ_conf / x.s₀ ^ 2) ∧
    nsuff =O[𝓟 clusterRegion] (fun x =>
      x.ℓ_conf / (x.pAP * x.s₀ ^ 2) + log (1 / x.δ) / x.pAP) ∧
    npref =O[𝓟 clusterRegion] (fun x =>
      x.nJ ^ 2 * log (x.nJ / (x.δ * x.pAP * x.εcov * x.ε_u * x.κ * x.s₀))
          * (1 / (x.εcov ^ 2 * x.s₀ ^ 8) + 1 / (x.ε_u ^ 2 * x.s₀ ^ 4))
        + log (x.nJ / x.δ) / x.ε_s) ∧
    (fun x => 1 / atomTol x) =O[𝓟 clusterRegion] (fun x =>
      x.nJ ^ 3 * log (x.nJ / (x.δ * x.pAP * x.εcov * x.ε_u * x.κ * x.s₀))
          * (1 / (x.εcov ^ 2 * x.s₀ ^ 8) + 1 / (x.ε_u ^ 2 * x.s₀ ^ 4))
          * (1 / (x.εcov * x.s₀ ^ 3) + 1 / (x.ε_u * x.s₀) + 1 / x.ε_s)
        + x.nJ / x.δ * log (x.nJ / (x.δ * x.pAP * x.εcov * x.ε_s * x.ε_u * x.κ * x.s₀))
          * (1 / x.ε_s ^ 2 + 1 / x.ε_u ^ 2)) ∧
    (fun x => 1 / collTol x) =O[𝓟 clusterRegion] (fun x =>
      x.nJ / (x.δ * x.pAP) * (x.ℓ_conf / (x.pAP * x.s₀ ^ 2) + log (1 / x.δ) / x.pAP)) ∧
    ClusteringWithin.{u, v, w} k nsuff npref atomTol collTol

open Real in
/-- `ClusteringClaim` with each shape a single product. -/
def ClusteringClaimSimple.{u, v, w} : Prop :=
  ∃ k nsuff npref atomTol collTol : ClusterInputs → ℝ,
    k =O[𝓟 clusterRegion] (fun x => x.ℓ_conf / x.s₀ ^ 2) ∧
    nsuff =O[𝓟 clusterRegion] (fun x =>
      (x.ℓ_conf + log (1 / x.δ)) / (x.pAP * x.s₀ ^ 2)) ∧
    npref =O[𝓟 clusterRegion] (fun x =>
      x.nJ ^ 2 * log (x.nJ / (x.δ * x.pAP * x.εcov * x.ε_s * x.ε_u * x.κ * x.s₀))
        / (x.εcov ^ 2 * x.ε_s * x.ε_u ^ 2 * x.s₀ ^ 8)) ∧
    (fun x => 1 / atomTol x) =O[𝓟 clusterRegion] (fun x =>
      x.nJ ^ 3 * log (x.nJ / (x.δ * x.pAP * x.εcov * x.ε_s * x.ε_u * x.κ * x.s₀))
        / (x.δ * x.εcov ^ 3 * x.ε_s ^ 2 * x.ε_u ^ 3 * x.s₀ ^ 11)) ∧
    (fun x => 1 / collTol x) =O[𝓟 clusterRegion] (fun x =>
      x.nJ * (x.ℓ_conf + log (1 / x.δ)) / (x.δ * x.pAP ^ 2 * x.s₀ ^ 2)) ∧
    ClusteringWithin.{u, v, w} k nsuff npref atomTol collTol

end OrthoDFA
