import Mathlib.Analysis.Asymptotics.Defs
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import OrthoDFA.LowLevelClaims.Clustering

/-!
# The clustering's costs, asymptotically

`ClusteringClaim`: the clustering stops at a state whose band width `k`, suffix pool `nsuff` and
prefix count `npref` are within the big-O shapes below, some state already within the smaller
`npref₀`, and returns a family as `ClusteringGuarantee` says, wherever the tame populations' atoms
are under `atomTol` and the suffix measure's collision mass under `collTol`; how small those must
be is bounded below the same way. Each bound has one constant over all of `clusterRegion`, where
every instance's `ClusterVec` lies; `s₀ = 1/2 − η₀` is the signal, `ε_u` the indecision limit
and `ε_w` the band's cross limit.
-/

namespace OrthoDFA

open MeasureTheory Asymptotics Filter

/-- `(nJ, 1/δ, 1/α, 1/pAP, 1/εcov, 1/ε_u, 1/ε_w, 1/s₀, nH)`, `nJ` the number of populations and
`nH` of strings harvested. -/
abbrev ClusterVec := ℝ × ℝ × ℝ × ℝ × ℝ × ℝ × ℝ × ℝ × ℝ

def clusterRegion : Set ClusterVec := Set.Ici (1, 2, 2, 1, 2, 2, 1, 2, 0)

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
  {S : Type*} [Stringlike S]
  {J : Type*} [Fintype J]

/-- `ClusteringGuarantee`'s conclusion at `states`, its conditions on each state's atoms replaced
by `atomTol` and `collTol`. -/
def ClusteringOutcome (O : Oracle μ S) (populations tame : Finset J) (uni : J)
    (H : J → Finset S) (Pre Suf : Set S) (η₀ ε_u εcov α δ pAP ε_w atomTol collTol : ℝ)
    (states : Finset State) : Prop :=
  (∀ B ∈ states, ∀ F : Finset S, F.card + 1 ≤ B.k → ∀ p,
      (B.hi < meanVote O F p → μ.real {ω | voteCount O.mq F p ω ≤ B.lo} ≤ ε_w)
      ∧ (meanVote O F p ≤ B.lo → μ.real {ω | B.hi < voteCount O.mq F p ω} ≤ ε_w)) ∧
  ∀ (D : J → Measure S) (Dsf : Measure S),
    (∀ j, IsProbabilityMeasure (D j)) → IsProbabilityMeasure Dsf →
    (∀ j ∈ populations, D j Preᶜ = 0) →
    Dsf Sufᶜ = 0 →
    pAP ≤ Dsf.real {v | ∀ p, p * v ∈ O.L ↔ p ∈ O.L} →
    (∀ j ∈ tame, ∀ t : S, (D j).real {t} ≤ atomTol) →
    collisionMass Dsf ≤ collTol →
    1 - δ - α ≤ (runMeasure μ D Dsf).real
      {x | (∃ B : {B : State // B ∈ states},
            streams H uni x ∈ ret O.mq populations uni η₀ ε_u εcov α B.val)
        ∧ ∀ B : {B : State // B ∈ states},
          streams H uni x ∈ ret O.mq populations uni η₀ ε_u εcov α B.val →
          1 - 513 / 512 * εcov
            ≤ (D uni).real {p | cutCorrect O B.val.lo (B.val.hi + 1)
                (familyAt O.mq populations (streams H uni x) B.val) p (oracleNoise x.1)}
          ∧ ∀ j ∈ tame, 1 / 2 - εcov / 512
            ≤ (D j).real {p | cutCorrect O B.val.lo (B.val.hi + 1)
                (familyAt O.mq populations (streams H uni x) B.val) p (oracleNoise x.1)}
            ∧ (D j).real {p | ¬ decided O.mq B.val.lo (B.val.hi + 1)
                (familyAt O.mq populations (streams H uni x) B.val) p (oracleNoise x.1)}
              ≤ 2 * ε_u}

open Real in
def ClusteringClaim : Prop :=
  ∃ k nsuff npref npref₀ atomTol collTol : ClusterVec → ℝ,
    k =O[𝓟 clusterRegion] (fun (_, _, _, _, ie, iu, iw, is, _) =>
      log (ie * iu * iw * is) * is ^ 2) ∧
    nsuff =O[𝓟 clusterRegion] (fun (nJ, iδ, _, ip, ie, iu, iw, is, _) =>
      log (ie * iu * iw * is) * ip * is ^ 2 + log (nJ * iδ) * ip ^ 2) ∧
    npref₀ =O[𝓟 clusterRegion] (fun (nJ, iδ, _, _, ie, iu, _, is, _) =>
      log (nJ * iδ) * (iu ^ 2 + iu * is + ie * is ^ 3)) ∧
    npref =O[𝓟 clusterRegion] (fun (nJ, iδ, iα, ip, ie, iu, iw, is, nH) =>
      nJ ^ 2 * log (nJ * iδ * iα * ip * ie * iu * iw * is) * (ie ^ 2 * is ^ 8 + iu ^ 2 * is ^ 4)
        + nH * (ie * is ^ 2 + iu)) ∧
    (fun v => 1 / atomTol v) =O[𝓟 clusterRegion] (fun (nJ, iδ, iα, ip, ie, iu, iw, is, nH) =>
      nJ * (nJ ^ 2 * log (nJ * iδ * iα * ip * ie * iu * iw * is)
              * (ie ^ 2 * is ^ 8 + iu ^ 2 * is ^ 4)
          + nH * (ie * is ^ 2 + iu)) * (ie * is ^ 3 + iu * is)
        + nJ * iδ * log (nJ * iδ * iα * ip * ie * iu * iw * is * (nH + 1)) * (ie ^ 2 + iu ^ 2)) ∧
    (fun v => 1 / collTol v) =O[𝓟 clusterRegion] (fun (nJ, iδ, _, ip, ie, iu, iw, is, _) =>
      nJ * iδ * ip * (log (ie * iu * iw * is) * ip * is ^ 2 + log (nJ * iδ) * ip ^ 2)) ∧
    (∀ v, 0 < atomTol v ∧ 0 < collTol v) ∧
    ∀ {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
      {S : Type*} [Stringlike S] {J : Type*} [Fintype J]
      (O : Oracle μ S) (populations tame : Finset J) (uni : J) (H : J → Finset S)
      (Pre Suf : Set S) (η₀ ε_u εcov α δ pAP ε_w : ℝ),
    O.η ≤ η₀ →
    0 ≤ η₀ →
    η₀ < 1 / 2 →
    uni ∈ tame →
    tame ⊆ populations →
    Flat Pre Suf →
    (∀ j, ∀ p ∈ H j, p ∈ Pre) →
    0 < pAP →
    pAP ≤ 1 →
    0 < ε_u →
    ε_u ≤ 1 / 2 →
    0 < α →
    α < 1 / 2 →
    0 < εcov →
    εcov ≤ 1 / 2 →
    0 < δ →
    δ ≤ 1 / 2 →
    0 < ε_w →
    ε_w ≤ 1 →
    let v : ClusterVec := (populations.card, 1 / δ, 1 / α, 1 / pAP, 1 / εcov, 1 / ε_u, 1 / ε_w,
      1 / (1 / 2 - η₀), (allHarvest H).card)
    v ∈ clusterRegion ∧ ∃ states : Finset State,
      (∀ B ∈ states, (B.npref : ℝ) ≤ npref v ∧ (B.k : ℝ) ≤ k v ∧ (B.nsuff : ℝ) ≤ nsuff v) ∧
      (∃ B ∈ states, (B.npref : ℝ) ≤ npref₀ v) ∧
      ClusteringOutcome O populations tame uni H Pre Suf η₀ ε_u εcov α δ pAP ε_w
        (atomTol v) (collTol v) states

end OrthoDFA
