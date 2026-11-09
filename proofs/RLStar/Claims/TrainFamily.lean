import Mathlib.Analysis.Asymptotics.Defs
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import RLStar.LowLevelClaims.Clustering

/-!
# The clustering's costs, asymptotically

No strings are harvested. The clustering runs at `η₀ = 1/2 − s₀/4`, `α = δ/2` and cross limit `κ⁴`,
its gate at `εcov`; its schedule is sized for coverage `εc = min(εcov, β/2)` and redraw share
`min(ε_s, εc)`. `ε_s` is the share of the cut its redraws and hash may cost, `ε_u` the indecision
limit, and `β` the least share of the uniform pool either class holds.
-/

namespace OrthoDFA

open MeasureTheory Asymptotics Filter

/-- `nJ` populations. -/
structure TrainFamilyInputs where
  nJ : ℕ
  δ : ℝ
  pAP : ℝ
  εcov : ℝ
  ε_s : ℝ
  ε_u : ℝ
  κ : ℝ
  s₀ : ℝ
  β : ℝ

def trainFamilyRegion : Set TrainFamilyInputs :=
  {x | 1 ≤ x.nJ ∧ 0 < x.δ ∧ x.δ ≤ 1 / 2 ∧ 0 < x.pAP ∧ x.pAP ≤ 1 ∧
    0 < x.εcov ∧ x.εcov ≤ 1 / 2 ∧ 0 < x.ε_s ∧ x.ε_s ≤ 1 / 2 ∧ 0 < x.ε_u ∧ x.ε_u ≤ 1 / 2 ∧
    0 < x.κ ∧ x.κ ≤ 1 ∧ 0 < x.s₀ ∧ x.s₀ ≤ 1 / 2 ∧ 0 < x.β}

noncomputable def TrainFamilyInputs.εc (x : TrainFamilyInputs) : ℝ := min x.εcov (x.β / 2)

noncomputable def TrainFamilyInputs.εsc (x : TrainFamilyInputs) : ℝ := min x.ε_s x.εc

open Real in
noncomputable def TrainFamilyInputs.ℓ_conf (x : TrainFamilyInputs) : ℝ :=
  log (max (1 / (x.εc * x.s₀)) (max (1 / x.ε_u) (1 / x.κ)))

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
  {S : Type*} [Stringlike S]
  {J : Type*} [Fintype J]

/-- `ClusteringGuarantee`'s conclusion at `states`, its conditions on each state's atoms replaced
by `atomTol` and `collTol`. -/
def TrainFamilyOutcome (O : Oracle μ S) (populations tame : Finset J) (uni : J)
    (Pre Suf : Set S) (x : TrainFamilyInputs) (atomTol collTol : ℝ)
    (states : Finset State) : Prop :=
  (∀ B ∈ states, ∀ F : Finset S, F.card + 1 ≤ B.k → ∀ p,
      (B.hi < meanVote O F p → μ.real {ω | voteCount O.mq F p ω ≤ B.lo} ≤ x.κ ^ 4)
      ∧ (meanVote O F p ≤ B.lo → μ.real {ω | B.hi < voteCount O.mq F p ω} ≤ x.κ ^ 4)) ∧
  (∀ B ∈ states, ∀ N, N + 1 ≤ B.k → ∀ p r : ℝ, 0 ≤ p → p ≤ 1 → 0 ≤ r → r ≤ 1 →
      RarerBounded N B.lo (B.hi + 1) p r x.κ) ∧
  ∀ (D : J → Measure S) (Dsf : Measure S),
    (∀ j, IsProbabilityMeasure (D j)) → IsProbabilityMeasure Dsf →
    (∀ j ∈ populations, D j Preᶜ = 0) →
    x.β ≤ (D uni).real {p | p ∈ O.L} →
    x.β ≤ (D uni).real {p | p ∉ O.L} →
    Dsf Sufᶜ = 0 →
    x.pAP ≤ Dsf.real {v | ∀ p, p * v ∈ O.L ↔ p ∈ O.L} →
    (∀ j ∈ tame, ∀ t : S, (D j).real {t} ≤ atomTol) →
    collisionMass Dsf ≤ collTol →
    1 - x.δ ≤ (runMeasure μ D Dsf).real
      {r | let R := streams (fun _ : J => (∅ : Finset S)) uni r
        (∃ B : {B : State // B ∈ states},
            R ∈ ret O.mq populations uni (1 / 2 - x.s₀ / 4) x.ε_u x.εcov (x.δ / 2) B.val)
        ∧ ∀ B : {B : State // B ∈ states},
          R ∈ ret O.mq populations uni (1 / 2 - x.s₀ / 4) x.ε_u x.εcov (x.δ / 2) B.val →
          1 - x.εcov - x.ε_s
            ≤ (D uni).real {p | cutCorrect O B.val.lo (B.val.hi + 1)
                (familyAt O.mq populations R B.val) p (oracleNoise r.1)}
          ∧ ∀ j ∈ tame, 1 / 2 - x.ε_s
            ≤ (D j).real {p | cutCorrect O B.val.lo (B.val.hi + 1)
                (familyAt O.mq populations R B.val) p (oracleNoise r.1)}
            ∧ (D j).real {p | ¬ decided O.mq B.val.lo (B.val.hi + 1)
                (familyAt O.mq populations R B.val) p (oracleNoise r.1)}
              ≤ 2 * x.ε_u}

def TrainFamilyWithin.{u, v, w} (k nsuff npref atomTol collTol : TrainFamilyInputs → ℝ) : Prop :=
  (∀ x, 0 < atomTol x ∧ 0 < collTol x) ∧
    ∀ {Ω : Type u} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
      {S : Type v} [Stringlike S] {J : Type w} [Fintype J]
      (x : TrainFamilyInputs) (O : Oracle μ S) (populations tame : Finset J) (uni : J)
      (Pre Suf : Set S),
    x ∈ trainFamilyRegion →
    populations.card = x.nJ →
    x.s₀ ≤ O.gap →
    uni ∈ tame →
    tame ⊆ populations →
    Flat Pre Suf →
    ∃ states : Finset State,
      (∀ B ∈ states, (B.npref : ℝ) ≤ npref x ∧ (B.k : ℝ) ≤ k x ∧ (B.nsuff : ℝ) ≤ nsuff x) ∧
      TrainFamilyOutcome O populations tame uni Pre Suf x (atomTol x) (collTol x) states

open Real in
def TrainFamilyClaim.{u, v, w} : Prop :=
  ∃ k nsuff npref atomTol collTol : TrainFamilyInputs → ℝ,
    k =O[𝓟 trainFamilyRegion] (fun x => x.ℓ_conf / x.s₀ ^ 2) ∧
    nsuff =O[𝓟 trainFamilyRegion] (fun x =>
      x.ℓ_conf / (x.pAP * x.s₀ ^ 2) + log (1 / x.δ) / x.pAP) ∧
    npref =O[𝓟 trainFamilyRegion] (fun x =>
      x.nJ * log (x.ℓ_conf / (x.δ * x.pAP * x.s₀))
          * (1 / (x.εc ^ 2 * x.s₀ ^ 4) + 1 / (x.ε_u ^ 2 * x.s₀ ^ 4))
        + log (x.nJ / (x.δ * x.pAP * x.εc * x.ε_u * x.κ * x.s₀))
          * (1 / (x.εc ^ 2 * x.s₀ ^ 4) + 1 / x.ε_u ^ 2)
        + log (x.nJ / x.δ) / x.εsc) ∧
    (fun x => 1 / atomTol x) =O[𝓟 trainFamilyRegion] (fun x =>
      (x.nJ * log (x.ℓ_conf / (x.δ * x.pAP * x.s₀))
            * (1 / (x.εc ^ 2 * x.s₀ ^ 4) + 1 / (x.ε_u ^ 2 * x.s₀ ^ 4))
          + log (x.nJ / (x.δ * x.pAP * x.εc * x.ε_u * x.κ * x.s₀))
            * (1 / (x.εc ^ 2 * x.s₀ ^ 4) + 1 / x.ε_u ^ 2))
          * (x.nJ * (1 / x.ε_u + 1 / x.εsc) + 1 / (x.εc * x.s₀ ^ 2) + 1 / (x.ε_u * x.s₀))
        + x.nJ * log (x.nJ / (x.δ * x.pAP * x.εc * x.εsc * x.ε_u * x.κ * x.s₀)) / x.εsc ^ 2) ∧
    (fun x => 1 / collTol x) =O[𝓟 trainFamilyRegion] (fun x =>
      1 / (x.δ * x.pAP) * (x.ℓ_conf / (x.pAP * x.s₀ ^ 2) + log (1 / x.δ) / x.pAP)) ∧
    TrainFamilyWithin.{u, v, w} k nsuff npref atomTol collTol

end OrthoDFA
