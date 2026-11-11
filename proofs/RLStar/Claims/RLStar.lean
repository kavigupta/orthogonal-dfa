import Mathlib.Analysis.Asymptotics.Defs
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import RLStar.LowLevelClaims.LoopDFA
import RLStar.Code.LearnerLoopTrie

/-!
# The learner's costs, asymptotically
-/

namespace OrthoDFA

open MeasureTheory Asymptotics Filter

/-- `Q` states, `A` letters, `ξ` the accuracy slack, `pAP` the share of suffixes that preserve
membership, and the signal `s₀ = 1/2 − η₀`. -/
structure RLStarInputs where
  Q : ℕ
  A : ℕ
  δ : ℝ
  ξ : ℝ
  pAP : ℝ
  s₀ : ℝ

def rlStarRegion : Set RLStarInputs :=
  {x | 1 ≤ x.Q ∧ 2 ≤ x.A ∧ 0 < x.δ ∧ x.δ ≤ 1 / 2 ∧ 0 < x.ξ ∧ x.ξ ≤ 1 ∧
    0 < x.pAP ∧ x.pAP ≤ 1 ∧ 0 < x.s₀ ∧ x.s₀ ≤ 1 / 2}

/-- `L` the Python's length. -/
structure RLStarRun extends RLStarInputs where
  L : ℕ

def rlStarRunRegion : Set RLStarRun := {r | r.toRLStarInputs ∈ rlStarRegion}

def DependsOnlyOn {X Y : Type*} (f : (X → ℝ) → Y) (oracle : X → ℝ) (S : Finset X) : Prop :=
  ∀ o : X → ℝ, (∀ w ∈ S, o w = oracle w) → f o = f oracle

def RLStarSucceeds {A : Type*} [Fintype A] [DecidableEq A] {Q : Type*} [Fintype Q] {Ω : Type*}
    [MeasurableSpace Ω] {μ : Measure Ω} {K : ℕ} (settings : LoopSettings K) (cap : ℕ)
    (O : Oracle μ (FreeMonoid A)) (D Dsf T : Measure (FreeMonoid A)) (M : _root_.DFA A Q) (ε : ℝ)
    (totalReads : ℝ)
    (p : LoopDraws Ω A K settings.level3.harvestReadCap settings.level3.probeBudget) : Prop :=
  (∃ rounds, indecisiveLStarTrieLoop settings cap O D Dsf p = some rounds ∧
    (rounds.length ≤ Fintype.card Q + 1 ∧ ∃ o ∈ rounds.getLast?,
        o.result.ending = .consistent ∧ T.real {w | ¬
          o.result.hypothesis.TrieClassifies (o.read O p.1) o.result.expanded w
            (M.eval w.toList ∈ M.accept)} ≤ ε) ∧
    ∀ o ∈ rounds, o.result.hypothesis.tree.leaves.length ≤ Fintype.card Q + 2) ∧
  ∃ S : Finset (FreeMonoid A), (S.card : ℝ) ≤ totalReads ∧
    DependsOnlyOn (fun o => (trieLoopRoundsQ settings cap D Dsf p.2 K).eval o)
      (fun w => O.mq w p.1) S

def RLStarWithin.{u, v, w} (p₀Tol totalPref totalSuff totalProbes totalReads : RLStarRun → ℝ) :
    Prop :=
  ∀ {A : Type u} [Fintype A] [DecidableEq A] [Nonempty A] {Q : Type v}
    [Fintype Q] {Ω : Type w}
    [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
    (x : RLStarInputs) (M : _root_.DFA A Q) (O : Oracle μ (FreeMonoid A)) (L k : ℕ)
    (D : Measure (FreeMonoid A)) [IsProbabilityMeasure D],
  let L₁ : ℕ := L + Fintype.card Q + 3
  let Ls : ℕ := L₁ + Fintype.card Q + 3
  x ∈ rlStarRegion → Fintype.card Q = x.Q → Fintype.card A = x.A →
  (∀ w, w ∈ O.L ↔ M.eval w.toList ∈ M.accept) →
  O.η ≤ 1 / 2 - x.s₀ →
  (∀ᵐ w ∂D, Ls ≤ w.toList.length) →
  k ≤ L →
  (∀ u, D.real {w | prefixOf w k = u} ≤ p₀Tol { x with L }) →
  x.pAP ≤ (prefixLaw D Ls).real {v | ∀ p, p * v ∈ O.L ↔ p ∈ O.L} →
  ∃ settings : LoopSettings (Fintype.card Q + 1),
    settings.level3.walkStart = k ∧
    (∀ p : LoopDraws Ω A (Fintype.card Q + 1) settings.level3.harvestReadCap
          settings.level3.probeBudget,
      ∀ rounds,
        indecisiveLStarTrieLoop settings k O (prefixLaw D L) (prefixLaw D Ls) p = some rounds →
      ((rounds.map fun o => o.prefs.card).sum : ℝ) ≤ totalPref { x with L }
        ∧ ((rounds.map fun o => o.suffs.card).sum : ℝ) ≤ totalSuff { x with L }
        ∧ (rounds.length * settings.level3.probeBudget : ℝ) ≤ totalProbes { x with L }) ∧
    (loopMeasure μ A (Fintype.card Q + 1) settings.level3.harvestReadCap
        settings.level3.probeBudget).real
      {p | ¬ RLStarSucceeds settings k O (prefixLaw D L) (prefixLaw D Ls) (prefixLaw D L) M
        x.ξ (totalReads { x with L }) p}
      ≤ x.δ

open Real in
def RLStarClaim.{u, v, w} : Prop :=
  ∃ (p₀Tol totalPref totalSuff totalProbes totalReads : RLStarRun → ℝ),
    (fun r => 1 / p₀Tol r) =O[𝓟 rlStarRunRegion] (fun r =>
      ((r.L + r.Q) ^ 4 * log ((r.L + r.Q) * r.A / (r.δ * r.ξ * r.pAP * r.s₀))
          * log (r.Q * r.A / (r.δ * r.ξ * r.pAP * r.s₀)) * r.Q ^ 3 * r.A / r.ξ
          * (r.Q ^ 2 * r.A ^ 2 / r.ξ ^ 3
              * (1 / r.s₀ ^ 6 + r.Q ^ 2 * r.A / r.s₀ ^ 5 + r.Q ^ 4 * r.A / r.s₀ ^ 4)
            + 1 / (r.δ * r.pAP ^ 2 * r.s₀ ^ 2))
        + (r.L + r.Q) ^ 2 * log ((r.L + r.Q) * r.A / (r.δ * r.ξ * r.pAP * r.s₀))
          * log (r.Q * r.A / (r.δ * r.ξ * r.pAP * r.s₀)) ^ 2 * r.Q ^ 6 * r.A ^ 2
          / (r.ξ ^ 4 * r.δ * r.s₀ ^ 4)
        + (r.L + r.Q) ^ 2 * log (r.Q * r.A / (r.δ * r.ξ * r.pAP * r.s₀)) ^ 2
          * r.Q ^ 5 * r.A ^ 4 / (r.ξ ^ 2 * r.δ)) ^ 5) ∧
    totalPref =O[𝓟 rlStarRunRegion] (fun r =>
      r.Q ^ 7 * r.A ^ 2 * (r.L + r.Q) ^ 2 / (r.ξ ^ 2 * r.s₀ ^ 4)
          * log (r.Q * log ((r.L + r.Q) * r.A / (r.δ * r.ξ * r.s₀)) / (r.δ * r.pAP * r.s₀))
        + r.Q ^ 2 * (r.L + r.Q) ^ 2 / r.ξ ^ 2
          * (1 / r.s₀ ^ 4 + r.Q ^ 4 * r.A ^ 2)
          * log ((r.L + r.Q) * r.A / (r.δ * r.ξ * r.pAP * r.s₀))) ∧
    totalSuff =O[𝓟 rlStarRunRegion] (fun r =>
      r.Q * (log ((r.L + r.Q) * r.A / (r.δ * r.ξ * r.s₀)) / (r.pAP * r.s₀ ^ 2)
        + log (r.Q / r.δ) / r.pAP)) ∧
    totalProbes =O[𝓟 rlStarRunRegion] (fun r =>
      r.Q ^ 3 * r.A ^ 2 * (r.L + r.Q) / r.ξ * log (r.Q * r.A / r.δ)) ∧
    totalReads =O[𝓟 rlStarRunRegion] (fun r =>
      let nPref : ℝ :=
        r.Q ^ 7 * r.A ^ 2 * (r.L + r.Q) ^ 2 / (r.ξ ^ 2 * r.s₀ ^ 4)
            * log (r.Q * log ((r.L + r.Q) * r.A / (r.δ * r.ξ * r.s₀)) / (r.δ * r.pAP * r.s₀))
          + r.Q ^ 2 * (r.L + r.Q) ^ 2 / r.ξ ^ 2
            * (1 / r.s₀ ^ 4 + r.Q ^ 4 * r.A ^ 2)
            * log ((r.L + r.Q) * r.A / (r.δ * r.ξ * r.pAP * r.s₀))
      let nSuff : ℝ :=
        r.Q * (log ((r.L + r.Q) * r.A / (r.δ * r.ξ * r.s₀)) / (r.pAP * r.s₀ ^ 2)
          + log (r.Q / r.δ) / r.pAP)
      let nProbe : ℝ :=
        r.Q ^ 3 * r.A ^ 2 * (r.L + r.Q) / r.ξ * log (r.Q * r.A / r.δ)
      -- `|V̂|`, a round's suffix family
      let nFam : ℝ := log ((r.L + r.Q) * r.A / (r.δ * r.ξ * r.s₀)) / r.s₀ ^ 2
      let ρInv : ℝ := r.A * (r.L + r.Q) / r.ξ
      let lg : ℝ := log (r.L + r.Q + 1)
      -- clusterings, counterexample search, harvest
      nPref * nSuff / r.Q + nProbe * nFam * r.Q * lg + r.Q * nPref * nFam * lg * ρInv) ∧
    (∀ r, 0 < p₀Tol r) ∧
    RLStarWithin.{u, v, w} p₀Tol totalPref totalSuff totalProbes totalReads

def RLStarWithinUniform.{u, v, w} (k₀ : RLStarRun → ℝ)
    (totalPref totalSuff totalProbes totalReads : RLStarRun → ℝ) : Prop :=
  ∀ {A : Type u} [Fintype A] [DecidableEq A] [Nonempty A] {Q : Type v}
    [Fintype Q] {Ω : Type w}
    [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
    (x : RLStarInputs) (M : _root_.DFA A Q) (O : Oracle μ (FreeMonoid A)) (L k : ℕ),
  let L₁ : ℕ := L + Fintype.card Q + 3
  x ∈ rlStarRegion → Fintype.card Q = x.Q → Fintype.card A = x.A →
  (∀ w, w ∈ O.L ↔ M.eval w.toList ∈ M.accept) →
  O.η ≤ 1 / 2 - x.s₀ →
  x.pAP ≤ (uniformStrings (L₁ + Fintype.card Q + 3)).real {v | ∀ p, p * v ∈ O.L ↔ p ∈ O.L} →
  k₀ { x with L } ≤ k → k ≤ L →
  ∃ settings : LoopSettings (Fintype.card Q + 1),
    settings.level3.walkStart = k ∧
    (∀ p : LoopDraws Ω A (Fintype.card Q + 1) settings.level3.harvestReadCap
          settings.level3.probeBudget,
      ∀ rounds, indecisiveLStarTrieLoop settings k O (uniformStrings L)
        (uniformStrings (L₁ + Fintype.card Q + 3)) p = some rounds →
      ((rounds.map fun o => o.prefs.card).sum : ℝ) ≤ totalPref { x with L }
        ∧ ((rounds.map fun o => o.suffs.card).sum : ℝ) ≤ totalSuff { x with L }
        ∧ (rounds.length * settings.level3.probeBudget : ℝ) ≤ totalProbes { x with L }) ∧
    (loopMeasure μ A (Fintype.card Q + 1) settings.level3.harvestReadCap
        settings.level3.probeBudget).real
      {p | ¬ RLStarSucceeds settings k O (uniformStrings L)
        (uniformStrings (L₁ + Fintype.card Q + 3)) (uniformStrings L) M x.ξ
        (totalReads { x with L }) p}
      ≤ x.δ

open Real in
/-- `RLStarClaim` at uniform strings: `p₀Tol` gives way to `k₀`. -/
def RLStarClaimUniform.{u, v, w} : Prop :=
  ∃ (k₀ totalPref totalSuff totalProbes totalReads : RLStarRun → ℝ),
    k₀ =O[𝓟 rlStarRunRegion] (fun r =>
      1 + log ((r.L + r.Q) / (r.δ * r.ξ * r.pAP * r.s₀)) / log r.A) ∧
    totalPref =O[𝓟 rlStarRunRegion] (fun r =>
      r.Q ^ 7 * r.A ^ 2 * (r.L + r.Q) ^ 2 / (r.ξ ^ 2 * r.s₀ ^ 4)
          * log (r.Q * log ((r.L + r.Q) * r.A / (r.δ * r.ξ * r.s₀)) / (r.δ * r.pAP * r.s₀))
        + r.Q ^ 2 * (r.L + r.Q) ^ 2 / r.ξ ^ 2
          * (1 / r.s₀ ^ 4 + r.Q ^ 4 * r.A ^ 2)
          * log ((r.L + r.Q) * r.A / (r.δ * r.ξ * r.pAP * r.s₀))) ∧
    totalSuff =O[𝓟 rlStarRunRegion] (fun r =>
      r.Q * (log ((r.L + r.Q) * r.A / (r.δ * r.ξ * r.s₀)) / (r.pAP * r.s₀ ^ 2)
        + log (r.Q / r.δ) / r.pAP)) ∧
    totalProbes =O[𝓟 rlStarRunRegion] (fun r =>
      r.Q ^ 3 * r.A ^ 2 * (r.L + r.Q) / r.ξ * log (r.Q * r.A / r.δ)) ∧
    totalReads =O[𝓟 rlStarRunRegion] (fun r =>
      let nPref : ℝ := r.Q ^ 7 * r.A ^ 2 * (r.L + r.Q) ^ 2 / (r.ξ ^ 2 * r.s₀ ^ 4)
          * log (r.Q * log ((r.L + r.Q) * r.A / (r.δ * r.ξ * r.s₀)) / (r.δ * r.pAP * r.s₀))
        + r.Q ^ 2 * (r.L + r.Q) ^ 2 / r.ξ ^ 2
          * (1 / r.s₀ ^ 4 + r.Q ^ 4 * r.A ^ 2)
          * log ((r.L + r.Q) * r.A / (r.δ * r.ξ * r.pAP * r.s₀))
      let nSuff : ℝ :=
        r.Q * (log ((r.L + r.Q) * r.A / (r.δ * r.ξ * r.s₀)) / (r.pAP * r.s₀ ^ 2)
          + log (r.Q / r.δ) / r.pAP)
      let nProbe : ℝ :=
        r.Q ^ 3 * r.A ^ 2 * (r.L + r.Q) / r.ξ * log (r.Q * r.A / r.δ)
      let nFam : ℝ := log ((r.L + r.Q) * r.A / (r.δ * r.ξ * r.s₀)) / r.s₀ ^ 2
      let ρInv : ℝ := r.A * (r.L + r.Q) / r.ξ
      let lg : ℝ := log (r.L + r.Q + 1)
      -- clusterings, counterexample search, harvest
      nPref * nSuff / r.Q + nProbe * nFam * r.Q * lg + r.Q * nPref * nFam * lg * ρInv) ∧
    RLStarWithinUniform.{u, v, w} k₀ totalPref totalSuff totalProbes totalReads

open Real in
def RLStarClaimUniformSimple.{u, v, w} : Prop :=
  ∃ (k₀ totalPref totalSuff totalProbes totalReads : RLStarRun → ℝ),
    k₀ =O[𝓟 rlStarRunRegion] (fun r =>
      1 + log ((r.L + r.Q) / (r.δ * r.ξ * r.pAP * r.s₀)) / log r.A) ∧
    totalPref =O[𝓟 rlStarRunRegion] (fun r =>
      r.Q ^ 7 * r.A ^ 2 * (r.L + r.Q) ^ 2
        * log ((r.L + r.Q) * r.A / (r.δ * r.ξ * r.pAP * r.s₀))
        / (r.ξ ^ 2 * r.s₀ ^ 4)) ∧
    totalSuff =O[𝓟 rlStarRunRegion] (fun r =>
      r.Q * log ((r.L + r.Q) * r.A / (r.δ * r.ξ * r.s₀)) / (r.pAP * r.s₀ ^ 2)) ∧
    totalProbes =O[𝓟 rlStarRunRegion] (fun r =>
      r.Q ^ 3 * r.A ^ 2 * (r.L + r.Q) / r.ξ * log (r.Q * r.A / r.δ)) ∧
    totalReads =O[𝓟 rlStarRunRegion] (fun r =>
      let nPref : ℝ := r.Q ^ 7 * r.A ^ 2 * (r.L + r.Q) ^ 2
        * log ((r.L + r.Q) * r.A / (r.δ * r.ξ * r.pAP * r.s₀))
        / (r.ξ ^ 2 * r.s₀ ^ 4)
      let nSuff : ℝ := r.Q * log ((r.L + r.Q) * r.A / (r.δ * r.ξ * r.s₀)) / (r.pAP * r.s₀ ^ 2)
      let nFam : ℝ := log ((r.L + r.Q) * r.A / (r.δ * r.ξ * r.s₀)) / r.s₀ ^ 2
      let ρInv : ℝ := r.A * (r.L + r.Q) / r.ξ
      let lg : ℝ := log (r.L + r.Q + 1)
      -- clusterings and harvest; the search's reads are within the harvest's
      nPref * (nSuff / r.Q + r.Q * nFam * lg * ρInv)) ∧
    RLStarWithinUniform.{u, v, w} k₀ totalPref totalSuff totalProbes totalReads

end OrthoDFA
