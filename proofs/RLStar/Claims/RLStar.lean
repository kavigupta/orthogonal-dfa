import Mathlib.Analysis.Asymptotics.Defs
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import RLStar.LowLevelClaims.LoopDFA
import RLStar.Code.LearnerLoopTrie

/-!
# The learner's costs, asymptotically
-/

namespace OrthoDFA

open MeasureTheory Asymptotics Filter

/-- `Q` states, `A` letters, `R` the visit ratio of `nonStartZoneStates`, `ξ` the accuracy slack,
`pAP` the share of suffixes that preserve membership, and the signal `s₀ = 1/2 − η₀`. -/
structure RLStarInputs where
  Q : ℕ
  A : ℕ
  δ : ℝ
  R : ℝ
  ξ : ℝ
  pAP : ℝ
  s₀ : ℝ

/-- A target admitted at `R < 1` is admitted at `R = 1`. -/
def rlStarRegion : Set RLStarInputs :=
  {x | 1 ≤ x.Q ∧ 2 ≤ x.A ∧ 0 < x.δ ∧ x.δ ≤ 1 / 2 ∧ 1 ≤ x.R ∧ 0 < x.ξ ∧ x.ξ ≤ 1 ∧
    0 < x.pAP ∧ x.pAP ≤ 1 ∧ 0 < x.s₀ ∧ x.s₀ ≤ 1 / 2}

/-- `β` the least rate at which the input distribution extends a prefix by any one letter. -/
structure RLStarDistInputs extends RLStarInputs where
  β : ℝ

def rlStarDistRegion : Set RLStarDistInputs :=
  {x | x.toRLStarInputs ∈ rlStarRegion ∧ 0 < x.β ∧ x.β * x.A ≤ 1}

structure RLStarDistRun extends RLStarDistInputs where
  L : ℕ

def rlStarDistRunRegion : Set RLStarDistRun := {r | r.toRLStarDistInputs ∈ rlStarDistRegion}

def DependsOnlyOn {X Y : Type*} (f : (X → ℝ) → Y) (oracle : X → ℝ) (S : Finset X) : Prop :=
  ∀ o : X → ℝ, (∀ w ∈ S, o w = oracle w) → f o = f oracle

def RLStarSucceeds {A : Type*} [Fintype A] [DecidableEq A] {Q : Type*} [Fintype Q] {Ω : Type*}
    [MeasurableSpace Ω] {μ : Measure Ω} {K : ℕ} (settings : LoopSettings K)
    (O : Oracle μ (FreeMonoid A)) (D Dsf T : Measure (FreeMonoid A)) (M : _root_.DFA A Q) (ε : ℝ)
    (totalReads : ℝ) {n : ℕ}
    (q : LoopDraws Ω A K settings.level3.harvestReadCap settings.level3.probeBudget
      × (Fin n → FreeMonoid A)) : Prop :=
  (∃ rounds, indecisiveLStarLoop settings O D Dsf q.1 = some rounds ∧
    (rounds.length ≤ Fintype.card Q + 1 ∧ ∃ o ∈ rounds.getLast?,
        o.result.ending = .consistent ∧ T.real {w | ¬
          o.result.hypothesis.rerootedDFA (o.read O q.1.1) q.2 w
            (M.eval w.toList ∈ M.accept)} ≤ ε) ∧
    ∀ o ∈ rounds, o.result.hypothesis.tree.leaves.length ≤ Fintype.card Q + 2) ∧
  ∃ S : Finset (FreeMonoid A), (S.card : ℝ) ≤ totalReads ∧
    DependsOnlyOn (fun o => (learnerQ settings D Dsf q.1.2 q.2).eval o) (fun w => O.mq w q.1.1) S

def RLStarWithin.{u, v, w} (p₀Tol totalPref totalSuff totalProbes totalReads : RLStarDistRun → ℝ)
    (heldOut : RLStarDistInputs → ℕ) : Prop :=
  ∀ {A : Type u} [Fintype A] [DecidableEq A] [Nonempty A] {Q : Type v}
    [Fintype Q] {Ω : Type w}
    [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
    (x : RLStarDistInputs) (M : _root_.DFA A Q) (O : Oracle μ (FreeMonoid A)) (L : ℕ) (γ : ℝ)
    (D : Measure (FreeMonoid A)) [IsProbabilityMeasure D],
  let k : ℕ := (L + 1) / 2
  let L₁ : ℕ := L + Fintype.card Q + 3
  let Ls : ℕ := L₁ + Fintype.card Q + 3
  x ∈ rlStarDistRegion → Fintype.card Q = x.Q → Fintype.card A = x.A →
  (∀ w, w ∈ O.L ↔ M.eval w.toList ∈ M.accept) →
  O.η ≤ 1 / 2 - x.s₀ →
  (∀ᵐ w ∂D, Ls ≤ w.toList.length) →
  (∀ u, D.real {w | prefixOf w k = u} ≤ p₀Tol { x with L }) →
  (∀ i, k ≤ i → i ≤ L → ∀ y a,
    x.β * D.real {w | prefixOf w i = y} ≤ D.real {w | prefixOf w (i + 1) = y * FreeMonoid.of a}) →
  x.pAP ≤ (prefixLaw D Ls).real {v | ∀ p, p * v ∈ O.L ↔ p ∈ O.L} →
  (∃ q, q ∈ nonStartZoneStatesOn (prefixLaw D L) M L x.R q ∧
    1 - γ ≤ rerootAccuracyOn (prefixLaw D L) M L (nonStartZoneStatesOn (prefixLaw D L) M L x.R q)
      q) →
  ∃ settings : LoopSettings (Fintype.card Q + 1),
    settings.level3.walkStart = k ∧
    (∀ p : LoopDraws Ω A (Fintype.card Q + 1) settings.level3.harvestReadCap
          settings.level3.probeBudget,
      ∀ rounds,
        indecisiveLStarLoop settings O (windowLaw D k L₁) (prefixLaw D Ls) p = some rounds →
      ((rounds.map fun o => o.prefs.card).sum : ℝ) ≤ totalPref { x with L }
        ∧ ((rounds.map fun o => o.suffs.card).sum : ℝ) ≤ totalSuff { x with L }
        ∧ (rounds.length * settings.level3.probeBudget : ℝ) ≤ totalProbes { x with L }) ∧
    ((loopMeasure μ A (Fintype.card Q + 1) settings.level3.harvestReadCap
        settings.level3.probeBudget).prod
        (Measure.pi fun _ : Fin (heldOut x) => prefixLaw D L)).real
      {q | ¬ RLStarSucceeds settings O (windowLaw D k L₁) (prefixLaw D Ls) (prefixLaw D L) M
        (γ + x.ξ) (totalReads { x with L }) q}
      ≤ x.δ

open Real in
def RLStarClaim.{u, v, w} : Prop :=
  ∃ (p₀Tol totalPref totalSuff totalProbes totalReads : RLStarDistRun → ℝ)
    (heldOut : RLStarDistInputs → ℕ),
    (fun r => 1 / p₀Tol r) =O[𝓟 rlStarDistRunRegion] (fun r =>
      (r.L + r.Q) ^ 4 * log ((r.L + r.Q) * r.R / (r.β * r.δ * r.ξ * r.pAP * r.s₀))
          * log (r.Q * r.R / (r.β * r.δ * r.ξ * r.pAP * r.s₀)) * r.Q ^ 3 * r.R * (1 / r.β)
          / r.ξ
          * (r.R ^ 3 * r.Q ^ 2 * (1 / r.β) ^ 2 / r.ξ ^ 3
              * (1 / r.s₀ ^ 6 + r.Q ^ 2 * (1 / r.β) / r.s₀ ^ 5
                + r.Q ^ 4 * (1 / r.β) / r.s₀ ^ 4)
            + 1 / (r.δ * r.pAP ^ 2 * r.s₀ ^ 2))
        + (r.L + r.Q) ^ 2 * log ((r.L + r.Q) * r.R / (r.β * r.δ * r.ξ * r.pAP * r.s₀))
          * log (r.Q * r.R / (r.β * r.δ * r.ξ * r.pAP * r.s₀)) ^ 2 * r.Q ^ 6 * (1 / r.β) ^ 2
          * r.R ^ 2 / (r.ξ ^ 4 * r.δ * r.s₀ ^ 4)
        + (r.L + r.Q) ^ 2 * log (r.Q * r.R / (r.β * r.δ * r.ξ * r.pAP * r.s₀)) ^ 2
          * r.Q ^ 5 * r.A ^ 2 * (1 / r.β) ^ 2 * r.R ^ 2 / (r.ξ ^ 2 * r.δ)) ∧
    totalPref =O[𝓟 rlStarDistRunRegion] (fun r =>
      r.Q ^ 7 * (1 / r.β) ^ 2 * (r.L + r.Q) ^ 2 * r.R ^ 2 / (r.ξ ^ 2 * r.s₀ ^ 4)
          * log (r.Q * log ((r.L + r.Q) * r.R / (r.β * r.δ * r.ξ * r.s₀)) / (r.δ * r.pAP * r.s₀))
        + r.Q ^ 2 * (r.L + r.Q) ^ 2 * r.R ^ 2 / r.ξ ^ 2
          * (1 / r.s₀ ^ 4 + r.Q ^ 4 * (1 / r.β) ^ 2)
          * log ((r.L + r.Q) * r.R / (r.β * r.δ * r.ξ * r.pAP * r.s₀))) ∧
    totalSuff =O[𝓟 rlStarDistRunRegion] (fun r =>
      r.Q * (log ((r.L + r.Q) * r.R / (r.β * r.δ * r.ξ * r.s₀)) / (r.pAP * r.s₀ ^ 2)
        + log (r.Q / r.δ) / r.pAP)) ∧
    totalProbes =O[𝓟 rlStarDistRunRegion] (fun r =>
      r.Q ^ 3 * r.A * (1 / r.β) * (r.L + r.Q) * r.R / r.ξ * log (r.Q * r.A / r.δ)) ∧
    totalReads =O[𝓟 rlStarDistRunRegion] (fun r =>
      let nPref : ℝ :=
        r.Q ^ 7 * (1 / r.β) ^ 2 * (r.L + r.Q) ^ 2 * r.R ^ 2 / (r.ξ ^ 2 * r.s₀ ^ 4)
            * log (r.Q * log ((r.L + r.Q) * r.R / (r.β * r.δ * r.ξ * r.s₀)) / (r.δ * r.pAP * r.s₀))
          + r.Q ^ 2 * (r.L + r.Q) ^ 2 * r.R ^ 2 / r.ξ ^ 2
            * (1 / r.s₀ ^ 4 + r.Q ^ 4 * (1 / r.β) ^ 2)
            * log ((r.L + r.Q) * r.R / (r.β * r.δ * r.ξ * r.pAP * r.s₀))
      let nSuff : ℝ :=
        r.Q * (log ((r.L + r.Q) * r.R / (r.β * r.δ * r.ξ * r.s₀)) / (r.pAP * r.s₀ ^ 2)
          + log (r.Q / r.δ) / r.pAP)
      let nProbe : ℝ :=
        r.Q ^ 3 * r.A * (1 / r.β) * (r.L + r.Q) * r.R / r.ξ * log (r.Q * r.A / r.δ)
      let nHeld : ℝ := log (r.Q / r.δ) / r.ξ ^ 2
      -- `|V̂|`, a round's suffix family
      let nFam : ℝ := log ((r.L + r.Q) * r.R / (r.β * r.δ * r.ξ * r.s₀)) / r.s₀ ^ 2
      let ρInv : ℝ := r.R * (1 / r.β) * (r.L + r.Q) / r.ξ
      let lg : ℝ := log (r.L + r.Q + 1)
      -- clusterings, counterexample search, harvest, start leaf
      nPref * nSuff / r.Q + nProbe * nFam * r.Q * lg + r.Q * nPref * nFam * lg * ρInv
        + nHeld * nFam) ∧
    (fun x => (heldOut x : ℝ)) =O[𝓟 rlStarDistRegion] (fun x => log (x.Q / x.δ) / x.ξ ^ 2) ∧
    (∀ r, 0 < p₀Tol r) ∧
    RLStarWithin.{u, v, w} p₀Tol totalPref totalSuff totalProbes totalReads heldOut

/-- `Q` states, `A` letters, `ξ` the accuracy slack, `pAP` the share of suffixes that preserve
membership, and the signal `s₀ = 1/2 − η₀`. -/
structure RLStarUniformInputs where
  Q : ℕ
  A : ℕ
  δ : ℝ
  ξ : ℝ
  pAP : ℝ
  s₀ : ℝ

def rlStarUniformRegion : Set RLStarUniformInputs :=
  {x | 1 ≤ x.Q ∧ 2 ≤ x.A ∧ 0 < x.δ ∧ x.δ ≤ 1 / 2 ∧ 0 < x.ξ ∧ x.ξ ≤ 1 ∧
    0 < x.pAP ∧ x.pAP ≤ 1 ∧ 0 < x.s₀ ∧ x.s₀ ≤ 1 / 2}

/-- `L` the Python's length. -/
structure RLStarUniformRun extends RLStarUniformInputs where
  L : ℕ

def rlStarUniformRunRegion : Set RLStarUniformRun :=
  {r | r.toRLStarUniformInputs ∈ rlStarUniformRegion}

/-- Within `|Q| + 1` rounds the loop ends consistent and its DFA misjudges at most `ξ` of `T`'s
strings, reading at most `totalReads` strings. -/
def RLStarTrieSucceeds {A : Type*} [Fintype A] [DecidableEq A] {Q : Type*} [Fintype Q] {Ω : Type*}
    [MeasurableSpace Ω] {μ : Measure Ω} {K : ℕ} (settings : LoopSettings K) (cap : ℕ)
    (O : Oracle μ (FreeMonoid A)) (D Dsf T : Measure (FreeMonoid A)) (M : _root_.DFA A Q) (ξ : ℝ)
    (totalReads : ℝ)
    (p : LoopDraws Ω A K settings.level3.harvestReadCap settings.level3.probeBudget) : Prop :=
  (∃ rounds, indecisiveLStarTrieLoop settings cap O D Dsf p = some rounds ∧
    rounds.length ≤ Fintype.card Q + 1 ∧ ∃ o ∈ rounds.getLast?,
      o.result.ending = .consistent ∧ T.real {w | ¬ o.result.hypothesis.TrieClassifies
        (o.read O p.1) o.result.expanded w (M.eval w.toList ∈ M.accept)} ≤ ξ) ∧
  ∃ S : Finset (FreeMonoid A), (S.card : ℝ) ≤ totalReads ∧
    DependsOnlyOn (fun o => (trieLoopRoundsQ settings cap D Dsf p.2 K).eval o)
      (fun w => O.mq w p.1) S

def RLStarWithinUniform.{u, v, w} (L₀ : RLStarUniformInputs → ℝ)
    (totalPref totalSuff totalProbes totalReads : RLStarUniformRun → ℝ) : Prop :=
  ∀ {A : Type u} [Fintype A] [DecidableEq A] [Nonempty A] {Q : Type v} [Fintype Q] {Ω : Type w}
    [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
    (x : RLStarUniformInputs) (M : _root_.DFA A Q) (O : Oracle μ (FreeMonoid A)) (L : ℕ),
  let Ls : ℕ := L + 2 * Fintype.card Q + 6
  x ∈ rlStarUniformRegion → Fintype.card Q = x.Q → Fintype.card A = x.A →
  (∀ w, w ∈ O.L ↔ M.eval w.toList ∈ M.accept) →
  O.η ≤ 1 / 2 - x.s₀ →
  x.pAP ≤ (uniformStrings Ls).real {v | ∀ p, p * v ∈ O.L ↔ p ∈ O.L} →
  L₀ x ≤ L →
  ∃ settings : LoopSettings (Fintype.card Q + 1),
    (∀ p : LoopDraws Ω A (Fintype.card Q + 1) settings.level3.harvestReadCap
          settings.level3.probeBudget,
      ∀ rounds, indecisiveLStarTrieLoop settings ((L + 1) / 2) O (uniformStrings L)
          (uniformStrings Ls) p = some rounds →
        ((rounds.map fun o => o.prefs.card).sum : ℝ) ≤ totalPref { x with L }
          ∧ ((rounds.map fun o => o.suffs.card).sum : ℝ) ≤ totalSuff { x with L }
          ∧ (rounds.length * settings.level3.probeBudget : ℝ) ≤ totalProbes { x with L }) ∧
    (loopMeasure μ A (Fintype.card Q + 1) settings.level3.harvestReadCap
        settings.level3.probeBudget).real
      {p | ¬ RLStarTrieSucceeds settings ((L + 1) / 2) O (uniformStrings L) (uniformStrings Ls)
        (uniformStrings L) M x.ξ (totalReads { x with L }) p}
      ≤ x.δ

open Real in
/-- Over uniform strings, the loop with local walk starts expanded below `⌈L/2⌉` letters, with no
start premise: `RLStarClaim`'s bounds at `R = 1` and `β = 1/|Σ|`, once `L ≥ L₀`. -/
def RLStarClaimUniform.{u, v, w} : Prop :=
  ∃ (L₀ : RLStarUniformInputs → ℝ)
    (totalPref totalSuff totalProbes totalReads : RLStarUniformRun → ℝ),
    L₀ =O[𝓟 rlStarUniformRegion] (fun x =>
      1 + log (x.Q / (x.δ * x.ξ * x.pAP * x.s₀)) / log x.A) ∧
    totalPref =O[𝓟 rlStarUniformRunRegion] (fun r =>
      r.Q ^ 7 * r.A ^ 2 * (r.L + r.Q) ^ 2 / (r.ξ ^ 2 * r.s₀ ^ 4)
          * log (r.Q * log ((r.L + r.Q) * r.A / (r.δ * r.ξ * r.s₀)) / (r.δ * r.pAP * r.s₀))
        + r.Q ^ 2 * (r.L + r.Q) ^ 2 / r.ξ ^ 2
          * (1 / r.s₀ ^ 4 + r.Q ^ 4 * r.A ^ 2)
          * log ((r.L + r.Q) * r.A / (r.δ * r.ξ * r.pAP * r.s₀))) ∧
    totalSuff =O[𝓟 rlStarUniformRunRegion] (fun r =>
      r.Q * (log ((r.L + r.Q) * r.A / (r.δ * r.ξ * r.s₀)) / (r.pAP * r.s₀ ^ 2)
        + log (r.Q / r.δ) / r.pAP)) ∧
    totalProbes =O[𝓟 rlStarUniformRunRegion] (fun r =>
      r.Q ^ 3 * r.A ^ 2 * (r.L + r.Q) / r.ξ * log (r.Q * r.A / r.δ)) ∧
    totalReads =O[𝓟 rlStarUniformRunRegion] (fun r =>
      let nPref : ℝ := r.Q ^ 7 * r.A ^ 2 * (r.L + r.Q) ^ 2 / (r.ξ ^ 2 * r.s₀ ^ 4)
          * log (r.Q * log ((r.L + r.Q) * r.A / (r.δ * r.ξ * r.s₀)) / (r.δ * r.pAP * r.s₀))
        + r.Q ^ 2 * (r.L + r.Q) ^ 2 / r.ξ ^ 2
          * (1 / r.s₀ ^ 4 + r.Q ^ 4 * r.A ^ 2)
          * log ((r.L + r.Q) * r.A / (r.δ * r.ξ * r.pAP * r.s₀))
      let nSuff : ℝ :=
        r.Q * (log ((r.L + r.Q) * r.A / (r.δ * r.ξ * r.s₀)) / (r.pAP * r.s₀ ^ 2)
          + log (r.Q / r.δ) / r.pAP)
      let nProbe : ℝ := r.Q ^ 3 * r.A ^ 2 * (r.L + r.Q) / r.ξ * log (r.Q * r.A / r.δ)
      let nHeld : ℝ := log (r.Q / r.δ) / r.ξ ^ 2
      let nFam : ℝ := log ((r.L + r.Q) * r.A / (r.δ * r.ξ * r.s₀)) / r.s₀ ^ 2
      let ρInv : ℝ := r.A * (r.L + r.Q) / r.ξ
      let lg : ℝ := log (r.L + r.Q + 1)
      nPref * nSuff / r.Q + nProbe * nFam * r.Q * lg + r.Q * nPref * nFam * lg * ρInv
        + nHeld * nFam) ∧
    RLStarWithinUniform.{u, v, w} L₀ totalPref totalSuff totalProbes totalReads

open Real in
/-- `RLStarClaimUniform` with the prefix and suffix totals each a single product. -/
def RLStarClaimUniformSimple.{u, v, w} : Prop :=
  ∃ (L₀ : RLStarUniformInputs → ℝ)
    (totalPref totalSuff totalProbes totalReads : RLStarUniformRun → ℝ),
    L₀ =O[𝓟 rlStarUniformRegion] (fun x =>
      1 + log (x.Q / (x.δ * x.ξ * x.pAP * x.s₀)) / log x.A) ∧
    totalPref =O[𝓟 rlStarUniformRunRegion] (fun r =>
      r.Q ^ 7 * r.A ^ 2 * (r.L + r.Q) ^ 2
        * log ((r.L + r.Q) * r.A / (r.δ * r.ξ * r.pAP * r.s₀)) / (r.ξ ^ 2 * r.s₀ ^ 4)) ∧
    totalSuff =O[𝓟 rlStarUniformRunRegion] (fun r =>
      r.Q * log ((r.L + r.Q) * r.A / (r.δ * r.ξ * r.s₀)) / (r.pAP * r.s₀ ^ 2)) ∧
    totalProbes =O[𝓟 rlStarUniformRunRegion] (fun r =>
      r.Q ^ 3 * r.A ^ 2 * (r.L + r.Q) / r.ξ * log (r.Q * r.A / r.δ)) ∧
    totalReads =O[𝓟 rlStarUniformRunRegion] (fun r =>
      let nPref : ℝ := r.Q ^ 7 * r.A ^ 2 * (r.L + r.Q) ^ 2
        * log ((r.L + r.Q) * r.A / (r.δ * r.ξ * r.pAP * r.s₀)) / (r.ξ ^ 2 * r.s₀ ^ 4)
      let nSuff : ℝ := r.Q * log ((r.L + r.Q) * r.A / (r.δ * r.ξ * r.s₀)) / (r.pAP * r.s₀ ^ 2)
      let nFam : ℝ := log ((r.L + r.Q) * r.A / (r.δ * r.ξ * r.s₀)) / r.s₀ ^ 2
      let ρInv : ℝ := r.A * (r.L + r.Q) / r.ξ
      let lg : ℝ := log (r.L + r.Q + 1)
      -- clusterings and harvest; the search's and start leaf's reads are within the harvest's
      nPref * (nSuff / r.Q + r.Q * nFam * lg * ρInv)) ∧
    RLStarWithinUniform.{u, v, w} L₀ totalPref totalSuff totalProbes totalReads

end OrthoDFA
