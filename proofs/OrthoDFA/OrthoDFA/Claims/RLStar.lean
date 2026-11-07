import Mathlib.Analysis.Asymptotics.Defs
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import OrthoDFA.LowLevelClaims.LoopDFA

/-!
# The learner's costs, asymptotically

`RLStarClaim`: over an input distribution `D`, with `β` the least rate at which `D` extends a prefix
by any one letter, the loop's re-rooted DFA misclassifies at most `γ + ξ` of `D`'s strings cut at
length `L`, but for chance `δ`, its start leaf picked on `heldOut` strings, when the target admits a
start by the relative rule at visit ratio `R` and `D` holds at most `p₀Tol` on any first `k`
letters, `1/p₀Tol` growing as `L⁴ log L`, and every round's tree has at most `|Q| + 2` leaves
(`RLStarSucceeds`); over the whole run its clusterings read at most `totalPref` prefixes and
`totalSuff` suffixes, its rounds draw at most `totalProbes` probes, whatever they draw;
and, but for chance `δ`, the run's output depends on at most `totalReads` of the oracle's answers,
start leaf included.
Every one is within the big-O shape below, with `s₀ = 1/2 − η₀` the signal. The settings the proof
picks hold each round's disagreement rate at `m/(3(1/β + |Q| + 1)N)`, `m = 3ξ/(8R)`, over the
window's `N = Θ(L)` lengths, so its inverse enters the shapes as `R(1/β + |Q|)N/ξ`: the prefixes
grow as `(L + |Q|)²` and the suffixes as `log L`. The prefix total's populations' group reads its
log off the suffix pool alone, so that log grows as `log log L`. The strings asked are four sources'
sum: the clusterings' prefixes against their suffixes; level 3's probes, each a family read of
`|V̂|` strings at the `log L` mids of a sift at most `|Q| + 1` deep, with its splits' re-sifts (at
most `|Q|` a round, each over at most `|Q| + 2` leaves' edges, `|Σ|` letters and `|Q| + 2`
midfixes) within that term; the harvested draws' walks, a family read at the sifts of each probe,
about `1/ρ` probes a draw taken; and the start leaf's family read at each held-out string. Each
bound has one constant over all of `rlStarDistRegion`.

`RLStarClaimUniform` is it at `D` uniform, where `β = 1/|Σ|` and `|Σ|^-k ≤ p₀Tol` once the length
`L` is at least `L₀`; the bounds have one constant over all of `rlStarRegion`.
`RLStarClaimUniformSimple` states the same with the prefix and suffix totals each a single product.
-/

namespace OrthoDFA

open MeasureTheory Asymptotics Filter

/-- The learner's inputs: `Q` states, `A` letters, the failure chance `δ`, the visit ratio `R` of
`nonStartZoneStates`, the accuracy slack `ξ`, `pAP` the share of suffixes that preserve
membership, and the signal `s₀ = 1/2 − η₀`. -/
structure RLStarInputs where
  Q : ℕ
  A : ℕ
  δ : ℝ
  R : ℝ
  ξ : ℝ
  pAP : ℝ
  s₀ : ℝ

/-- A ratio `R < 1` asks more of the target than `R = 1`: fewer non-start-zone states. -/
def rlStarRegion : Set RLStarInputs :=
  {x | 1 ≤ x.Q ∧ 2 ≤ x.A ∧ 0 < x.δ ∧ x.δ ≤ 1 / 2 ∧ 1 ≤ x.R ∧ 0 < x.ξ ∧ x.ξ ≤ 1 ∧
    0 < x.pAP ∧ x.pAP ≤ 1 ∧ 0 < x.s₀ ∧ x.s₀ ≤ 1 / 2}

/-- The inputs with the Python's length `L`, which the totals grow with. -/
structure RLStarRun extends RLStarInputs where
  L : ℕ

def rlStarRunRegion : Set RLStarRun := {r | r.toRLStarInputs ∈ rlStarRegion}

/-- `RLStarInputs` with `β`, the least rate at which the input distribution extends a prefix by any
one letter: `1/|Σ|` for uniform strings. -/
structure RLStarDistInputs extends RLStarInputs where
  β : ℝ

def rlStarDistRegion : Set RLStarDistInputs :=
  {x | x.toRLStarInputs ∈ rlStarRegion ∧ 0 < x.β ∧ x.β * x.A ≤ 1}

structure RLStarDistRun extends RLStarDistInputs where
  L : ℕ

def rlStarDistRunRegion : Set RLStarDistRun := {r | r.toRLStarDistInputs ∈ rlStarDistRegion}

/-- `f`'s value at `oracle` depends only on `oracle`'s answers on `S`. -/
def DependsOnlyOn {X Y : Type*} (f : (X → ℝ) → Y) (oracle : X → ℝ) (S : Finset X) : Prop :=
  ∀ o : X → ℝ, (∀ w ∈ S, o w = oracle w) → f o = f oracle

/-- Within the round and leaf caps, the loop ends consistent and errs on at most `ε` of `T`, and
its output depends on at most `totalReads` of the oracle's answers. -/
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

/-- On every instance in `rlStarDistRegion`, for strings drawn from `D` whose first `k` letters
`p₀Tol` bounds, settings whose run reads within `totalPref`, `totalSuff` and `totalProbes`, and
which but for chance `δ` succeed within `totalReads`. -/
def RLStarWithin.{u, v, w} (p₀Tol totalPref totalSuff totalProbes totalReads : RLStarDistRun → ℝ)
    (heldOut : RLStarDistInputs → ℕ) : Prop :=
  ∀ {A : Type u} [Fintype A] [DecidableEq A] [Nonempty A] {Q : Type v}
    [Fintype Q] {Ω : Type w}
    [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
    (x : RLStarDistInputs) (M : _root_.DFA A Q) (O : Oracle μ (FreeMonoid A)) (L : ℕ) (γ : ℝ)
    (D : Measure (FreeMonoid A)) [IsProbabilityMeasure D],
  -- Probes are `D`'s strings cut at a length uniform over `k` to `L₁`, suffixes their first `Ls`
  -- letters, and the coverage, error and held-out strings are over their first `L`.
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
/-- The learner's costs over an input distribution `D`: `p₀Tol` bounds the most `D` may hold on any
first `k` letters, and `β` is the rate edges are crossed at. -/
def RLStarClaim.{u, v, w} : Prop :=
  ∃ (p₀Tol totalPref totalSuff totalProbes totalReads : RLStarDistRun → ℝ)
    (heldOut : RLStarDistInputs → ℕ),
    (fun r => 1 / p₀Tol r) =O[𝓟 rlStarDistRunRegion] (fun r =>
      (r.L + r.Q) ^ 4 * log ((r.L + r.Q) * r.R / (r.β * r.δ * r.ξ * r.pAP * r.s₀))
          * log (r.Q * r.R / (r.β * r.δ * r.ξ * r.pAP * r.s₀)) * r.Q ^ 3 * r.R * (1 / r.β + r.Q)
          / r.ξ
          * (r.R ^ 3 * r.Q ^ 2 * (1 / r.β + r.Q) ^ 2 / r.ξ ^ 3
              * (1 / r.s₀ ^ 6 + r.Q ^ 2 * (1 / r.β + r.Q) / r.s₀ ^ 5
                + r.Q ^ 4 * (1 / r.β + r.Q) / r.s₀ ^ 4)
            + 1 / (r.δ * r.pAP ^ 2 * r.s₀ ^ 2))
        + (r.L + r.Q) ^ 2 * log ((r.L + r.Q) * r.R / (r.β * r.δ * r.ξ * r.pAP * r.s₀))
          * log (r.Q * r.R / (r.β * r.δ * r.ξ * r.pAP * r.s₀)) ^ 2 * r.Q ^ 6 * (1 / r.β + r.Q) ^ 2
          * r.R ^ 2 / (r.ξ ^ 4 * r.δ * r.s₀ ^ 4)
        + (r.L + r.Q) ^ 2 * log (r.Q * r.R / (r.β * r.δ * r.ξ * r.pAP * r.s₀)) ^ 2
          * r.Q ^ 5 * r.A ^ 2 * (1 / r.β + r.Q) ^ 2 * r.R ^ 2 / (r.ξ ^ 2 * r.δ)) ∧
    totalPref =O[𝓟 rlStarDistRunRegion] (fun r =>
      r.Q ^ 7 * (1 / r.β + r.Q) ^ 2 * (r.L + r.Q) ^ 2 * r.R ^ 2 / (r.ξ ^ 2 * r.s₀ ^ 4)
          * log (r.Q * log ((r.L + r.Q) * r.R / (r.β * r.δ * r.ξ * r.s₀)) / (r.δ * r.pAP * r.s₀))
        + r.Q ^ 2 * (r.L + r.Q) ^ 2 * r.R ^ 2 / r.ξ ^ 2
          * (1 / r.s₀ ^ 4 + r.Q ^ 4 * (1 / r.β + r.Q) ^ 2)
          * log ((r.L + r.Q) * r.R / (r.β * r.δ * r.ξ * r.pAP * r.s₀))) ∧
    totalSuff =O[𝓟 rlStarDistRunRegion] (fun r =>
      r.Q * (log ((r.L + r.Q) * r.R / (r.β * r.δ * r.ξ * r.s₀)) / (r.pAP * r.s₀ ^ 2)
        + log (r.Q / r.δ) / r.pAP)) ∧
    totalProbes =O[𝓟 rlStarDistRunRegion] (fun r =>
      r.Q ^ 3 * r.A * (1 / r.β + r.Q) * (r.L + r.Q) * r.R / r.ξ * log (r.Q * r.A / r.δ)) ∧
    totalReads =O[𝓟 rlStarDistRunRegion] (fun r =>
      let nPref : ℝ :=
        r.Q ^ 7 * (1 / r.β + r.Q) ^ 2 * (r.L + r.Q) ^ 2 * r.R ^ 2 / (r.ξ ^ 2 * r.s₀ ^ 4)
            * log (r.Q * log ((r.L + r.Q) * r.R / (r.β * r.δ * r.ξ * r.s₀)) / (r.δ * r.pAP * r.s₀))
          + r.Q ^ 2 * (r.L + r.Q) ^ 2 * r.R ^ 2 / r.ξ ^ 2
            * (1 / r.s₀ ^ 4 + r.Q ^ 4 * (1 / r.β + r.Q) ^ 2)
            * log ((r.L + r.Q) * r.R / (r.β * r.δ * r.ξ * r.pAP * r.s₀))
      let nSuff : ℝ :=
        r.Q * (log ((r.L + r.Q) * r.R / (r.β * r.δ * r.ξ * r.s₀)) / (r.pAP * r.s₀ ^ 2)
          + log (r.Q / r.δ) / r.pAP)
      let nProbe : ℝ :=
        r.Q ^ 3 * r.A * (1 / r.β + r.Q) * (r.L + r.Q) * r.R / r.ξ * log (r.Q * r.A / r.δ)
      let nHeld : ℝ := log (r.Q / r.δ) / r.ξ ^ 2
      -- `|V̂|`, a round's suffix family
      let nFam : ℝ := log ((r.L + r.Q) * r.R / (r.β * r.δ * r.ξ * r.s₀)) / r.s₀ ^ 2
      let ρInv : ℝ := r.R * (1 / r.β + r.Q) * (r.L + r.Q) / r.ξ
      let lg : ℝ := log (r.L + r.Q + 1)
      -- clusterings, counterexample search (with its re-sifts), harvest, start leaf
      nPref * nSuff / r.Q + nProbe * nFam * r.Q * lg + r.Q * nPref * nFam * lg * ρInv
        + nHeld * nFam) ∧
    (fun x => (heldOut x : ℝ)) =O[𝓟 rlStarDistRegion] (fun x => log (x.Q / x.δ) / x.ξ ^ 2) ∧
    (∀ r, 0 < p₀Tol r) ∧
    RLStarWithin.{u, v, w} p₀Tol totalPref totalSuff totalProbes totalReads heldOut

/-- On every instance in `rlStarRegion` with `L₀ x ≤ L`, settings whose run reads within
`totalPref`, `totalSuff` and `totalProbes`, and which but for chance `δ` succeed within
`totalReads`. -/
def RLStarWithinUniform.{u, v, w} (L₀ : RLStarInputs → ℝ)
    (totalPref totalSuff totalProbes totalReads : RLStarRun → ℝ) (heldOut : RLStarInputs → ℕ) :
    Prop :=
  ∀ {A : Type u} [Fintype A] [DecidableEq A] [Nonempty A] {Q : Type v}
    [Fintype Q] {Ω : Type w}
    [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
    (x : RLStarInputs) (M : _root_.DFA A Q) (O : Oracle μ (FreeMonoid A)) (L : ℕ) (γ : ℝ),
  -- Probes are walked from half the Python's length, uniform over the lengths `k` to `L₁`.
  let k : ℕ := (L + 1) / 2
  let L₁ : ℕ := L + Fintype.card Q + 3
  x ∈ rlStarRegion → Fintype.card Q = x.Q → Fintype.card A = x.A →
  (∀ w, w ∈ O.L ↔ M.eval w.toList ∈ M.accept) →
  O.η ≤ 1 / 2 - x.s₀ →
  x.pAP ≤ (uniformStrings (L₁ + Fintype.card Q + 3)).real {v | ∀ p, p * v ∈ O.L ↔ p ∈ O.L} →
  (∃ q, q ∈ nonStartZoneStates M L x.R q ∧
    1 - γ ≤ rerootAccuracy M L (nonStartZoneStates M L x.R q) q) →
  L₀ x ≤ L →
  ∃ settings : LoopSettings (Fintype.card Q + 1),
    settings.level3.walkStart = k ∧
    (∀ p : LoopDraws Ω A (Fintype.card Q + 1) settings.level3.harvestReadCap
          settings.level3.probeBudget,
      ∀ rounds, indecisiveLStarLoop settings O (windowStrings k L₁)
        (uniformStrings (L₁ + Fintype.card Q + 3)) p = some rounds →
      ((rounds.map fun o => o.prefs.card).sum : ℝ) ≤ totalPref { x with L }
        ∧ ((rounds.map fun o => o.suffs.card).sum : ℝ) ≤ totalSuff { x with L }
        ∧ (rounds.length * settings.level3.probeBudget : ℝ) ≤ totalProbes { x with L }) ∧
    ((loopMeasure μ A (Fintype.card Q + 1) settings.level3.harvestReadCap
        settings.level3.probeBudget).prod
        (Measure.pi fun _ : Fin (heldOut x) => uniformStrings L)).real
      {q | ¬ RLStarSucceeds settings O (windowStrings k L₁)
        (uniformStrings (L₁ + Fintype.card Q + 3)) (uniformStrings L) M (γ + x.ξ)
        (totalReads { x with L }) q}
      ≤ x.δ

open Real in
/-- `RLStarClaim` at uniform strings: `L₀` gives way to `p₀Tol`, and `1/|Σ|` to `β`. -/
def RLStarClaimUniform.{u, v, w} : Prop :=
  ∃ (L₀ : RLStarInputs → ℝ) (totalPref totalSuff totalProbes totalReads : RLStarRun → ℝ)
    (heldOut : RLStarInputs → ℕ),
    L₀ =O[𝓟 rlStarRegion] (fun x =>
      1 + log (x.Q * x.R / (x.δ * x.ξ * x.pAP * x.s₀)) / log x.A) ∧
    totalPref =O[𝓟 rlStarRunRegion] (fun r =>
      r.Q ^ 7 * (r.A + r.Q) ^ 2 * (r.L + r.Q) ^ 2 * r.R ^ 2 / (r.ξ ^ 2 * r.s₀ ^ 4)
          * log (r.Q * log ((r.L + r.Q) * r.A * r.R / (r.δ * r.ξ * r.s₀)) / (r.δ * r.pAP * r.s₀))
        + r.Q ^ 2 * (r.L + r.Q) ^ 2 * r.R ^ 2 / r.ξ ^ 2
          * (1 / r.s₀ ^ 4 + r.Q ^ 4 * (r.A + r.Q) ^ 2)
          * log ((r.L + r.Q) * r.A * r.R / (r.δ * r.ξ * r.pAP * r.s₀))) ∧
    totalSuff =O[𝓟 rlStarRunRegion] (fun r =>
      r.Q * (log ((r.L + r.Q) * r.A * r.R / (r.δ * r.ξ * r.s₀)) / (r.pAP * r.s₀ ^ 2)
        + log (r.Q / r.δ) / r.pAP)) ∧
    totalProbes =O[𝓟 rlStarRunRegion] (fun r =>
      r.Q ^ 3 * r.A * (r.A + r.Q) * (r.L + r.Q) * r.R / r.ξ * log (r.Q * r.A / r.δ)) ∧
    totalReads =O[𝓟 rlStarRunRegion] (fun r =>
      let nPref : ℝ := r.Q ^ 7 * (r.A + r.Q) ^ 2 * (r.L + r.Q) ^ 2 * r.R ^ 2 / (r.ξ ^ 2 * r.s₀ ^ 4)
          * log (r.Q * log ((r.L + r.Q) * r.A * r.R / (r.δ * r.ξ * r.s₀)) / (r.δ * r.pAP * r.s₀))
        + r.Q ^ 2 * (r.L + r.Q) ^ 2 * r.R ^ 2 / r.ξ ^ 2
          * (1 / r.s₀ ^ 4 + r.Q ^ 4 * (r.A + r.Q) ^ 2)
          * log ((r.L + r.Q) * r.A * r.R / (r.δ * r.ξ * r.pAP * r.s₀))
      let nSuff : ℝ :=
        r.Q * (log ((r.L + r.Q) * r.A * r.R / (r.δ * r.ξ * r.s₀)) / (r.pAP * r.s₀ ^ 2)
          + log (r.Q / r.δ) / r.pAP)
      let nProbe : ℝ :=
        r.Q ^ 3 * r.A * (r.A + r.Q) * (r.L + r.Q) * r.R / r.ξ * log (r.Q * r.A / r.δ)
      let nHeld : ℝ := log (r.Q / r.δ) / r.ξ ^ 2
      let nFam : ℝ := log ((r.L + r.Q) * r.A * r.R / (r.δ * r.ξ * r.s₀)) / r.s₀ ^ 2
      let ρInv : ℝ := r.R * (r.A + r.Q) * (r.L + r.Q) / r.ξ
      let lg : ℝ := log (r.L + r.Q + 1)
      -- clusterings, counterexample search (with its re-sifts), harvest, start leaf
      nPref * nSuff / r.Q + nProbe * nFam * r.Q * lg + r.Q * nPref * nFam * lg * ρInv
        + nHeld * nFam) ∧
    (fun x => (heldOut x : ℝ)) =O[𝓟 rlStarRegion] (fun x => log (x.Q / x.δ) / x.ξ ^ 2) ∧
    RLStarWithinUniform.{u, v, w} L₀ totalPref totalSuff totalProbes totalReads heldOut

open Real in
/-- The uniform claim with the prefix and suffix totals each a single product. -/
def RLStarClaimUniformSimple.{u, v, w} : Prop :=
  ∃ (L₀ : RLStarInputs → ℝ) (totalPref totalSuff totalProbes totalReads : RLStarRun → ℝ)
    (heldOut : RLStarInputs → ℕ),
    L₀ =O[𝓟 rlStarRegion] (fun x =>
      1 + log (x.Q * x.R / (x.δ * x.ξ * x.pAP * x.s₀)) / log x.A) ∧
    totalPref =O[𝓟 rlStarRunRegion] (fun r =>
      r.Q ^ 7 * (r.A + r.Q) ^ 2 * (r.L + r.Q) ^ 2 * r.R ^ 2
        * log ((r.L + r.Q) * r.A * r.R / (r.δ * r.ξ * r.pAP * r.s₀))
        / (r.ξ ^ 2 * r.s₀ ^ 4)) ∧
    totalSuff =O[𝓟 rlStarRunRegion] (fun r =>
      r.Q * log ((r.L + r.Q) * r.A * r.R / (r.δ * r.ξ * r.s₀)) / (r.pAP * r.s₀ ^ 2)) ∧
    totalProbes =O[𝓟 rlStarRunRegion] (fun r =>
      r.Q ^ 3 * r.A * (r.A + r.Q) * (r.L + r.Q) * r.R / r.ξ * log (r.Q * r.A / r.δ)) ∧
    totalReads =O[𝓟 rlStarRunRegion] (fun r =>
      let nPref : ℝ := r.Q ^ 7 * (r.A + r.Q) ^ 2 * (r.L + r.Q) ^ 2 * r.R ^ 2
        * log ((r.L + r.Q) * r.A * r.R / (r.δ * r.ξ * r.pAP * r.s₀))
        / (r.ξ ^ 2 * r.s₀ ^ 4)
      let nSuff : ℝ := r.Q * log ((r.L + r.Q) * r.A * r.R / (r.δ * r.ξ * r.s₀)) / (r.pAP * r.s₀ ^ 2)
      let nProbe : ℝ :=
        r.Q ^ 3 * r.A * (r.A + r.Q) * (r.L + r.Q) * r.R / r.ξ * log (r.Q * r.A / r.δ)
      let nHeld : ℝ := log (r.Q / r.δ) / r.ξ ^ 2
      let nFam : ℝ := log ((r.L + r.Q) * r.A * r.R / (r.δ * r.ξ * r.s₀)) / r.s₀ ^ 2
      let ρInv : ℝ := r.R * (r.A + r.Q) * (r.L + r.Q) / r.ξ
      let lg : ℝ := log (r.L + r.Q + 1)
      -- clusterings, counterexample search (with its re-sifts), harvest, start leaf
      nPref * nSuff / r.Q + nProbe * nFam * r.Q * lg + r.Q * nPref * nFam * lg * ρInv
        + nHeld * nFam) ∧
    (fun x => (heldOut x : ℝ)) =O[𝓟 rlStarRegion] (fun x => log (x.Q / x.δ) / x.ξ ^ 2) ∧
    RLStarWithinUniform.{u, v, w} L₀ totalPref totalSuff totalProbes totalReads heldOut

end OrthoDFA
