import Mathlib.Analysis.Asymptotics.Defs
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import RLStar.LowLevelClaims.CoveredStart
import RLStar.Code.LearnerLoop

/-!
# The learner's costs, asymptotically

The learner is given `η₀ = 1/2 − s₀/2` and runs at accuracy `ξc = min(ξ, (L + |Q|) β)`, so its
clustering's coverage, about `ξc/L`, is within `β`, the least share either class holds of the
strings the DFA is judged on.
-/

namespace OrthoDFA

open MeasureTheory Asymptotics Filter

/-- `Q` states, `A` letters, `ξ` the accuracy slack, `pAP` the share of suffixes that preserve
membership, `s₀` a lower bound on half the gap between the classes' accept rates and `β` the
least share of either class. -/
structure RLStarInputs where
  Q : ℕ
  A : ℕ
  δ : ℝ
  ξ : ℝ
  pAP : ℝ
  s₀ : ℝ
  β : ℝ

def rlStarRegion : Set RLStarInputs :=
  {x | 1 ≤ x.Q ∧ 2 ≤ x.A ∧ 0 < x.δ ∧ x.δ ≤ 1 / 2 ∧ 0 < x.ξ ∧ x.ξ ≤ 1 ∧
    0 < x.pAP ∧ x.pAP ≤ 1 ∧ 0 < x.s₀ ∧ x.s₀ ≤ 1 / 2 ∧ 0 < x.β}

/-- `L` the Python's length and `k` the walk start. -/
structure RLStarRun extends RLStarInputs where
  L : ℕ
  k : ℕ

def rlStarRunRegion : Set RLStarRun := {r | r.toRLStarInputs ∈ rlStarRegion}

noncomputable def RLStarRun.ξc (r : RLStarRun) : ℝ := min r.ξ ((r.L + r.Q) * r.β)

def DependsOnlyOn {X Y : Type*} (f : (X → ℝ) → Y) (oracle : X → ℝ) (S : Finset X) : Prop :=
  ∀ o : X → ℝ, (∀ w ∈ S, o w = oracle w) → f o = f oracle

/-- The loop stops in agreement within `|Q| + 1` rounds, the last round's hypothesis, its walk
starts' sifts read off the table, misclassifying at most `ε` of `T` from a trie of at most
`trieSize` strings; no epoch's tree has more than `|Q| + 2` leaves; and the learner reads at most
`totalReads` strings. -/
def RLStarSucceeds {A : Type*} [Fintype A] [DecidableEq A] {Q : Type*} [Fintype Q] {Ω : Type*}
    [MeasurableSpace Ω] {μ : Measure Ω} {K : ℕ} (settings : LoopSettings K)
    (O : Oracle μ (FreeMonoid A)) (D Dsf T : Measure (FreeMonoid A)) (M : _root_.DFA A Q) (ε : ℝ)
    (trieSize totalReads : ℝ) (p : settings.Draws Ω A) : Prop :=
  (∃ rounds, indecisiveLStarLoop settings O D Dsf p = some rounds ∧
    (rounds.length ≤ Fintype.card Q + 1 ∧ ∃ o ∈ rounds.getLast?,
        o.result.ending = .consistent ∧ T.real {w | ¬ o.result.hypothesis.LazyClassifies
          ((Random.lazyTableQ settings.level3.walkStart o.result.hypothesis.tree
            o.result.expanded).eval (o.read O p.1)) o.result.expanded w
            (M.eval w.toList ∈ M.accept)} ≤ ε
        ∧ (o.result.expanded.card : ℝ) ≤ trieSize) ∧
    ∀ o ∈ rounds, o.result.peak ≤ Fintype.card Q + 2) ∧
  ∃ S : Finset (FreeMonoid A), (S.card : ℝ) ≤ totalReads ∧
    DependsOnlyOn (fun o => (learnerQ settings D Dsf p.2).eval o) (fun w => O.mq w p.1) S

/-- The trie has at most `trieWidth` strings per letter of the walk start `k`. -/
def RLStarWithin.{u, v, w}
    (p₀Tol totalPref totalSuff totalProbes trieWidth totalReads : RLStarRun → ℝ) : Prop :=
  ∀ {A : Type u} [Fintype A] [DecidableEq A] [Nonempty A] {Q : Type v}
    [Fintype Q] {Ω : Type w}
    [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
    (x : RLStarInputs) (M : _root_.DFA A Q) (O : Oracle μ (FreeMonoid A)) (L k : ℕ)
    (D : Measure (FreeMonoid A)) [IsProbabilityMeasure D],
  let L₁ : ℕ := L + Fintype.card Q + 3
  let Ls : ℕ := L₁ + Fintype.card Q + 3
  x ∈ rlStarRegion → Fintype.card Q = x.Q → Fintype.card A = x.A →
  (∀ w, w ∈ O.L ↔ M.eval w.toList ∈ M.accept) →
  2 * x.s₀ ≤ O.gap →
  x.β ≤ (prefixLaw D L).real {p | p ∈ O.L} → x.β ≤ (prefixLaw D L).real {p | p ∉ O.L} →
  (∀ᵐ w ∂D, Ls ≤ w.toList.length) →
  k ≤ L →
  (∀ u, D.real {w | prefixOf w k = u} ≤ p₀Tol { x with L, k }) →
  x.pAP ≤ (prefixLaw D Ls).real {v | ∀ p, p * v ∈ O.L ↔ p ∈ O.L} →
  ∃ settings : LoopSettings (Fintype.card Q + 1),
    settings.level3.walkStart = k ∧
    (∀ p : settings.Draws Ω A, ∀ rounds,
        indecisiveLStarLoop settings O (prefixLaw D L) (prefixLaw D Ls) p = some rounds →
      ((rounds.map fun o => o.prefs.card).sum : ℝ) ≤ totalPref { x with L, k }
        ∧ ((rounds.map fun o => o.suffs.card).sum : ℝ) ≤ totalSuff { x with L, k }
        ∧ (rounds.length * (settings.heavyDraws + settings.epochs
            * (settings.level3.probeBudget + settings.blocks * settings.blockSize)) : ℝ)
          ≤ totalProbes { x with L, k }) ∧
    (settings.measure μ A).real
      {p | ¬ RLStarSucceeds settings O (prefixLaw D L) (prefixLaw D Ls) (prefixLaw D L) M
        x.ξ (k * trieWidth { x with L, k }) (totalReads { x with L, k }) p}
      ≤ x.δ

open Real in
/-- `log ((L + |Q|) |Σ| / (δ ξc pAP s₀))`. -/
noncomputable def RLStarRun.logZ (r : RLStarRun) : ℝ :=
  log ((r.L + r.Q) * r.A / (r.δ * r.ξc * r.pAP * r.s₀))

/-- The inverse of the tolerance `ι` of a family's undecided rate on a population. -/
noncomputable def RLStarRun.ιInv (r : RLStarRun) : ℝ := r.Q ^ 2 * r.A * (r.L + r.Q) / r.ξc

/-- The inverse of the cap on a population's atoms. -/
noncomputable def RLStarRun.nCap (r : RLStarRun) : ℝ :=
  r.Q * r.logZ * r.ιInv ^ 3 * (1 / r.s₀ ^ 6 + r.Q ^ 2 / r.s₀ ^ 4)
    + r.Q * r.logZ / (r.pAP ^ 2 * r.s₀ ^ 2)

noncomputable def RLStarRun.nPref (r : RLStarRun) : ℝ := r.Q ^ 3 * r.logZ * r.ιInv ^ 2 / r.s₀ ^ 4

noncomputable def RLStarRun.nSuff (r : RLStarRun) : ℝ := r.Q * r.logZ / (r.pAP * r.s₀ ^ 2)

/-- A round's trie draws, `logZ / θ` at the light-prefix level `θ`, the atom cap times the harvest
rate. -/
noncomputable def RLStarRun.nHeavy (r : RLStarRun) : ℝ :=
  r.A * (r.L + r.Q) * r.logZ * r.nCap / r.ξc

noncomputable def RLStarRun.nProbe (r : RLStarRun) : ℝ := r.Q * r.nHeavy

/-- Clusterings, then the walk-start table: `|Q| + 2` sifts at `|V̂| = logZ / s₀²` for each of
the trie's `k |Σ| nHeavy` frontier strings; the samplers read as much as `logZ` letters of `k`. -/
noncomputable def RLStarRun.nReads (r : RLStarRun) : ℝ :=
  r.nPref * r.nSuff / r.Q + r.Q * r.A * (r.k + r.logZ) * r.nHeavy * (r.logZ / r.s₀ ^ 2)

def RLStarClaim.{u, v, w} : Prop :=
  ∃ (p₀Tol totalPref totalSuff totalProbes trieWidth totalReads : RLStarRun → ℝ),
    (fun r => 1 / p₀Tol r) =O[𝓟 rlStarRunRegion]
      (fun r => r.A * (r.L + r.Q) * r.nCap / r.ξc) ∧
    totalPref =O[𝓟 rlStarRunRegion] RLStarRun.nPref ∧
    totalSuff =O[𝓟 rlStarRunRegion] RLStarRun.nSuff ∧
    totalProbes =O[𝓟 rlStarRunRegion] RLStarRun.nProbe ∧
    trieWidth =O[𝓟 rlStarRunRegion] RLStarRun.nHeavy ∧
    totalReads =O[𝓟 rlStarRunRegion] RLStarRun.nReads ∧
    (∀ r, 0 < p₀Tol r) ∧
    RLStarWithin.{u, v, w} p₀Tol totalPref totalSuff totalProbes trieWidth totalReads

def RLStarWithinUniform.{u, v, w} (k₀ : RLStarRun → ℝ)
    (totalPref totalSuff totalProbes trieWidth totalReads : RLStarRun → ℝ) : Prop :=
  ∀ {A : Type u} [Fintype A] [DecidableEq A] [Nonempty A] {Q : Type v}
    [Fintype Q] {Ω : Type w}
    [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
    (x : RLStarInputs) (M : _root_.DFA A Q) (O : Oracle μ (FreeMonoid A)) (L k : ℕ),
  let L₁ : ℕ := L + Fintype.card Q + 3
  x ∈ rlStarRegion → Fintype.card Q = x.Q → Fintype.card A = x.A →
  (∀ w, w ∈ O.L ↔ M.eval w.toList ∈ M.accept) →
  2 * x.s₀ ≤ O.gap →
  x.β ≤ (uniformStrings L).real {p | p ∈ O.L} → x.β ≤ (uniformStrings L).real {p | p ∉ O.L} →
  x.pAP ≤ (uniformStrings (L₁ + Fintype.card Q + 3)).real {v | ∀ p, p * v ∈ O.L ↔ p ∈ O.L} →
  k₀ { x with L, k } ≤ k → k ≤ L →
  ∃ settings : LoopSettings (Fintype.card Q + 1),
    settings.level3.walkStart = k ∧
    (∀ p : settings.Draws Ω A, ∀ rounds, indecisiveLStarLoop settings O (uniformStrings L)
        (uniformStrings (L₁ + Fintype.card Q + 3)) p = some rounds →
      ((rounds.map fun o => o.prefs.card).sum : ℝ) ≤ totalPref { x with L, k }
        ∧ ((rounds.map fun o => o.suffs.card).sum : ℝ) ≤ totalSuff { x with L, k }
        ∧ (rounds.length * (settings.heavyDraws + settings.epochs
            * (settings.level3.probeBudget + settings.blocks * settings.blockSize)) : ℝ)
          ≤ totalProbes { x with L, k }) ∧
    (settings.measure μ A).real
      {p | ¬ RLStarSucceeds settings O (uniformStrings L)
        (uniformStrings (L₁ + Fintype.card Q + 3)) (uniformStrings L) M x.ξ
        (k * trieWidth { x with L, k }) (totalReads { x with L, k }) p}
      ≤ x.δ

open Real in
/-- `RLStarClaim` at uniform strings: `p₀Tol` gives way to `k₀`. -/
def RLStarClaimUniform.{u, v, w} : Prop :=
  ∃ (k₀ totalPref totalSuff totalProbes trieWidth totalReads : RLStarRun → ℝ),
    k₀ =O[𝓟 rlStarRunRegion] (fun r =>
      1 + log ((r.L + r.Q) * log (1 / r.δ) / (r.ξc * r.pAP * r.s₀)) / log r.A) ∧
    totalPref =O[𝓟 rlStarRunRegion] RLStarRun.nPref ∧
    totalSuff =O[𝓟 rlStarRunRegion] RLStarRun.nSuff ∧
    totalProbes =O[𝓟 rlStarRunRegion] RLStarRun.nProbe ∧
    trieWidth =O[𝓟 rlStarRunRegion] RLStarRun.nHeavy ∧
    totalReads =O[𝓟 rlStarRunRegion] RLStarRun.nReads ∧
    RLStarWithinUniform.{u, v, w} k₀ totalPref totalSuff totalProbes trieWidth totalReads

open Real in
def RLStarClaimUniformSimple.{u, v, w} : Prop :=
  ∃ (k₀ totalPref totalSuff totalProbes trieWidth totalReads : RLStarRun → ℝ),
    k₀ =O[𝓟 rlStarRunRegion] (fun r =>
      1 + log ((r.L + r.Q) * log (1 / r.δ) / (r.ξc * r.pAP * r.s₀)) / log r.A) ∧
    totalPref =O[𝓟 rlStarRunRegion] (fun r =>
      r.Q ^ 7 * r.A ^ 2 * (r.L + r.Q) ^ 2 * r.logZ / (r.ξc ^ 2 * r.s₀ ^ 4)) ∧
    totalSuff =O[𝓟 rlStarRunRegion] (fun r => r.Q * r.logZ / (r.pAP * r.s₀ ^ 2)) ∧
    totalProbes =O[𝓟 rlStarRunRegion] (fun r =>
      r.Q ^ 8 * r.A ^ 4 * (r.L + r.Q) ^ 4 * r.logZ ^ 2 / (r.ξc ^ 4 * r.s₀ ^ 6)
        + r.Q ^ 10 * r.A ^ 4 * (r.L + r.Q) ^ 4 * r.logZ ^ 2 / (r.ξc ^ 4 * r.s₀ ^ 4)
        + r.Q ^ 2 * r.A * (r.L + r.Q) * r.logZ ^ 2 / (r.ξc * r.pAP ^ 2 * r.s₀ ^ 2)) ∧
    trieWidth =O[𝓟 rlStarRunRegion] (fun r =>
      r.Q ^ 7 * r.A ^ 4 * (r.L + r.Q) ^ 4 * r.logZ ^ 2 / (r.ξc ^ 4 * r.s₀ ^ 6)
        + r.Q ^ 9 * r.A ^ 4 * (r.L + r.Q) ^ 4 * r.logZ ^ 2 / (r.ξc ^ 4 * r.s₀ ^ 4)
        + r.Q * r.A * (r.L + r.Q) * r.logZ ^ 2 / (r.ξc * r.pAP ^ 2 * r.s₀ ^ 2)) ∧
    totalReads =O[𝓟 rlStarRunRegion] (fun r =>
      r.Q ^ 7 * r.A ^ 2 * (r.L + r.Q) ^ 2 * r.logZ ^ 2 / (r.ξc ^ 2 * r.pAP * r.s₀ ^ 6)
        + (r.k + r.logZ) * (r.Q ^ 8 * r.A ^ 5 * (r.L + r.Q) ^ 4 * r.logZ ^ 3 / (r.ξc ^ 4 * r.s₀ ^ 8)
          + r.Q ^ 10 * r.A ^ 5 * (r.L + r.Q) ^ 4 * r.logZ ^ 3 / (r.ξc ^ 4 * r.s₀ ^ 6)
          + r.Q ^ 2 * r.A ^ 2 * (r.L + r.Q) * r.logZ ^ 3 / (r.ξc * r.pAP ^ 2 * r.s₀ ^ 4))) ∧
    RLStarWithinUniform.{u, v, w} k₀ totalPref totalSuff totalProbes trieWidth totalReads

end OrthoDFA
