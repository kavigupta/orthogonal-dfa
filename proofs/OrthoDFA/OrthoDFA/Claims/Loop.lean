import Mathlib.Analysis.Asymptotics.Defs
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import OrthoDFA.LowLevelClaims.LoopDFA

/-!
# The learner's costs, asymptotically

`LoopClaim`: once the Python's length `L` is at least `L₀`, the loop's re-rooted DFA misclassifies
at most `γ + ξ` of the strings of length `L`, but for chance `δ`, its start leaf picked on
`heldOut` strings, when the target admits a start by the relative rule at visit ratio `R`; over the
whole run its clusterings read at most `totalPref` prefixes and `totalSuff` suffixes, and its
rounds draw at most `totalProbes` probes, whatever they draw. Every one is within the big-O shape
below, with `s₀ = 1/2 − η₀` the signal. The run is the coverage rule's at `c = ξ/(2C|Q|)`, with
`R ≤ C ≤ 2R`, so `1/c` enters the shapes as `R|Q|/ξ`. The totals grow with `L` because the
settings the proof picks hold each round's disagreement rate at `c/(4(|Σ| + 1)N)` over the
window's `N = Θ(L)` lengths: the prefixes grow as `(L + |Q|)²` and the suffixes as `log L`. Each
bound has one constant over all of `loopRegion`. `LoopClaimSimple` states the same with the prefix
and suffix totals each a single product.

`LoopDistClaim` states the same over an input distribution `D`, with `β` the least rate at which
`D` extends a prefix by any one letter in place of `1/|Σ|`, and in place of `L ≥ L₀` the condition
that `D` hold at most `p₀Tol` on any first `k` letters, `1/p₀Tol` polynomial in `L`. `LoopClaim` is
it at `D` uniform, where `|Σ|^-k ≤ p₀Tol` once `L ≥ L₀`.
-/

namespace OrthoDFA

open MeasureTheory Asymptotics Filter

/-- The learner's inputs: `Q` states, `A` letters, the failure chance `δ`, the visit ratio `R` of
`nonStartZoneStates`, the accuracy slack `ξ`, `pAP` the share of suffixes that preserve
membership, and the signal `s₀ = 1/2 − η₀`. -/
structure LoopInputs where
  Q : ℕ
  A : ℕ
  δ : ℝ
  R : ℝ
  ξ : ℝ
  pAP : ℝ
  s₀ : ℝ

/-- A ratio `R < 1` asks more of the target than `R = 1`: fewer non-start-zone states, and a
higher coverage. -/
def loopRegion : Set LoopInputs :=
  {x | 1 ≤ x.Q ∧ 2 ≤ x.A ∧ 0 < x.δ ∧ x.δ ≤ 1 / 2 ∧ 1 ≤ x.R ∧ 0 < x.ξ ∧ x.ξ ≤ 1 ∧
    0 < x.pAP ∧ x.pAP ≤ 1 ∧ 0 < x.s₀ ∧ x.s₀ ≤ 1 / 2}

/-- The inputs with the Python's length `L`, which the totals grow with. -/
structure LoopRun extends LoopInputs where
  L : ℕ

def loopRunRegion : Set LoopRun := {r | r.toLoopInputs ∈ loopRegion}

/-- On every instance in `loopRegion` with `L₀ x ≤ L`, settings whose run reads within
`totalPref`, `totalSuff` and `totalProbes` and is right but for `γ + ξ` on `heldOut` strings. -/
def LoopWithin.{u, v, w} (L₀ : LoopInputs → ℝ) (totalPref totalSuff totalProbes : LoopRun → ℝ)
    (heldOut : LoopInputs → ℕ) : Prop :=
  ∀ {A : Type u} [Fintype A] [DecidableEq A] [Nonempty A] {Q : Type v}
    [Fintype Q] {Ω : Type w}
    [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
    (x : LoopInputs) (M : _root_.DFA A Q) (O : Oracle μ (FreeMonoid A)) (L : ℕ) (γ : ℝ),
  -- Probes are walked from half the Python's length, uniform over the lengths `k` to `L₁`.
  let k : ℕ := (L + 1) / 2
  let L₁ : ℕ := L + Fintype.card Q + 3
  x ∈ loopRegion → Fintype.card Q = x.Q → Fintype.card A = x.A →
  (∀ w, w ∈ O.L ↔ M.eval w.toList ∈ M.accept) →
  O.η ≤ 1 / 2 - x.s₀ →
  x.pAP ≤ (uniformStrings (L₁ + Fintype.card Q + 3)).real {v | ∀ p, p * v ∈ O.L ↔ p ∈ O.L} →
  (∃ q, q ∈ nonStartZoneStates M L x.R q ∧
    1 - γ ≤ rerootAccuracy M L (nonStartZoneStates M L x.R q) q) →
  L₀ x ≤ L →
  ∃ settings : LoopSettings (Fintype.card Q + 1),
    settings.level3.walkStart = k ∧
    (∀ draws : Fin (Fintype.card Q + 1)
        → LoopRoundDraws Ω A (Fintype.card Q + 1) settings.level3.probeBudget,
      let rounds := indecisiveLStarLoop settings O (windowStrings k L₁)
        (uniformStrings (L₁ + Fintype.card Q + 3)) draws
      ((rounds.map fun o => o.prefs.card).sum : ℝ) ≤ totalPref { x with L }
        ∧ ((rounds.map fun o => o.suffs.card).sum : ℝ) ≤ totalSuff { x with L }
        ∧ (rounds.length * settings.level3.probeBudget : ℝ) ≤ totalProbes { x with L }) ∧
    ((loopMeasure μ A (Fintype.card Q + 1) settings.level3.probeBudget).prod
        (Measure.pi fun _ : Fin (heldOut x) => uniformStrings L)).real
      {q | let rounds := indecisiveLStarLoop settings O (windowStrings k L₁)
            (uniformStrings (L₁ + Fintype.card Q + 3)) (persistDraws q.1)
        ¬ (rounds.length ≤ Fintype.card Q + 1 ∧ ∃ o ∈ rounds.getLast?,
            o.result.ending = .consistent ∧ (uniformStrings L).real {w | ¬
              o.result.hypothesis.rerootedDFA o.read q.2 w
                (M.eval w.toList ∈ M.accept)} ≤ γ + x.ξ)}
      ≤ x.δ

open Real in
def LoopClaim.{u, v, w} : Prop :=
  ∃ (L₀ : LoopInputs → ℝ) (totalPref totalSuff totalProbes : LoopRun → ℝ)
    (heldOut : LoopInputs → ℕ),
    L₀ =O[𝓟 loopRegion] (fun x =>
      1 + log (x.Q * x.R / (x.δ * x.ξ * x.pAP * x.s₀)) / log x.A) ∧
    totalPref =O[𝓟 loopRunRegion] (fun r =>
      r.Q ^ 6 * (r.L + r.Q) ^ 2 * r.A ^ 2 * r.R ^ 2
        * log ((r.L + r.Q) * r.A * r.R / (r.δ * r.ξ * r.pAP * r.s₀)) / r.ξ ^ 2
        * (1 / r.s₀ ^ 8 + r.Q ^ 4 / r.s₀ ^ 4)) ∧
    totalSuff =O[𝓟 loopRunRegion] (fun r =>
      r.Q * (log ((r.L + r.Q) * r.A * r.R / (r.δ * r.ξ * r.s₀)) / (r.pAP * r.s₀ ^ 2)
        + log (r.Q / r.δ) / r.pAP ^ 2)) ∧
    totalProbes =O[𝓟 loopRunRegion] (fun r =>
      r.Q ^ 4 * r.A ^ 2 * (r.L + r.Q) * r.R / r.ξ * log (r.Q * r.A / r.δ)) ∧
    (fun x => (heldOut x : ℝ)) =O[𝓟 loopRegion] (fun x => log (x.Q / x.δ) / x.ξ ^ 2) ∧
    LoopWithin.{u, v, w} L₀ totalPref totalSuff totalProbes heldOut

open Real in
/-- `LoopClaim` with the prefix and suffix totals each a single product. -/
def LoopClaimSimple.{u, v, w} : Prop :=
  ∃ (L₀ : LoopInputs → ℝ) (totalPref totalSuff totalProbes : LoopRun → ℝ)
    (heldOut : LoopInputs → ℕ),
    L₀ =O[𝓟 loopRegion] (fun x =>
      1 + log (x.Q * x.R / (x.δ * x.ξ * x.pAP * x.s₀)) / log x.A) ∧
    totalPref =O[𝓟 loopRunRegion] (fun r =>
      r.Q ^ 10 * (r.L + r.Q) ^ 2 * r.A ^ 2 * r.R ^ 2
        * log ((r.L + r.Q) * r.A * r.R / (r.δ * r.ξ * r.pAP * r.s₀))
        / (r.ξ ^ 2 * r.s₀ ^ 8)) ∧
    totalSuff =O[𝓟 loopRunRegion] (fun r =>
      r.Q * log ((r.L + r.Q) * r.A * r.R / (r.δ * r.ξ * r.pAP * r.s₀)) / (r.pAP ^ 2 * r.s₀ ^ 2)) ∧
    totalProbes =O[𝓟 loopRunRegion] (fun r =>
      r.Q ^ 4 * r.A ^ 2 * (r.L + r.Q) * r.R / r.ξ * log (r.Q * r.A / r.δ)) ∧
    (fun x => (heldOut x : ℝ)) =O[𝓟 loopRegion] (fun x => log (x.Q / x.δ) / x.ξ ^ 2) ∧
    LoopWithin.{u, v, w} L₀ totalPref totalSuff totalProbes heldOut

/-- `LoopInputs` with `β`, the least rate at which the input distribution extends a prefix by any
one letter: `1/|Σ|` for uniform strings. -/
structure LoopDistInputs extends LoopInputs where
  β : ℝ

def loopDistRegion : Set LoopDistInputs :=
  {x | x.toLoopInputs ∈ loopRegion ∧ 0 < x.β ∧ x.β * x.A ≤ 1}

structure LoopDistRun extends LoopDistInputs where
  L : ℕ

def loopDistRunRegion : Set LoopDistRun := {r | r.toLoopDistInputs ∈ loopDistRegion}

/-- On every instance in `loopDistRegion`, for strings drawn from `D` whose first `k` letters
`p₀Tol` bounds, settings whose run reads within `totalPref`, `totalSuff` and `totalProbes` and is
right but for `γ + ξ` on `heldOut` strings. -/
def LoopDistWithin.{u, v, w} (p₀Tol totalPref totalSuff totalProbes : LoopDistRun → ℝ)
    (heldOut : LoopDistInputs → ℕ) : Prop :=
  ∀ {A : Type u} [Fintype A] [DecidableEq A] [Nonempty A] {Q : Type v}
    [Fintype Q] {Ω : Type w}
    [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
    (x : LoopDistInputs) (M : _root_.DFA A Q) (O : Oracle μ (FreeMonoid A)) (L : ℕ) (γ : ℝ)
    (D : Measure (FreeMonoid A)) [IsProbabilityMeasure D],
  -- Probes are `D`'s strings cut at a length uniform over `k` to `L₁`, suffixes their first `Ls`
  -- letters, and the coverage, error and held-out strings are over their first `L`.
  let k : ℕ := (L + 1) / 2
  let L₁ : ℕ := L + Fintype.card Q + 3
  let Ls : ℕ := L₁ + Fintype.card Q + 3
  x ∈ loopDistRegion → Fintype.card Q = x.Q → Fintype.card A = x.A →
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
    (∀ draws : Fin (Fintype.card Q + 1)
        → LoopRoundDraws Ω A (Fintype.card Q + 1) settings.level3.probeBudget,
      let rounds := indecisiveLStarLoop settings O (windowLaw D k L₁) (prefixLaw D Ls) draws
      ((rounds.map fun o => o.prefs.card).sum : ℝ) ≤ totalPref { x with L }
        ∧ ((rounds.map fun o => o.suffs.card).sum : ℝ) ≤ totalSuff { x with L }
        ∧ (rounds.length * settings.level3.probeBudget : ℝ) ≤ totalProbes { x with L }) ∧
    ((loopMeasure μ A (Fintype.card Q + 1) settings.level3.probeBudget).prod
        (Measure.pi fun _ : Fin (heldOut x) => prefixLaw D L)).real
      {q | let rounds := indecisiveLStarLoop settings O (windowLaw D k L₁) (prefixLaw D Ls)
            (persistDraws q.1)
        ¬ (rounds.length ≤ Fintype.card Q + 1 ∧ ∃ o ∈ rounds.getLast?,
            o.result.ending = .consistent ∧ (prefixLaw D L).real {w | ¬
              o.result.hypothesis.rerootedDFA o.read q.2 w
                (M.eval w.toList ∈ M.accept)} ≤ γ + x.ξ)}
      ≤ x.δ

open Real in
/-- `LoopClaim` over an input distribution: its `L₀` gives way to `p₀Tol`, the most `D` may hold
on any first `k` letters, and `β` takes `1/|Σ|`'s place in the rate edges are crossed at. -/
def LoopDistClaim.{u, v, w} : Prop :=
  ∃ (p₀Tol totalPref totalSuff totalProbes : LoopDistRun → ℝ) (heldOut : LoopDistInputs → ℕ),
    (fun r => 1 / p₀Tol r) =O[𝓟 loopDistRunRegion] (fun r =>
      (r.L + r.Q) ^ 6 * r.Q ^ 14 * r.R ^ 4
        * log (r.Q * r.R / (r.β * r.δ * r.ξ * r.pAP * r.s₀)) ^ 2
        / (r.β ^ 4 * r.δ * r.ξ ^ 4 * r.pAP ^ 3 * r.s₀ ^ 11)) ∧
    totalPref =O[𝓟 loopDistRunRegion] (fun r =>
      r.Q ^ 6 * (r.L + r.Q) ^ 2 * r.R ^ 2
        * log ((r.L + r.Q) * r.R / (r.β * r.δ * r.ξ * r.pAP * r.s₀)) / (r.β ^ 2 * r.ξ ^ 2)
        * (1 / r.s₀ ^ 8 + r.Q ^ 4 / r.s₀ ^ 4)) ∧
    totalSuff =O[𝓟 loopDistRunRegion] (fun r =>
      r.Q * (log ((r.L + r.Q) * r.R / (r.β * r.δ * r.ξ * r.s₀)) / (r.pAP * r.s₀ ^ 2)
        + log (r.Q / r.δ) / r.pAP ^ 2)) ∧
    totalProbes =O[𝓟 loopDistRunRegion] (fun r =>
      r.Q ^ 4 * r.A * (r.L + r.Q) * r.R / (r.β * r.ξ) * log (r.Q * r.A / r.δ)) ∧
    (fun x => (heldOut x : ℝ)) =O[𝓟 loopDistRegion] (fun x => log (x.Q / x.δ) / x.ξ ^ 2) ∧
    (∀ r, 0 < p₀Tol r) ∧
    LoopDistWithin.{u, v, w} p₀Tol totalPref totalSuff totalProbes heldOut

end OrthoDFA
