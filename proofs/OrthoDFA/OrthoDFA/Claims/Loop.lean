import Mathlib.Analysis.Asymptotics.Defs
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import OrthoDFA.LowLevelClaims.LoopDFA

/-!
# The learner's costs, asymptotically

`LoopClaim`: once the Python's length `L` is at least `L₀`, the loop's re-rooted DFA misclassifies
at most `γ + ξ` of the strings of length `L`, but for chance `δ`, its start leaf picked on
`heldOut` strings; over the whole run its clusterings read at most `totalPref` prefixes and
`totalSuff` suffixes, and its rounds draw at most `totalProbes` probes, whatever they draw. Every
one is within the big-O shape below, with `c` the coverage and `s₀ = 1/2 − η₀` the signal. The
totals grow with `L` because the settings the proof picks hold each round's disagreement rate at
`c/(4(|Σ| + 1)N)` over the window's `N = Θ(L)` lengths: the prefixes grow as `(L + |Q|)²` and
the suffixes as `log L`. Each bound has one constant over all of `loopRegion`.
-/

namespace OrthoDFA

open MeasureTheory Asymptotics Filter

/-- The learner's inputs: `Q` states, `A` letters, the failure chance `δ`, the coverage `c`, the
accuracy slack `ξ`, `pAP` the share of suffixes that preserve membership, and the signal
`s₀ = 1/2 − η₀`. -/
structure LoopInputs where
  Q : ℕ
  A : ℕ
  δ : ℝ
  c : ℝ
  ξ : ℝ
  pAP : ℝ
  s₀ : ℝ

def loopRegion : Set LoopInputs :=
  {x | 1 ≤ x.Q ∧ 2 ≤ x.A ∧ 0 < x.δ ∧ x.δ ≤ 1 / 2 ∧ 0 < x.c ∧ x.c ≤ 1 ∧ 0 < x.ξ ∧ x.ξ ≤ 1 ∧
    0 < x.pAP ∧ x.pAP ≤ 1 ∧ 0 < x.s₀ ∧ x.s₀ ≤ 1 / 2}

/-- The inputs with the Python's length `L`, which the totals grow with. -/
structure LoopRun extends LoopInputs where
  L : ℕ

def loopRunRegion : Set LoopRun := {r | r.toLoopInputs ∈ loopRegion}

open Real in
def LoopClaim : Prop :=
  ∃ (L₀ : LoopInputs → ℝ) (totalPref totalSuff totalProbes : LoopRun → ℝ)
    (heldOut : LoopInputs → ℕ),
    L₀ =O[𝓟 loopRegion] (fun x =>
      log (x.Q * x.A / (x.δ * x.c * x.ξ * x.pAP * x.s₀)) / log x.A) ∧
    totalPref =O[𝓟 loopRunRegion] (fun r =>
      r.Q ^ 4 * (r.L + r.Q) ^ 2 * log ((r.L + r.Q) * r.A / (r.δ * r.c * r.ξ * r.pAP * r.s₀))
        * ((r.A ^ 2 / r.c ^ 2 + 1 / r.ξ ^ 2) / r.s₀ ^ 8
          + r.Q ^ 4 * r.A ^ 2 / (r.c ^ 2 * r.s₀ ^ 4))) ∧
    totalSuff =O[𝓟 loopRunRegion] (fun r =>
      r.Q * (log ((r.L + r.Q) * r.A / (r.δ * r.c * r.ξ * r.s₀)) / (r.pAP * r.s₀ ^ 2)
        + log (r.Q / r.δ) / r.pAP ^ 2)) ∧
    totalProbes =O[𝓟 loopRunRegion] (fun r =>
      r.Q ^ 3 * r.A ^ 2 * (r.L + r.Q) / r.c * log (r.Q * r.A / r.δ)) ∧
    (fun x => (heldOut x : ℝ)) =O[𝓟 loopRegion] (fun x => log (x.Q / x.δ) / x.ξ ^ 2) ∧
    ∀ {α : Type*} [Fintype α] [DecidableEq α] [Nonempty α] {σ : Type*} [Fintype σ] {Ω : Type*}
      [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
      (x : LoopInputs) (M : _root_.DFA α σ) (O : Oracle μ (FreeMonoid α)) (L : ℕ) (γ : ℝ),
    -- Probes are walked from half the Python's length, uniform over the lengths `k` to `L₁`.
    let k : ℕ := (L + 1) / 2
    let L₁ : ℕ := L + Fintype.card σ + 3
    x ∈ loopRegion → Fintype.card σ = x.Q → Fintype.card α = x.A →
    (∀ w, w ∈ O.L ↔ M.eval w.toList ∈ M.accept) →
    O.η ≤ 1 / 2 - x.s₀ →
    x.pAP ≤ (uniformStrings (L₁ + Fintype.card σ + 3)).real {v | ∀ p, p * v ∈ O.L ↔ p ∈ O.L} →
    (∃ q ∈ coveredStates M L x.c, 1 - γ ≤ rerootAccuracy M L (coveredStates M L x.c) q) →
    L₀ x ≤ L →
    ∃ settings : LoopSettings (Fintype.card σ + 1),
      settings.level3.walkStart = k ∧
      (∀ draws : Fin (Fintype.card σ + 1)
          → LoopRoundDraws Ω α (Fintype.card σ + 1) settings.level3.probeBudget,
        let rounds := indecisiveLStarLoop settings O (windowStrings k L₁)
          (uniformStrings (L₁ + Fintype.card σ + 3)) draws
        ((rounds.map fun o => o.prefs.card).sum : ℝ) ≤ totalPref { x with L }
          ∧ ((rounds.map fun o => o.suffs.card).sum : ℝ) ≤ totalSuff { x with L }
          ∧ (rounds.length * settings.level3.probeBudget : ℝ) ≤ totalProbes { x with L }) ∧
      ((loopMeasure μ α (Fintype.card σ + 1) settings.level3.probeBudget).prod
          (Measure.pi fun _ : Fin (heldOut x) => uniformStrings L)).real
        {q | let rounds := indecisiveLStarLoop settings O (windowStrings k L₁)
              (uniformStrings (L₁ + Fintype.card σ + 3)) (persistDraws q.1)
          ¬ (rounds.length ≤ Fintype.card σ + 1 ∧ ∃ o ∈ rounds.getLast?,
              o.result.ending = .consistent ∧ (uniformStrings L).real {w | ¬
                o.result.hypothesis.rerootedDFA o.read q.2 w
                  (M.eval w.toList ∈ M.accept)} ≤ γ + x.ξ)}
        ≤ x.δ

end OrthoDFA
