import Mathlib.Analysis.Asymptotics.Defs
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import OrthoDFA.LoopDFA

/-!
# The learner's costs, asymptotically

`LoopClaim`: once the Python's length `L` is at least `L₀`, the loop's re-rooted DFA misclassifies
at most `γ + ξ` of the strings of length `L`, but for chance `δ`, its start leaf picked on
`heldOut` strings; over the whole run its clusterings read at most `totalPref` prefixes and
`totalSuff` suffixes, and its rounds draw at most `totalProbes` probes, whatever they draw. Every
one is within the big-O shape below, with `c` the coverage and `s₀ = 1/2 − η₀` the signal. The
totals grow with `L` because the settings the proof picks hold each round's disagreement rate at
`c/(4(|Σ| + 1)N)` over the window's `N = Θ(L)` lengths: the prefixes grow as `(L + |Q|)²` and
the suffixes as `log L`. Each bound has one constant over all of `loopRegion` (with `L ≥ 0` for the
totals), where every instance's vector lies.
-/

namespace OrthoDFA

open MeasureTheory Asymptotics Filter

/-- `(|Q|, |Σ|, 1/δ, 1/c, 1/ξ, 1/pAP, 1/s₀)`. -/
abbrev LoopVec := ℝ × ℝ × ℝ × ℝ × ℝ × ℝ × ℝ

def loopRegion : Set LoopVec := Set.Ici (1, 2, 2, 1, 1, 1, 2)

open Real in
def LoopClaim : Prop :=
  ∃ (L₀ : LoopVec → ℝ) (totalPref totalSuff totalProbes : ℝ × LoopVec → ℝ)
    (heldOut : LoopVec → ℕ),
    L₀ =O[𝓟 loopRegion] (fun (Q, A, iδ, ic, iξ, ip, is) =>
      log (Q * A * iδ * ic * iξ * ip * is) / log A) ∧
    totalPref =O[𝓟 (Set.Ici 0 ×ˢ loopRegion)] (fun (L, Q, A, iδ, ic, iξ, ip, is) =>
      Q ^ 4 * (L + Q) ^ 2 * log ((L + Q) * A * iδ * ic * iξ * ip * is)
        * ((A ^ 2 * ic ^ 2 + iξ ^ 2) * is ^ 8 + Q ^ 4 * A ^ 2 * ic ^ 2 * is ^ 4)) ∧
    totalSuff =O[𝓟 (Set.Ici 0 ×ˢ loopRegion)] (fun (L, Q, A, iδ, ic, iξ, ip, is) =>
      Q * (log ((L + Q) * A * iδ * ic * iξ * is) * ip * is ^ 2 + log (Q * iδ) * ip ^ 2)) ∧
    totalProbes =O[𝓟 (Set.Ici 0 ×ˢ loopRegion)] (fun (L, Q, A, iδ, ic, _, _, _) =>
      Q ^ 3 * A ^ 2 * (L + Q) * ic * log (Q * A * iδ)) ∧
    (fun v => (heldOut v : ℝ)) =O[𝓟 loopRegion] (fun (Q, _, iδ, _, iξ, _, _) =>
      log (Q * iδ) * iξ ^ 2) ∧
    ∀ {α : Type*} [Fintype α] [DecidableEq α] [Nonempty α] {σ : Type*} [Fintype σ] {Ω : Type*}
      [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
      (M : _root_.DFA α σ) (O : Oracle μ (FreeMonoid α)) (L : ℕ) (η₀ c γ δ pAP ξ : ℝ),
    let v : LoopVec :=
      (Fintype.card σ, Fintype.card α, 1 / δ, 1 / c, 1 / ξ, 1 / pAP, 1 / (1 / 2 - η₀))
    -- Probes are walked from half the Python's length, uniform over the lengths `k` to `L₁`.
    let k : ℕ := (L + 1) / 2
    let L₁ : ℕ := L + Fintype.card σ + 3
    (∀ w, w ∈ O.L ↔ M.eval w.toList ∈ M.accept) →
    O.η ≤ η₀ → 0 ≤ η₀ → η₀ < 1 / 2 → 0 < c → c ≤ 1 → 0 < δ → δ ≤ 1 / 2 → 0 < pAP → pAP ≤ 1 →
    0 < ξ → ξ ≤ 1 → 2 ≤ Fintype.card α →
    pAP ≤ (uniformStrings (L₁ + Fintype.card σ + 3)).real {v | ∀ p, p * v ∈ O.L ↔ p ∈ O.L} →
    (∃ q ∈ coveredStates M L c, 1 - γ ≤ rerootAccuracy M L (coveredStates M L c) q) →
    L₀ v ≤ L →
    v ∈ loopRegion ∧ ∃ settings : LoopSettings (Fintype.card σ + 1),
      settings.level3.walkStart = k ∧
      (∀ draws : Fin (Fintype.card σ + 1)
          → LoopRoundDraws Ω α (Fintype.card σ + 1) settings.level3.probeBudget,
        let rounds := indecisiveLStarLoop settings O (windowStrings k L₁)
          (uniformStrings (L₁ + Fintype.card σ + 3)) draws
        ((rounds.map fun o => o.prefs.card).sum : ℝ) ≤ totalPref (L, v)
          ∧ ((rounds.map fun o => o.suffs.card).sum : ℝ) ≤ totalSuff (L, v)
          ∧ (rounds.length * settings.level3.probeBudget : ℝ) ≤ totalProbes (L, v)) ∧
      ((loopMeasure μ α (Fintype.card σ + 1) settings.level3.probeBudget).prod
          (Measure.pi fun _ : Fin (heldOut v) => uniformStrings L)).real
        {q | let rounds := indecisiveLStarLoop settings O (windowStrings k L₁)
              (uniformStrings (L₁ + Fintype.card σ + 3)) (persistDraws q.1)
          ¬ (rounds.length ≤ Fintype.card σ + 1 ∧ ∃ o ∈ rounds.getLast?,
              o.result.ending = .consistent ∧ (uniformStrings L).real {x | ¬
                o.result.hypothesis.rerootedDFA o.read q.2 x
                  (M.eval x.toList ∈ M.accept)} ≤ γ + ξ)}
        ≤ δ

end OrthoDFA
