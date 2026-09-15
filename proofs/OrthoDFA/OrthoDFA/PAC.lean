import OrthoDFA.Algorithm
import OrthoDFA.Complexity

/-!
# The end-to-end PAC bound

Puts the chain together at the top: the run-level failure bound
`(1−p)^N + N·b` (from `algorithm_correct_general`) is driven below the confidence
`δ` by the explicit termination threshold (`geom_le`) and the soundness budget.

`p` is the per-round good-pass probability (delivered by liveness:
`chosen_avoids_bad_whp` / `liveness_produces_good`, whose flip-mass bounds come from
`denoised_loss_eq_flip` ∘ `read_disagreement_mean`); `b` is the per-round
false-admission (delivered by `certErr_bound`).  This theorem is where they meet the
confidence target.

Result (PAC form): with probability ≥ 1 − δ the run produces a good family and
admits no bad one — provided `N ≥ log(2/δ)/p` and `N·b ≤ δ/2`, both explicit.
-/

namespace OrthoDFA

open MeasureTheory
open scoped ENNReal

variable {N : ℕ} {α : Fin N → Type*} [∀ r, MeasurableSpace (α r)]

/-- **End-to-end PAC bound.**  Under the accumulating-pool model, if each round is a
good pass with probability ≥ `p` (liveness) and admits a bad family with
probability ≤ `b` (soundness), then for `N ≥ log(2/δ)/p` and `N·b ≤ δ/2` the
failure probability is at most `δ` — i.e. with probability ≥ 1 − δ some round
produced a good family and none admitted a bad one. -/
theorem clustering_pac
    /-
      ν is the per-round distribution of the random choices (the "pool" of
      random bits) that drive the algorithm; each round is independent
      (implicitly enforced by having the conclusion be over a product measure).
    -/
    (ν : ∀ r, Measure (α r)) [∀ r, IsProbabilityMeasure (ν r)]
    /-
      trigger r ⊆ (round r's space): the block-local "findability" event — the
      fresh material drawn in round r that makes the round succeed (an accept-
      preserving suffix is drawn). htrig: it is measurable.
    -/
    (trigger : ∀ r, Set (α r)) (htrig : ∀ r, MeasurableSet (trigger r))
    /-
      goodErr r: "round r is NOT a good pass"; badErr r: "round r admits a bad
      (drifted) family". Both are events over the whole run (the accumulating
      pool), so they are sets of full trajectories `∀ r, α r`, not block-local.
    -/
    (goodErr badErr : Fin N → Set (∀ r, α r))
    /-
      Liveness link: a block-local trigger forces a good round. If round r's fresh
      draw lands in `trigger r`, then round r is a good pass (not in `goodErr r`).
      This is what lets the independent triggers drive the (correlated) good-passes.
    -/
    (himplies : ∀ r, ∀ x : ∀ r, α r, x r ∈ trigger r → x ∉ goodErr r)
    /-
      p = per-round good-pass probability (from liveness); b = per-round bad-admit
      probability (from `certErr_bound`); δ = the PAC confidence (failure budget).
      hp0/hp1: p is a genuine probability, 0 < p ≤ 1 (p>0 is what makes (1−p)^N decay).
    -/
    (p b δ : ℝ) (hp0 : 0 < p) (hp1 : p ≤ 1)
    /- Findability, quantitative: each round's trigger fires with probability ≥ p. -/
    (hp : ∀ r, p ≤ (ν r).real (trigger r))
    /- Soundness, quantitative: each round admits a bad family with probability ≤ b. -/
    (hbad : ∀ r, (Measure.pi ν).real (badErr r) ≤ b)
    /- The confidence target is a genuine probability budget. -/
    (hδ : 0 < δ)
    /-
      Sample-complexity conditions making the two failure sources each ≤ δ/2:
      hN — enough rounds for termination, N ≥ log(2/δ)/p (drives (1−p)^N ≤ δ/2);
      hNb — small enough per-round false-admit, N·b ≤ δ/2 (drives the union bound).
    -/
    (hN : Real.log (2 / δ) / p ≤ (N : ℝ))
    (hNb : (N : ℝ) * b ≤ δ / 2) :
    (Measure.pi ν).real ({x | ∀ r, x ∈ goodErr r} ∪ (⋃ r, badErr r)) ≤ δ := by
  have hrun := algorithm_correct_general ν trigger htrig goodErr badErr himplies
    p b hp1 hp hbad
  have hterm : (1 - p) ^ N ≤ δ / 2 := by
    refine geom_le hp0 hp1 (by linarith) ?_
    rwa [show (1 : ℝ) / (δ / 2) = 2 / δ by rw [one_div, inv_div]]
  linarith [hrun, hterm, hNb]

#print axioms clustering_pac

end OrthoDFA
