import Mathlib.Analysis.SpecialFunctions.Log.Basic

/-!
# Sample complexity: the thresholds are explicit

Turns "for `k,m,N` large enough" into explicit closed forms.  Each failure term is
`c·exp(−2·k·t²)` or `(1−q)^N`; solving for the threshold gives the `k`, `m`, `N`
the proof needs, so the end-to-end theorem quantifies "for `k ≥ …`" rather than
"there exist `k`".
-/

namespace OrthoDFA

/-- Exponential-tail threshold.  `c·exp(−2 k t²) ≤ ε` as soon as
`k ≥ log(c/ε) / (2 t²)`.  Used for the decision term (`t = s−τ`) and the placement
term (`t = γ`). -/
theorem tail_le {t c ε k : ℝ} (ht : 0 < t) (hc : 0 < c) (hε : 0 < ε)
    (hk : Real.log (c / ε) / (2 * t ^ 2) ≤ k) :
    c * Real.exp (-2 * k * t ^ 2) ≤ ε := by
  have h2t : 0 < 2 * t ^ 2 := by positivity
  have hkey : Real.log (c / ε) ≤ 2 * k * t ^ 2 := by
    rw [div_le_iff₀ h2t] at hk; nlinarith [hk]
  calc c * Real.exp (-2 * k * t ^ 2)
      ≤ c * Real.exp (- Real.log (c / ε)) := by
        apply mul_le_mul_of_nonneg_left _ hc.le
        apply Real.exp_le_exp.mpr; linarith
    _ = c * (c / ε)⁻¹ := by rw [Real.exp_neg, Real.exp_log (by positivity)]
    _ = ε := by field_simp

/-- Geometric threshold.  `(1−q)^N ≤ ε` as soon as `N ≥ log(1/ε) / q`
(for `0 < q ≤ 1`).  Used for the termination term (`q = (1−reject)·pp`). -/
theorem geom_le {q ε : ℝ} {N : ℕ} (hq0 : 0 < q) (hq1 : q ≤ 1) (hε : 0 < ε)
    (hN : Real.log (1 / ε) / q ≤ (N : ℝ)) :
    (1 - q) ^ N ≤ ε := by
  have h1 : (1 - q) ≤ Real.exp (-q) := by
    have := Real.add_one_le_exp (-q); linarith
  have h1nn : (0 : ℝ) ≤ 1 - q := by linarith
  have h2 : (1 - q) ^ N ≤ (Real.exp (-q)) ^ N := pow_le_pow_left₀ h1nn h1 N
  have h3 : (Real.exp (-q)) ^ N = Real.exp (-q * (N : ℝ)) := by
    rw [← Real.exp_nat_mul]; ring_nf
  have h4 : -q * (N : ℝ) ≤ Real.log ε := by
    rw [div_le_iff₀ hq0] at hN
    rw [show Real.log (1 / ε) = - Real.log ε by rw [one_div, Real.log_inv]] at hN
    nlinarith [hN]
  calc (1 - q) ^ N ≤ Real.exp (-q * (N : ℝ)) := by rw [← h3]; exact h2
    _ ≤ Real.exp (Real.log ε) := Real.exp_le_exp.mpr h4
    _ = ε := Real.exp_log hε

#print axioms tail_le
#print axioms geom_le

end OrthoDFA
