import OrthoDFA.Proofs.SeqTest
import Mathlib.Analysis.Complex.ExponentialBounds

/-!
# The sequential test as a linear threshold

For `h ≤ n` the test fires exactly where `h log (p₁/p₀) + (n − h) log ((1 − p₁)/(1 − p₀))`
reaches `log (1/a)` (`seqRate_iff_log`), a line in `(n, h)`. Above `p₀` the test fires past a
rate between `p₀` and `p₁` (`seqRate_up_gt`), more hits only helping; below, fewer hits only help.
A count past `log (1/a) / (log 2 − 1/2)` hits, at least `φ` of the draws, fires the test of `φ/2`
against `φ` (`seqRate_share`).
-/

namespace OrthoDFA

variable {p₀ p₁ a : ℝ}

theorem likelihoodRatio_pos (hp₀ : 0 < p₀) (hp₀1 : p₀ < 1) (hp₁ : 0 < p₁) (hp₁1 : p₁ < 1)
    (n h : ℕ) : 0 < likelihoodRatio p₀ p₁ n h := by
  unfold likelihoodRatio
  have : 0 < (1 - p₁) / (1 - p₀) := div_pos (by linarith) (by linarith)
  have : 0 < p₁ / p₀ := div_pos hp₁ hp₀
  positivity

theorem log_likelihoodRatio_of_le (hp₀ : 0 < p₀) (hp₀1 : p₀ < 1) (hp₁ : 0 < p₁) (hp₁1 : p₁ < 1)
    {n h : ℕ} (hh : h ≤ n) :
    Real.log (likelihoodRatio p₀ p₁ n h)
      = h * Real.log (p₁ / p₀) + (n - h : ℝ) * Real.log ((1 - p₁) / (1 - p₀)) :=
  log_likelihoodRatio hp₀ hp₀1 hp₁ hp₁1 hh

/-- For `h ≤ n` the test fires where the log of the ratio reaches `log (1/a)`. -/
theorem seqRate_iff_log (hp₀ : 0 < p₀) (hp₀1 : p₀ < 1) (hp₁ : 0 < p₁) (hp₁1 : p₁ < 1)
    (ha : 0 < a) {n h : ℕ} (hh : h ≤ n) :
    seqRate p₀ p₁ a n h ↔ Real.log (1 / a)
      ≤ h * Real.log (p₁ / p₀) + (n - h : ℝ) * Real.log ((1 - p₁) / (1 - p₀)) := by
  unfold seqRate
  rw [← log_likelihoodRatio_of_le hp₀ hp₀1 hp₁ hp₁1 hh,
    Real.log_le_log_iff (by positivity) (likelihoodRatio_pos hp₀ hp₀1 hp₁ hp₁1 n h)]

/-- Fewer hits lower the ratio where `p₁ ≤ p₀`. -/
theorem likelihoodRatio_anti (hp₀ : 0 < p₀) (hp₀1 : p₀ < 1) (hp₁ : 0 ≤ p₁) (hp₁0 : p₁ ≤ p₀)
    {n h h' : ℕ} (hh : h ≤ h') (hh' : h' ≤ n) :
    likelihoodRatio p₀ p₁ n h' ≤ likelihoodRatio p₀ p₁ n h := by
  induction hh with
  | refl => exact le_rfl
  | @step k hk ih =>
    refine le_trans ?_ (ih (by omega))
    unfold likelihoodRatio
    rw [show n - k = n - (k + 1) + 1 by omega, pow_succ, pow_succ]
    have hA0 : 0 ≤ p₁ / p₀ := div_nonneg hp₁ hp₀.le
    have hA : p₁ / p₀ ≤ 1 := by rw [div_le_one hp₀]; exact hp₁0
    have hB : 1 ≤ (1 - p₁) / (1 - p₀) := by rw [le_div_iff₀ (by linarith)]; linarith
    have hX : 0 ≤ (p₁ / p₀) ^ k * ((1 - p₁) / (1 - p₀)) ^ (n - (k + 1)) := by
      have : 0 ≤ (1 - p₁) / (1 - p₀) := by linarith
      positivity
    nlinarith

/-- Above `p₀`, more hits keep the test firing. -/
theorem seqRate_mono_up (hp₀ : 0 < p₀) (hp₀1 : p₀ < 1) (hp₁ : p₀ ≤ p₁) (hp₁1 : p₁ ≤ 1)
    {n h h' : ℕ} (hs : seqRate p₀ p₁ a n h) (hh : h ≤ h') (hh' : h' ≤ n) :
    seqRate p₀ p₁ a n h' :=
  hs.trans (likelihoodRatio_mono hp₀ hp₀1 hp₁ hp₁1 hh hh')

/-- Below `p₀`, fewer hits keep the test firing. -/
theorem seqRate_anti_down (hp₀ : 0 < p₀) (hp₀1 : p₀ < 1) (hp₁ : 0 ≤ p₁) (hp₁0 : p₁ ≤ p₀)
    {n h h' : ℕ} (hs : seqRate p₀ p₁ a n h) (hh : h' ≤ h) (hh' : h ≤ n) :
    seqRate p₀ p₁ a n h' :=
  hs.trans (likelihoodRatio_anti hp₀ hp₀1 hp₁ hp₁0 hh hh')

/-- Before any draw the test does not fire. -/
theorem not_seqRate_zero (ha : 0 < a) (ha1 : a < 1) : ¬ seqRate p₀ p₁ a 0 0 := by
  unfold seqRate
  rw [likelihoodRatio_zero, not_le, lt_div_iff₀ ha]
  linarith

/-- Above `p₀`, the test fires only past a rate above `p₀`: `p₀ n < h`. -/
theorem seqRate_up_gt (hp₀ : 0 < p₀) (h01 : p₀ < p₁) (hp₁1 : p₁ < 1) (ha : 0 < a) (ha1 : a < 1)
    {n h : ℕ} (hs : seqRate p₀ p₁ a n h) : p₀ * n < h := by
  have hp₁ : 0 < p₁ := by linarith
  by_cases hh : h ≤ n
  swap
  · have : (n : ℝ) < h := by exact_mod_cast not_le.1 hh
    nlinarith [show (0 : ℝ) ≤ n from Nat.cast_nonneg n]
  rw [seqRate_iff_log hp₀ (by linarith) hp₁ hp₁1 ha hh] at hs
  have hlg : 0 < Real.log (1 / a) := Real.log_pos (by rw [one_div]; exact one_lt_inv_iff₀.2 ⟨ha, ha1⟩)
  -- the line `h A − (n − h) B ≥ log (1/a) > 0` sits above `h = p₀ n`
  have hA : 0 < Real.log (p₁ / p₀) := Real.log_pos (by rw [one_lt_div hp₀]; exact h01)
  have hB0 : Real.log ((1 - p₁) / (1 - p₀)) < 0 :=
    Real.log_neg (div_pos (by linarith) (by linarith)) (by rw [div_lt_one (by linarith)]; linarith)
  have hk : p₀ * Real.log (p₁ / p₀) + (1 - p₀) * Real.log ((1 - p₁) / (1 - p₀)) < 0 := by
    have s1 : p₀ * Real.log (p₁ / p₀) < p₁ - p₀ := by
      have := Real.log_lt_sub_one_of_pos (div_pos hp₁ hp₀) (by
        intro h; rw [div_eq_one_iff_eq hp₀.ne'] at h; linarith)
      have e : p₀ * (p₁ / p₀ - 1) = p₁ - p₀ := by field_simp
      nlinarith
    have s2 : (1 - p₀) * Real.log ((1 - p₁) / (1 - p₀)) ≤ p₀ - p₁ := by
      have := Real.log_le_sub_one_of_pos (div_pos (show 0 < 1 - p₁ by linarith)
        (show 0 < 1 - p₀ by linarith))
      have e : (1 - p₀) * ((1 - p₁) / (1 - p₀) - 1) = p₀ - p₁ := by
        field_simp [show (1 - p₀) ≠ 0 by linarith]; ring
      nlinarith
    linarith
  have hn0 : (0 : ℝ) ≤ n := Nat.cast_nonneg n
  have hhn : (h : ℝ) ≤ n := by exact_mod_cast hh
  by_contra hc
  push Not at hc
  -- at `h ≤ p₀ n` the line is at most `n (p₀ A + (1 − p₀) B) ≤ 0`
  have : (h : ℝ) * Real.log (p₁ / p₀) + (n - h : ℝ) * Real.log ((1 - p₁) / (1 - p₀))
      ≤ n * (p₀ * Real.log (p₁ / p₀) + (1 - p₀) * Real.log ((1 - p₁) / (1 - p₀))) := by
    nlinarith
  nlinarith

/-- A count at least `φ` of the draws and past `6 log (1/a)` hits fires the test of `φ/2` against
`φ`: each draw multiplies the ratio by 2 on a hit and by at least `e^{-(φ/2)/(1-φ)}` on a miss,
and there are at most `(1-φ)/φ` misses a hit. -/
theorem seqRate_share {φ : ℝ} (hφ0 : 0 < φ) (hφ1 : φ < 1) (ha : 0 < a) {n₀ n h : ℕ}
    (hn₀ : 6 * Real.log (1 / a) ≤ n₀) (hh₀ : n₀ ≤ h) (hhn : h ≤ n) (hφh : φ * n ≤ h) :
    seqRate (φ / 2) φ a n h := by
  rw [seqRate_iff_log (by linarith) (by linarith) hφ0 hφ1 ha hhn]
  have e2 : φ / (φ / 2) = 2 := by field_simp
  rw [e2]
  have hB : Real.log ((1 - φ / 2) / (1 - φ)) ≤ φ / 2 / (1 - φ) := by
    have := Real.log_le_sub_one_of_pos (div_pos (show 0 < 1 - φ / 2 by linarith)
      (show 0 < 1 - φ by linarith))
    have e : (1 - φ / 2) / (1 - φ) - 1 = φ / 2 / (1 - φ) := by
      field_simp [show (1 - φ) ≠ 0 by linarith]; ring
    linarith
  have hneg : Real.log ((1 - φ) / (1 - φ / 2)) = - Real.log ((1 - φ / 2) / (1 - φ)) := by
    rw [← Real.log_inv, inv_div]
  rw [hneg]
  have hl2 : (0.6931471803 : ℝ) < Real.log 2 := Real.log_two_gt_d9
  have hhr : (0 : ℝ) ≤ h := Nat.cast_nonneg h
  have hnh : (n - h : ℝ) ≥ 0 := by
    have : (h : ℝ) ≤ n := by exact_mod_cast hhn
    linarith
  -- the misses: `(n − h) φ ≤ (1 − φ) h`
  have hmiss : (n - h : ℝ) * (φ / 2 / (1 - φ)) ≤ h / 2 := by
    have h1 : (n - h : ℝ) * φ ≤ (1 - φ) * h := by nlinarith
    rw [show (n - h : ℝ) * (φ / 2 / (1 - φ)) = (n - h : ℝ) * φ / (2 * (1 - φ)) by
      field_simp]
    rw [div_le_iff₀ (by nlinarith)]
    nlinarith
  have hB0 : 0 ≤ Real.log ((1 - φ / 2) / (1 - φ)) :=
    Real.log_nonneg (by rw [le_div_iff₀ (by linarith)]; linarith)
  have h3 : (n - h : ℝ) * Real.log ((1 - φ / 2) / (1 - φ)) ≤ h / 2 :=
    (mul_le_mul_of_nonneg_left hB hnh).trans hmiss
  have h4 : (n₀ : ℝ) ≤ h := by exact_mod_cast hh₀
  nlinarith

end OrthoDFA
