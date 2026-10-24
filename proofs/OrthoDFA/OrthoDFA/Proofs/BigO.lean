import Mathlib.Analysis.Asymptotics.Defs
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Pow.Real

/-! Lemmas the big-O claims share: a bound past a threshold, and logs of products. -/

namespace OrthoDFA

open Asymptotics Filter Real

/-- A bound at every point past `a` is a big-O at `atTop`; on a product of reals, past `a`
means past it in every coordinate. -/
lemma isBigO_of_ge {ι : Type*} [SemilatticeSup ι] [Nonempty ι] {f g : ι → ℝ} (a : ι) (C : ℝ)
    (h : ∀ b, a ≤ b → |f b| ≤ C * |g b|) : f =O[atTop] g :=
  IsBigO.of_bound C (eventually_atTop.2 ⟨a, fun b hb => by
    simpa only [Real.norm_eq_abs] using h b hb⟩)

lemma log_pos_of_two_le {x : ℝ} (hx : 2 ≤ x) : 0 < log x :=
  Real.log_pos (by linarith)

lemma log_two_le_log {x : ℝ} (hx : 2 ≤ x) : log 2 ≤ log x :=
  Real.log_le_log (by norm_num) hx

/-- A constant is at most a multiple of the log of anything past `2`. -/
lemma le_mul_log_of_two_le {x c : ℝ} (hx : 2 ≤ x) (hc : 0 ≤ c) : c ≤ c / log 2 * log x := by
  have h2 := Real.log_pos (show (1 : ℝ) < 2 by norm_num)
  rw [div_mul_eq_mul_div, le_div_iff₀ h2]
  exact mul_le_mul_of_nonneg_left (log_two_le_log hx) hc

/-- The log of what `c·P^k` bounds is at most `(log c / log 2 + k)·log P`, for `P` past `2`. -/
lemma log_le_of_le_mul_pow {y c P : ℝ} {k : ℕ} (hy : 0 < y) (hc : 1 ≤ c) (hP : 2 ≤ P)
    (h : y ≤ c * P ^ k) : log y ≤ (log c / log 2 + k) * log P := by
  have hc0 : 0 < c := by linarith
  have hP0 : 0 < P := by linarith
  calc log y ≤ log (c * P ^ k) := Real.log_le_log hy h
    _ = log c + k * log P := by
        rw [Real.log_mul hc0.ne' (pow_pos hP0 k).ne', Real.log_pow]
    _ ≤ log c / log 2 * log P + k * log P := by
        gcongr
        exact le_mul_log_of_two_le hP (Real.log_nonneg hc)
    _ = (log c / log 2 + k) * log P := by ring

/-- The log of a factor is at most the log of a product whose other factor is past `1`. -/
lemma log_le_log_mul_right {x y : ℝ} (hx : 0 < x) (hy : 1 ≤ y) : log x ≤ log (x * y) :=
  Real.log_le_log hx (le_mul_of_one_le_right hx.le hy)

lemma log_le_log_mul_left {x y : ℝ} (hx : 0 < x) (hy : 1 ≤ y) : log x ≤ log (y * x) :=
  Real.log_le_log hx (le_mul_of_one_le_left hx.le hy)

/-- `1/min x y ≤ 1/x + 1/y` for positive `x` and `y`. -/
lemma one_div_min_le {x y : ℝ} (hx : 0 < x) (hy : 0 < y) : 1 / min x y ≤ 1 / x + 1 / y := by
  rcases min_cases x y with ⟨h, _⟩ | ⟨h, _⟩ <;> rw [h] <;>
    linarith [one_div_pos.2 hx, one_div_pos.2 hy]

end OrthoDFA
