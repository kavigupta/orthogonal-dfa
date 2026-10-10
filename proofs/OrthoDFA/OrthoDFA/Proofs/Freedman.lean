import Mathlib.Probability.Martingale.Basic
import Mathlib.Probability.Moments.Basic
import Mathlib.Analysis.Convex.SpecificFunctions.Basic

/-! # The exponential below its chord -/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory Finset Real

theorem exp_mul_le_chord {l b x : ℝ} (hb : 0 < b) (hx0 : 0 ≤ x) (hxb : x ≤ b) :
    exp (l * x) ≤ 1 + x * ((exp (l * b) - 1) / b) := by
  have h := convexOn_exp.2 (Set.mem_univ 0) (Set.mem_univ (l * b))
    (show (0 : ℝ) ≤ 1 - x / b by rw [sub_nonneg, div_le_one hb]; exact hxb)
    (div_nonneg hx0 hb.le) (show 1 - x / b + x / b = 1 by ring)
  simp only [smul_eq_mul, mul_zero, zero_add, exp_zero, mul_one] at h
  have hx : x / b * (l * b) = l * x := by field_simp
  rw [hx] at h
  calc exp (l * x) ≤ 1 - x / b + x / b * exp (l * b) := h
    _ = _ := by field_simp; ring

variable {Ω : Type*} {m0 : MeasurableSpace Ω} {μ : Measure Ω} [IsProbabilityMeasure μ]

end OrthoDFA
