import OrthoDFA.Proofs.TallyRound
import Mathlib.Analysis.Complex.Exponential

/-!
# The tally round's spurious tails in ratio form

`P(Bin(T, r) ≥ m) ≤ (e T r / m)^m`. With `T = (versionCap + 1) n` probes and `n ≤ 3m₁ / (c₀q₀)`
per version (enough for `P(Bin(n, c₀q₀) < m)` to be small), the light term of `TallyRound` is at
most `e^{-m₁}` once the light rate is a small enough share of the clean rate:
`ρ / (c₀q₀) ≤ 1 / (3 e² (versionCap + 1))`. The heavy term is at most `e^{-m₂}` once
`(versionCap + 1) J ρH ≤ m₂ / e²`.
-/

namespace OrthoDFA

theorem binomSfGe_le_choose {r : ℝ} (hr0 : 0 ≤ r) (hr1 : r ≤ 1) :
    ∀ T m : ℕ, binomSfGe T r m ≤ T.choose m * r ^ m
  | T, 0 => by simp [binomSfGe_zero_right]
  | 0, m + 1 => by simp [binomSfGe_zero_left]
  | T + 1, m + 1 => by
    rw [binomSfGe_succ]
    have h1 := binomSfGe_le_choose hr0 hr1 T m
    have h2 := binomSfGe_le_choose hr0 hr1 T (m + 1)
    have h3 : 0 ≤ binomSfGe T r (m + 1) := binomSfGe_nonneg hr0 hr1 _
    rw [Nat.choose_succ_succ', Nat.cast_add, add_mul]
    have a1 : r * binomSfGe T r m ≤ (T.choose m : ℝ) * r ^ (m + 1) := by
      have := mul_le_mul_of_nonneg_left h1 hr0
      calc r * binomSfGe T r m ≤ r * ((T.choose m : ℝ) * r ^ m) := this
        _ = (T.choose m : ℝ) * r ^ (m + 1) := by ring
    have a2 : (1 - r) * binomSfGe T r (m + 1) ≤ (T.choose (m + 1) : ℝ) * r ^ (m + 1) := by
      have : (1 - r) * binomSfGe T r (m + 1) ≤ binomSfGe T r (m + 1) := by nlinarith
      linarith
    linarith

/-- `P(Bin(T, r) ≥ m) ≤ (e T r / m)^m`. -/
theorem binomSfGe_le_pow {r : ℝ} (hr0 : 0 ≤ r) (hr1 : r ≤ 1) (T : ℕ) {m : ℕ} (hm : 1 ≤ m) :
    binomSfGe T r m ≤ (Real.exp 1 * T * r / m) ^ m := by
  refine (binomSfGe_le_choose hr0 hr1 T m).trans ?_
  have hc : (T.choose m : ℝ) ≤ (T : ℝ) ^ m / m.factorial := Nat.choose_le_pow_div m T
  have hf : (m : ℝ) ^ m / m.factorial ≤ Real.exp m :=
    Real.pow_div_factorial_le_exp (m : ℝ) (by positivity) m
  have hm0 : (0 : ℝ) < m := by exact_mod_cast hm
  have hfac : (0 : ℝ) < m.factorial := by exact_mod_cast Nat.factorial_pos m
  have h1 : (1 : ℝ) / m.factorial ≤ Real.exp m / (m : ℝ) ^ m := by
    rw [div_le_div_iff₀ hfac (by positivity)]
    rw [div_le_iff₀ hfac] at hf
    linarith
  calc (T.choose m : ℝ) * r ^ m ≤ (T : ℝ) ^ m / m.factorial * r ^ m :=
        mul_le_mul_of_nonneg_right hc (pow_nonneg hr0 m)
    _ = (T * r) ^ m * (1 / m.factorial) := by rw [mul_pow]; ring
    _ ≤ (T * r) ^ m * (Real.exp m / (m : ℝ) ^ m) :=
        mul_le_mul_of_nonneg_left h1 (pow_nonneg (by positivity) m)
    _ = (Real.exp 1 * T * r / m) ^ m := by
        rw [← Real.exp_one_pow, div_pow, mul_pow, mul_pow]; field_simp; ring

/-- The spurious term in ratio form: with `n ≤ 3m / (c₀q₀)` probes per version over
`versionCap + 1` versions, and the rate `r` of records that are not true at most
`c₀q₀ / (3 e² (versionCap + 1))`, it is at most `e^{-m}`. -/
theorem spurious_tail_ratio {r cq : ℝ} (hr0 : 0 ≤ r) (hr1 : r ≤ 1) (hcq : 0 < cq) {V n m : ℕ}
    (hm : 1 ≤ m) (hn : (n : ℝ) ≤ 3 * m / cq)
    (hratio : r / cq ≤ 1 / (3 * Real.exp 1 ^ 2 * (V + 1))) :
    binomSfGe ((V + 1) * n) r m ≤ Real.exp (-m) := by
  refine (binomSfGe_le_pow hr0 hr1 _ hm).trans ?_
  have hm0 : (0 : ℝ) < m := by exact_mod_cast hm
  have he : 0 < Real.exp 1 := Real.exp_pos 1
  have hbase : Real.exp 1 * ((V + 1) * n : ℕ) * r / m ≤ Real.exp (-1) := by
    rw [div_le_iff₀ hm0]
    have h1 : r ≤ cq / (3 * Real.exp 1 ^ 2 * (V + 1)) := by
      rw [div_le_iff₀ hcq] at hratio
      calc r ≤ 1 / (3 * Real.exp 1 ^ 2 * (V + 1)) * cq := hratio
        _ = cq / (3 * Real.exp 1 ^ 2 * (V + 1)) := by ring
    have h2 : ((V + 1) * n : ℕ) * r
        ≤ (V + 1) * (3 * m / cq) * (cq / (3 * Real.exp 1 ^ 2 * (V + 1))) := by
      push_cast
      have := mul_le_mul_of_nonneg_left hn (by positivity : (0 : ℝ) ≤ V + 1)
      exact mul_le_mul this h1 hr0 (by positivity)
    have h3 : (V + 1 : ℝ) * (3 * m / cq) * (cq / (3 * Real.exp 1 ^ 2 * (V + 1)))
        = m / Real.exp 1 ^ 2 := by field_simp
    rw [h3] at h2
    have h4 : Real.exp (-1) = 1 / Real.exp 1 := by rw [Real.exp_neg, one_div]
    rw [h4]
    calc Real.exp 1 * ((V + 1) * n : ℕ) * r = Real.exp 1 * (((V + 1) * n : ℕ) * r) := by ring
      _ ≤ Real.exp 1 * (m / Real.exp 1 ^ 2) := mul_le_mul_of_nonneg_left h2 he.le
      _ = 1 / Real.exp 1 * m := by field_simp
  have hnn : 0 ≤ Real.exp 1 * ((V + 1) * n : ℕ) * r / m := by positivity
  calc (Real.exp 1 * ((V + 1) * n : ℕ) * r / m) ^ m ≤ Real.exp (-1) ^ m :=
        pow_le_pow_left₀ hnn hbase m
    _ = Real.exp (-m) := by rw [← Real.exp_nat_mul]; ring_nf

/-- The heavy term in ratio form: over `N` probes at rate `r`, with `N r ≤ m / e²`, it is at most
`e^{-m}`. -/
theorem heavy_tail_ratio {r : ℝ} (hr0 : 0 ≤ r) (hr1 : r ≤ 1) {N m : ℕ} (hm : 1 ≤ m)
    (h : N * r ≤ m / Real.exp 1 ^ 2) : binomSfGe N r m ≤ Real.exp (-m) := by
  refine (binomSfGe_le_pow hr0 hr1 N hm).trans ?_
  have hm0 : (0 : ℝ) < m := by exact_mod_cast hm
  have he : 0 < Real.exp 1 := Real.exp_pos 1
  have hbase : Real.exp 1 * N * r / m ≤ Real.exp (-1) := by
    rw [div_le_iff₀ hm0, Real.exp_neg]
    calc Real.exp 1 * N * r = Real.exp 1 * (N * r) := by ring
      _ ≤ Real.exp 1 * (m / Real.exp 1 ^ 2) := mul_le_mul_of_nonneg_left h he.le
      _ = (Real.exp 1)⁻¹ * m := by field_simp
  calc (Real.exp 1 * N * r / m) ^ m ≤ Real.exp (-1) ^ m :=
        pow_le_pow_left₀ (by positivity) hbase m
    _ = Real.exp (-m) := by rw [← Real.exp_nat_mul]; ring_nf

end OrthoDFA
