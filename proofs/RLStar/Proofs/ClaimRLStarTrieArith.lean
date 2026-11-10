import RLStar.Proofs.ClaimRLStar

/-!
# The trie claim's arithmetic

`((L + Q)⁶ E)^m ≤ |Σ|^⌈L/2⌉` once L passes the tangent bound on `6 m log (L + 1)` at
`24 m/log |Σ|`.
-/

noncomputable section

set_option maxHeartbeats 400000

namespace OrthoDFA

open Real

lemma pow_of_tangent_m {A Q E : ℝ} {L m : ℕ} (hA : 2 ≤ A) (hQ : 1 ≤ Q) (hE : 1 ≤ E) (hm : 1 ≤ m)
    (hL : 1 + 4 * m * (log E + 6 * log (Q + 4) + 6 * log m + 24) / log A ≤ L) :
    (((L : ℝ) + Q) ^ 6 * E) ^ m ≤ A ^ ((L + 1) / 2) := by
  have hl2 := Real.log_two_gt_d9
  have ha : log 2 ≤ log A := log_two_le_log hA
  have ha0 : 0 < log A := by linarith
  have hm1 : (1 : ℝ) ≤ m := by exact_mod_cast hm
  have hm0 : (0 : ℝ) < m := by linarith
  have hlm : 0 ≤ log (m : ℝ) := Real.log_nonneg hm1
  have hL0 : (0 : ℝ) ≤ L := Nat.cast_nonneg L
  have hk : (L : ℝ) ≤ 2 * (((L + 1) / 2 : ℕ) : ℝ) := by
    exact_mod_cast (show L ≤ 2 * ((L + 1) / 2) by omega)
  have hlE : 0 ≤ log E := Real.log_nonneg hE
  have hlQ : 0 ≤ log (Q + 4) := Real.log_nonneg (by linarith)
  have hL' : 4 * m * (log E + 6 * log (Q + 4) + 6 * log m + 24) ≤ ((L : ℝ) - 1) * log A := by
    have := (div_le_iff₀ ha0).1 (show 4 * m * (log E + 6 * log (Q + 4) + 6 * log m + 24) / log A
      ≤ (L : ℝ) - 1 by linarith)
    linarith
  have hT0 : 0 < 24 * m / log A := by positivity
  have hL1 : (0 : ℝ) < L + 1 := by linarith
  have htan : log ((L : ℝ) + 1) ≤ log (24 * m / log A) + ((L : ℝ) + 1) * log A / (24 * m) - 1 := by
    have := Real.log_le_sub_one_of_pos (div_pos hL1 hT0)
    rw [Real.log_div hL1.ne' hT0.ne'] at this
    have e : ((L : ℝ) + 1) / (24 * m / log A) = ((L : ℝ) + 1) * log A / (24 * m) := by
      field_simp
    rw [e] at this
    linarith
  have hlT : log (24 / log A) ≤ 4 := by
    have hT35 : 24 / log A ≤ 35 := by
      rw [div_le_iff₀ ha0]; linarith
    have h35 : (35 : ℝ) ≤ Real.exp 4 := by
      have e4 : Real.exp 4 = Real.exp 1 ^ 4 := by rw [← Real.exp_nat_mul]; norm_num
      rw [e4]
      have : (2.7 : ℝ) ≤ Real.exp 1 := by have := Real.exp_one_gt_d9; linarith
      have : (2.7 : ℝ) ^ 4 ≤ Real.exp 1 ^ 4 := pow_le_pow_left₀ (by norm_num) this 4
      norm_num at this ⊢; linarith
    calc log (24 / log A) ≤ log 35 := Real.log_le_log (by positivity) hT35
      _ ≤ 4 := by rw [Real.log_le_iff_le_exp (by norm_num)]; exact h35
  have hlTm : log (24 * m / log A) ≤ log m + 4 := by
    rw [show 24 * (m : ℝ) / log A = m * (24 / log A) by ring,
      Real.log_mul hm0.ne' (by positivity)]
    linarith
  have hLQ : log ((L : ℝ) + Q) ≤ log ((L : ℝ) + 1) + log (Q + 4) := by
    rw [← Real.log_mul hL1.ne' (by linarith)]
    exact Real.log_le_log (by linarith) (by nlinarith)
  have h6 : 6 * m * log ((L : ℝ) + 1) ≤ 6 * m * (log m + 3) + ((L : ℝ) + 1) * log A / 4 := by
    have := mul_le_mul_of_nonneg_left (htan.trans (by linarith [hlTm] :
      log (24 * m / log A) + ((L : ℝ) + 1) * log A / (24 * m) - 1
        ≤ log m + 3 + ((L : ℝ) + 1) * log A / (24 * m))) (by positivity : (0 : ℝ) ≤ 6 * m)
    have e : 6 * m * (log m + 3 + ((L : ℝ) + 1) * log A / (24 * m))
        = 6 * m * (log m + 3) + ((L : ℝ) + 1) * log A / 4 := by field_simp; ring
    linarith
  have hmain : m * (6 * log ((L : ℝ) + Q) + log E) ≤ (L : ℝ) * log A / 2 := by
    have := mul_le_mul_of_nonneg_left hLQ (by positivity : (0 : ℝ) ≤ 6 * m)
    nlinarith
  have hk' : (L : ℝ) * log A / 2 ≤ (((L + 1) / 2 : ℕ) : ℝ) * log A := by
    have := mul_le_mul_of_nonneg_right hk ha0.le
    linarith
  have hpos : (0 : ℝ) < ((L : ℝ) + Q) ^ 6 * E := by positivity
  have hA0 : (0 : ℝ) < A := by linarith
  rw [← Real.log_le_log_iff (pow_pos hpos _) (pow_pos hA0 _), Real.log_pow, Real.log_pow,
    Real.log_mul (by positivity) (by linarith), Real.log_pow]
  push_cast
  nlinarith

end OrthoDFA

end
