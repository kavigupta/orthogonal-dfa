import OrthoDFA.Proofs.SeqTest

/-! Testing `y` and `ε` against their midpoint, one of the tests at level `a` fires within
`12 ε / (ε − y)² · log(1/a) + 1` draws. -/

namespace OrthoDFA

/-- The algebra behind `drift_ge`. -/
theorem drift_alg {s t W Y p q : ℝ} (hs0 : 0 < s) (hs1 : s ≤ 1) (ht0 : 0 < t) (ht1 : t ≤ 1)
    (hW1 : W ≤ s) (hW2 : s ≤ (1 + s) * W) (hY1 : Y ≤ t) (hY2 : t ≤ (1 + t) * Y)
    (hp : s ^ 2 ≤ p) (hq : t ^ 2 ≤ q) :
    s * t * (W + p + Y + (Y + q) + W) ≤ 3 * (1 + s) * ((W + p) * (Y + q) - Y * W) := by
  have hp0 : 0 ≤ p := le_trans (sq_nonneg s) hp
  have hq0 : 0 ≤ q := le_trans (sq_nonneg t) hq
  have hst : 0 < s * t := mul_pos hs0 ht0
  -- multiply through by `1 + t`
  suffices h : (1 + t) * (s * t * (W + p + Y + (Y + q) + W))
      ≤ (1 + t) * (3 * (1 + s) * ((W + p) * (Y + q) - Y * W)) from
    le_of_mul_le_mul_left h (by linarith)
  have a1 : 3 * (1 + t) * q * s ≤ 3 * (1 + t) * q * ((1 + s) * W) :=
    mul_le_mul_of_nonneg_left hW2 (by positivity)
  have a2 : 3 * (1 + s) * p * t ≤ 3 * (1 + s) * p * ((1 + t) * Y) :=
    mul_le_mul_of_nonneg_left hY2 (by positivity)
  have a3 : (1 + t) * s * t * (2 * W) ≤ (1 + t) * s * t * (2 * s) :=
    mul_le_mul_of_nonneg_left (by linarith) (by positivity)
  have a4 : (1 + t) * s * t * (2 * Y) ≤ (1 + t) * s * t * (2 * t) :=
    mul_le_mul_of_nonneg_left (by linarith) (by positivity)
  -- the remaining polynomial is increasing in `p` and `q`
  have cq : 0 ≤ 3 * s * (1 + t) - (1 + t) * s * t := by nlinarith
  have cp : 0 ≤ 3 * (1 + s) * t - (1 + t) * s * t := by nlinarith
  have b1 : (3 * s * (1 + t) - (1 + t) * s * t) * t ^ 2
      ≤ (3 * s * (1 + t) - (1 + t) * s * t) * q := mul_le_mul_of_nonneg_left hq cq
  have b2 : (3 * (1 + s) * t - (1 + t) * s * t) * s ^ 2
      ≤ (3 * (1 + s) * t - (1 + t) * s * t) * p := mul_le_mul_of_nonneg_left hp cp
  have b3 : 3 * (1 + s) * (1 + t) * (s ^ 2 * t ^ 2) ≤ 3 * (1 + s) * (1 + t) * (p * q) :=
    mul_le_mul_of_nonneg_left (mul_le_mul hp hq (sq_nonneg t) hp0) (by positivity)
  have poly : 0 ≤ s * t * (t - t ^ 3) + s * t * (s * t + 2 * s ^ 2 * t + s + 2 * s ^ 2
      + 3 * s * t ^ 2 + 3 * s ^ 2 * t ^ 2) := by
    have : 0 ≤ t - t ^ 3 := by nlinarith
    positivity
  nlinarith [a1, a2, a3, a4, b1, b2, b3, poly]

/-- Testing `y` and `ε` against their midpoint, the drift `A₁A₂ − B₁B₂` is at least
`(ε − y)² / (12 ε)` times `A₁ + B₁ + A₂ + B₂`. -/
theorem drift_ge {y ε : ℝ} (hy : 0 < y) (hyε : y < ε) (hε : ε < 1) :
    (ε - y) ^ 2 / (12 * ε)
        * (Real.log ((y + ε) / 2 / y) + Real.log ((1 - y) / (1 - (y + ε) / 2))
          + Real.log ((1 - (y + ε) / 2) / (1 - ε)) + Real.log (ε / ((y + ε) / 2)))
      ≤ Real.log ((y + ε) / 2 / y) * Real.log ((1 - (y + ε) / 2) / (1 - ε))
        - Real.log ((1 - y) / (1 - (y + ε) / 2)) * Real.log (ε / ((y + ε) / 2)) := by
  set m := (y + ε) / 2 with hm
  set h := (ε - y) / 2 with hh
  have hm0 : 0 < m := by linarith
  have hu0 : 0 < 1 - m := by linarith
  have hh0 : 0 < h := by linarith
  have hhm : h < m := by linarith
  have hhu : h < 1 - m := by linarith
  have hε' : ε = m + h := by linarith
  have hy' : y = m - h := by linarith
  set s := h / m with hs
  set t := h / (1 - m) with ht
  have hs0 : 0 < s := div_pos hh0 hm0
  have ht0 : 0 < t := div_pos hh0 hu0
  have hs1 : s ≤ 1 := by rw [hs, div_le_one hm0]; linarith
  have ht1 : t ≤ 1 := by rw [ht, div_le_one hu0]; linarith
  set X := Real.log (m / y)
  set Y := Real.log ((1 - y) / (1 - m))
  set Z := Real.log ((1 - m) / (1 - ε))
  set W := Real.log (ε / m)
  have hεpos : 0 < ε := by linarith
  have h1y : 0 < 1 - y := by linarith
  have h1ε : 0 < 1 - ε := by linarith
  -- upper bounds
  have hW1 : W ≤ s := by
    have := Real.log_le_sub_one_of_pos (div_pos hεpos hm0)
    have e : ε / m - 1 = s := by rw [hs, hε']; field_simp; ring
    linarith
  have hY1 : Y ≤ t := by
    have := Real.log_le_sub_one_of_pos (div_pos h1y hu0)
    have e : (1 - y) / (1 - m) - 1 = t := by rw [ht, hy']; field_simp; ring
    linarith
  -- lower bounds
  have hW2 : s ≤ (1 + s) * W := by
    have := Real.one_sub_inv_le_log_of_pos (div_pos hεpos hm0)
    have e : 1 - (ε / m)⁻¹ = h / (m + h) := by
      rw [inv_div, one_sub_div hεpos.ne']; congr 1; linarith
    have e2 : (1 + s) * (h / (m + h)) = s := by rw [hs]; field_simp
    have : (1 + s) * (h / (m + h)) ≤ (1 + s) * W :=
      mul_le_mul_of_nonneg_left (by linarith) (by positivity)
    linarith
  have hY2 : t ≤ (1 + t) * Y := by
    have := Real.one_sub_inv_le_log_of_pos (div_pos h1y hu0)
    have e : 1 - ((1 - y) / (1 - m))⁻¹ = h / (1 - m + h) := by
      rw [inv_div, one_sub_div h1y.ne']; congr 1 <;> linarith
    have e2 : (1 + t) * (h / (1 - m + h)) = t := by rw [ht]; field_simp
    have : (1 + t) * (h / (1 - m + h)) ≤ (1 + t) * Y :=
      mul_le_mul_of_nonneg_left (by linarith) (by positivity)
    linarith
  -- the gaps
  have hp : s ^ 2 ≤ X - W := by
    have hpos : 0 < m / y / (ε / m) := by positivity
    have e1 : X - W = Real.log (m / y / (ε / m)) := by
      rw [Real.log_div (by positivity) (by positivity)]
    have := Real.one_sub_inv_le_log_of_pos hpos
    have e2 : 1 - (m / y / (ε / m))⁻¹ = s ^ 2 := by
      rw [hs, hy', hε']; field_simp; ring
    linarith
  have hq : t ^ 2 ≤ Z - Y := by
    have hpos : 0 < (1 - m) / (1 - ε) / ((1 - y) / (1 - m)) := by positivity
    have e1 : Z - Y = Real.log ((1 - m) / (1 - ε) / ((1 - y) / (1 - m))) := by
      rw [Real.log_div (by positivity) (by positivity)]
    have := Real.one_sub_inv_le_log_of_pos hpos
    have e2 : 1 - ((1 - m) / (1 - ε) / ((1 - y) / (1 - m)))⁻¹ = t ^ 2 := by
      rw [ht, hy', hε']; field_simp; ring
    linarith
  have key : s * t * (X + Y + Z + W) ≤ 3 * (1 + s) * (X * Z - Y * W) := by
    have := drift_alg hs0 hs1 ht0 ht1 hW1 hW2 hY1 hY2 hp hq
    have e1 : W + (X - W) = X := by ring
    have e2 : Y + (Z - Y) = Z := by ring
    rw [e1, e2] at this
    linarith
  have hsum : 0 ≤ X + Y + Z + W := by nlinarith
  have hκ : 3 * (1 + s) * ((ε - y) ^ 2 / (12 * ε)) ≤ s * t := by
    have e : 3 * (1 + s) * ((ε - y) ^ 2 / (12 * ε)) = h ^ 2 / m := by
      rw [hs, hh, hε']; field_simp; ring
    have e2 : s * t = h ^ 2 / (m * (1 - m)) := by rw [hs, ht]; field_simp
    rw [e, e2]
    apply div_le_div_of_nonneg_left (by positivity) (by positivity)
    nlinarith
  have h3 : 0 < 3 * (1 + s) := by positivity
  refine le_of_mul_le_mul_left ?_ h3
  calc 3 * (1 + s) * ((ε - y) ^ 2 / (12 * ε) * (X + Y + Z + W))
      = 3 * (1 + s) * ((ε - y) ^ 2 / (12 * ε)) * (X + Y + Z + W) := by ring
    _ ≤ s * t * (X + Y + Z + W) := mul_le_mul_of_nonneg_right hκ hsum
    _ ≤ _ := key

/-- Testing `y` and `ε` against their midpoint, one of the tests at level `a` fires within
`12 ε / (ε − y)² · log(1/a) + 1` draws. -/
theorem stretchBound_mid_le {y ε a : ℝ} (hy : 0 < y) (hyε : y < ε) (hε : ε < 1) (ha0 : 0 < a)
    (ha1 : a ≤ 1) :
    (stretchBound y ((y + ε) / 2) ε a : ℝ) ≤ 12 * ε / (ε - y) ^ 2 * Real.log (1 / a) + 1 := by
  have hd := drift_ge hy hyε hε
  set m := (y + ε) / 2 with hm'
  set X := Real.log (m / y)
  set Y := Real.log ((1 - y) / (1 - m))
  set Z := Real.log ((1 - m) / (1 - ε))
  set W := Real.log (ε / m)
  have hX : 0 < X := Real.log_pos (by rw [one_lt_div hy]; linarith [hm'])
  have hY : 0 < Y := Real.log_pos (by rw [one_lt_div (by linarith [hm'])]; linarith [hm'])
  have hZ : 0 < Z := Real.log_pos (by rw [one_lt_div (by linarith [hm'])]; linarith [hm'])
  have hW : 0 < W := Real.log_pos (by rw [one_lt_div (by linarith [hm'])]; linarith [hm'])
  have hκ : 0 < (ε - y) ^ 2 / (12 * ε) := by
    have : 0 < ε - y := by linarith
    have : 0 < ε := by linarith
    positivity
  have hsum : 0 < X + Y + Z + W := by linarith
  have hN : 0 < X * Z - Y * W := lt_of_lt_of_le (mul_pos hκ hsum) hd
  have hL : 0 ≤ Real.log (1 / a) := Real.log_nonneg (by rw [le_div_iff₀ ha0]; linarith)
  have hx : 0 ≤ (X + Y + Z + W) * Real.log (1 / a) / (X * Z - Y * W) := by positivity
  unfold stretchBound
  refine (Nat.ceil_lt_add_one hx).le.trans ?_
  have : (X + Y + Z + W) * Real.log (1 / a) / (X * Z - Y * W)
      ≤ 12 * ε / (ε - y) ^ 2 * Real.log (1 / a) := by
    rw [div_le_iff₀ hN]
    have e : 12 * ε / (ε - y) ^ 2 = ((ε - y) ^ 2 / (12 * ε))⁻¹ := by rw [inv_div]
    rw [e]
    calc (X + Y + Z + W) * Real.log (1 / a)
        = ((ε - y) ^ 2 / (12 * ε))⁻¹ * ((ε - y) ^ 2 / (12 * ε) * (X + Y + Z + W))
          * Real.log (1 / a) := by rw [← mul_assoc, inv_mul_cancel₀ hκ.ne', one_mul]
      _ ≤ ((ε - y) ^ 2 / (12 * ε))⁻¹ * (X * Z - Y * W) * Real.log (1 / a) := by
          gcongr
      _ = _ := by ring
  linarith

end OrthoDFA
