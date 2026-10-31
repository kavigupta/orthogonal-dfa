import OrthoDFA.Claims.Loop
import OrthoDFA.Proofs.LoopDFAArith
import OrthoDFA.Proofs.BigO

/-!
# The learner's costs, asymptotically: the arithmetic

In the inverted coordinates `iδ = 1/δ`, `ic = 1/c`, `iξ = 1/ξ`, `ip = 1/pAP`, `is = 1/s₀`, `loopΘ`
is a polynomial, so `loopCount` is `ℓ² Θ²/s⁴` times the log of a polynomial, `loopPool` a sum of
logs of polynomials, and `loopMassTerms`' `E` a sum of polynomials times `log X`, its `E₂` and `E₃`
times `log² X` (`claim_terms_le`). `Ae ≥ |Σ|` is the rate edges are crossed at. The general claim
runs at `Ae = 1/β + |Q|`, `1/c = 2R/ξ` and `ξ/4` (`rel_pref_le`, …), where `Ae/c` enters as
`R(1/β + |Q|)/ξ`.
-/

noncomputable section

namespace OrthoDFA

open Real

lemma loopΘ_inv (Q A ic iξ is : ℝ) :
    loopΘ Q A (1 / ic) (1 / iξ) (1 / 2 - 1 / is)
      = (10 * ic + 8 * iξ) * is ^ 2 / 2 + 8 * (10 * ic + 8 * iξ)
        + 4608 * Q * (Q + 1) * (4 * (A + 1) * ic) := by
  unfold loopΘ
  rw [sub_sub_cancel]
  field_simp

lemma loopW₀_inv (Q A iδ : ℝ) :
    loopW₀ Q A (1 / iδ) = Real.exp 1 * Real.sqrt (1 / iδ / (8 * (Q + 1)))
      * (16 / 5 * (Q * ((Q + 2) * A + 1) + 2 * A + 1)
        * Real.log (8 * (Q * ((Q + 2) * A + 1) + 2 * A + 1)
          * (Q * ((Q + 2) * A + 1) + 2 * A + 1 + 1) * (8 * (Q + 1) * iδ))
        + 3 + (Q + 2) * A * (Q + 1)) / 2 + 1 := by
  unfold loopW₀
  dsimp only
  congr 5
  field_simp


lemma loopPool_inv (ℓ Q A Ae iδ ic iξ ip is : ℝ) :
    loopPool ℓ Q A Ae (1 / iδ) (1 / ic) (1 / iξ) (1 / ip) (1 / 2 - 1 / is)
      = (50 * (max (log (20 * loopΘ Q Ae (1 / ic) (1 / iξ) (1 / 2 - 1 / is)))
          (log (8 * loopW₀ Q A (1 / iδ) * (8 * (Q + 1) * iδ))) + 2 * log ℓ) * is ^ 2
          + 20 * is + 28) * ip
        + 120 * log (48 * (Q + 1) * iδ) * ip := by
  unfold loopPool
  dsimp only
  rw [sub_sub_cancel, show 8 * loopW₀ Q A (1 / iδ) / (1 / iδ / (8 * (Q + 1)))
    = 8 * loopW₀ Q A (1 / iδ) * (8 * (Q + 1) * iδ) by field_simp,
    show 3 / (1 / iδ / (8 * (Q + 1)) / 2) = 48 * (Q + 1) * iδ by field_simp; ring]
  field_simp

lemma loopCount_inv (ℓ Q A Ae iδ ic iξ ip is : ℝ) :
    loopCount ℓ Q A Ae (1 / iδ) (1 / ic) (1 / iξ) (1 / ip) (1 / 2 - 1 / is)
      = 161000 * (Q + 2) * loopΘ Q Ae (1 / ic) (1 / iξ) (1 / 2 - 1 / is) ^ 2 * ℓ ^ 2
        * log (16 * (Q + 4)
          * loopPool ℓ Q A Ae (1 / iδ) (1 / ic) (1 / iξ) (1 / ip) (1 / 2 - 1 / is)
          * loopΘ Q Ae (1 / ic) (1 / iξ) (1 / 2 - 1 / is) * ℓ * (8 * (Q + 1) * iδ) ^ 2 * ip)
        * is ^ 4 := by
  unfold loopCount
  dsimp only
  rw [sub_sub_cancel]
  set P := loopPool ℓ Q A Ae (1 / iδ) (1 / ic) (1 / iξ) (1 / ip) (1 / 2 - 1 / is)
  set Θ := loopΘ Q Ae (1 / ic) (1 / iξ) (1 / 2 - 1 / is)
  rw [show 16 * (Q + 4) * P * Θ * ℓ / ((1 / iδ / (8 * (Q + 1))) ^ 2 * (1 / ip))
    = 16 * (Q + 4) * P * Θ * ℓ * (8 * (Q + 1) * iδ) ^ 2 * ip by field_simp]
  field_simp

lemma loopBudget_inv (ℓ Q A Ae iδ ic : ℝ) :
    loopBudget ℓ Q A Ae (1 / iδ) (1 / ic)
      = (36 * (Q * ((Q + 2) * A + 1) + 2 * A + 1) * (4 * (Ae + 1) * ic)
        * log (16 * (Q * ((Q + 2) * A + 1) + 2 * A + 1) * (Q * ((Q + 2) * A + 1) + 2 * A + 1 + 1)
          * (8 * (Q + 1) * iδ))
        + (Q * ((Q + 2) * A + 1) + 2 * A + 1) + 1) * ℓ := by
  unfold loopBudget
  dsimp only
  rw [show 4 * (Ae + 1) / (1 / ic) = 4 * (Ae + 1) * ic by field_simp]
  congr 5
  field_simp


lemma loopΘ_le {Q A ic iξ is : ℝ} (hQ : 1 ≤ Q) (hA : 1 ≤ A) (hic : 1 ≤ ic) (hiξ : 1 ≤ iξ)
    (his : 1 ≤ is) :
    1 ≤ loopΘ Q A (1 / ic) (1 / iξ) (1 / 2 - 1 / is)
      ∧ loopΘ Q A (1 / ic) (1 / iξ) (1 / 2 - 1 / is)
        ≤ 2 ^ 17 * ((ic + iξ) * is ^ 2 + Q ^ 2 * A * ic) := by
  rw [loopΘ_inv]
  have hAc : 1 ≤ A * ic := one_le_mul_of_one_le_of_one_le hA hic
  have hs2 : 1 ≤ is ^ 2 := one_le_pow₀ his
  have hu : 0 ≤ (ic + iξ) := by positivity
  have h1 : (10 * ic + 8 * iξ) * is ^ 2 / 2 ≤ 5 * ((ic + iξ) * is ^ 2) := by nlinarith
  have h2 : 8 * (10 * ic + 8 * iξ) ≤ 80 * ((ic + iξ) * is ^ 2) := by nlinarith
  have h3 : 4608 * Q * (Q + 1) * (4 * (A + 1) * ic) ≤ 73728 * (Q ^ 2 * A * ic) := by
    have : Q * (Q + 1) ≤ 2 * Q ^ 2 := by nlinarith
    have : (A + 1) * ic ≤ 2 * (A * ic) := by nlinarith
    have hQ2 : 0 ≤ Q * (Q + 1) := by positivity
    nlinarith [mul_le_mul this (by assumption : Q * (Q + 1) ≤ 2 * Q ^ 2) hQ2 (by positivity)]
  have h4 : 0 ≤ Q ^ 2 * A * ic := by positivity
  constructor <;> nlinarith

lemma loopΘ_le_pow {Q A ic iξ is X : ℝ} (hQ : 1 ≤ Q) (hA : 1 ≤ A) (hic : 1 ≤ ic)
    (hiξ : 1 ≤ iξ) (his : 1 ≤ is) (hQX : Q ≤ X) (hAX : A ≤ X) (hcX : ic ≤ X) (hξX : iξ ≤ X)
    (hsX : is ≤ X) :
    loopΘ Q A (1 / ic) (1 / iξ) (1 / 2 - 1 / is) ≤ 2 ^ 19 * X ^ 4 := by
  have hX : 1 ≤ X := hQ.trans hQX
  refine (loopΘ_le hQ hA hic hiξ his).2.trans ?_
  have h : (ic + iξ) * is ^ 2 + Q ^ 2 * A * ic ≤ (X + X) * X ^ 2 + X ^ 2 * X * X := by
    gcongr
  have h3 : X ^ 3 ≤ X ^ 4 := pow_le_pow_right₀ hX (by norm_num)
  nlinarith

lemma loopW₀_le {Q A iδ X : ℝ} (hQ : 1 ≤ Q) (hA : 1 ≤ A) (hiδ : 1 ≤ iδ) (hQX : Q ≤ X)
    (hAX : A ≤ X) (hδX : iδ ≤ X) :
    1 ≤ loopW₀ Q A (1 / iδ) ∧ loopW₀ Q A (1 / iδ) ≤ 2 ^ 19 * X ^ 11 := by
  have hX : 1 ≤ X := hQ.trans hQX
  rw [loopW₀_inv]
  set S := Q * ((Q + 2) * A + 1) + 2 * A + 1 with hS
  have hS1 : 1 ≤ S := by
    have : 0 ≤ Q * ((Q + 2) * A + 1) := by positivity
    rw [hS]; linarith
  have hSX : S ≤ 7 * X ^ 3 := by
    have : S ≤ X * ((X + 2 * X) * X + X) + 2 * X + X := by rw [hS]; gcongr <;> linarith
    have h2 : X ^ 2 ≤ X ^ 3 := pow_le_pow_right₀ hX (by norm_num)
    have h1 : X ≤ X ^ 3 := le_self_pow₀ hX (by norm_num)
    nlinarith
  set y := 8 * S * (S + 1) * (8 * (Q + 1) * iδ) with hy
  have hy1 : 1 ≤ y := by
    rw [hy]
    have h1 : 1 ≤ 8 * S * (S + 1) := by nlinarith
    have h2 : 1 ≤ 8 * (Q + 1) * iδ := by nlinarith
    exact one_le_mul_of_one_le_of_one_le h1 h2
  have hyX : y ≤ 2 ^ 13 * X ^ 8 := by
    have : y ≤ 8 * (7 * X ^ 3) * (7 * X ^ 3 + X ^ 3) * (8 * (X + X) * X) := by
      rw [hy]; gcongr <;> nlinarith [one_le_pow₀ (n := 3) hX]
    nlinarith
  have hlog := Real.log_le_self (by linarith : (0 : ℝ) ≤ y)
  have hlog0 : 0 ≤ Real.log y := Real.log_nonneg hy1
  set Wm := 16 / 5 * S * Real.log y + 3 + (Q + 2) * A * (Q + 1) with hWm
  have hWm0 : 0 ≤ Wm := by rw [hWm]; positivity
  have hWmX : Wm ≤ 2 ^ 18 * X ^ 11 := by
    have h1 : 16 / 5 * S * Real.log y ≤ 16 / 5 * (7 * X ^ 3) * (2 ^ 13 * X ^ 8) := by
      gcongr; linarith
    have h2 : (Q + 2) * A * (Q + 1) ≤ (X + 2 * X) * X * (X + X) := by gcongr <;> linarith
    have h3 : X ^ 3 ≤ X ^ 11 := pow_le_pow_right₀ hX (by norm_num)
    have h4 : 1 ≤ X ^ 11 := one_le_pow₀ hX
    rw [hWm]
    nlinarith
  have hsq : Real.sqrt (1 / iδ / (8 * (Q + 1))) ≤ 1 := by
    rw [Real.sqrt_le_one]
    rw [div_le_one (by positivity), div_le_iff₀ (by positivity)]
    nlinarith
  have he : Real.exp 1 ≤ 3 := by have := Real.exp_one_lt_d9; linarith
  have hsq0 : 0 ≤ Real.sqrt (1 / iδ / (8 * (Q + 1))) := Real.sqrt_nonneg _
  constructor
  · have : 0 ≤ Real.exp 1 * Real.sqrt (1 / iδ / (8 * (Q + 1))) * Wm / 2 := by positivity
    linarith
  · have h1 : Real.exp 1 * Real.sqrt (1 / iδ / (8 * (Q + 1))) * Wm ≤ 3 * 1 * Wm := by gcongr
    have h4 : 1 ≤ X ^ 11 := one_le_pow₀ hX
    linarith


lemma loopPool_ge {ℓ Q A Ae iδ ic iξ ip is : ℝ} (hℓ : 1 ≤ ℓ) (hQ : 1 ≤ Q) (hAe : 1 ≤ Ae)
    (hiδ : 1 ≤ iδ) (hic : 1 ≤ ic) (hiξ : 1 ≤ iξ) (hip : 1 ≤ ip) (his : 1 ≤ is) :
    1 ≤ loopPool ℓ Q A Ae (1 / iδ) (1 / ic) (1 / iξ) (1 / ip) (1 / 2 - 1 / is) := by
  rw [loopPool_inv]
  have hΘ := (loopΘ_le hQ hAe hic hiξ his).1
  have h1 : 0 ≤ log (20 * loopΘ Q Ae (1 / ic) (1 / iξ) (1 / 2 - 1 / is)) :=
    Real.log_nonneg (by linarith)
  have hm : 0 ≤ max (log (20 * loopΘ Q Ae (1 / ic) (1 / iξ) (1 / 2 - 1 / is)))
      (log (8 * loopW₀ Q A (1 / iδ) * (8 * (Q + 1) * iδ))) := le_max_of_le_left h1
  have h2 : 0 ≤ log ℓ := Real.log_nonneg hℓ
  have h3 : 0 ≤ log (48 * (Q + 1) * iδ) :=
    Real.log_nonneg (one_le_mul_of_one_le_of_one_le (by nlinarith) hiδ)
  have h4 : 0 ≤ 120 * log (48 * (Q + 1) * iδ) * ip := by positivity
  have h5 : 0 ≤ (50 * (max (log (20 * loopΘ Q Ae (1 / ic) (1 / iξ) (1 / 2 - 1 / is)))
      (log (8 * loopW₀ Q A (1 / iδ) * (8 * (Q + 1) * iδ))) + 2 * log ℓ) * is ^ 2 + 20 * is) :=
    by positivity
  nlinarith

lemma loopPool_le_pow {ℓ Q A Ae iδ ic iξ ip is X : ℝ} (hℓ : 1 ≤ ℓ) (hQ : 1 ≤ Q) (hA : 1 ≤ A)
    (hAe : 1 ≤ Ae) (hiδ : 1 ≤ iδ) (hic : 1 ≤ ic) (hiξ : 1 ≤ iξ) (hip : 1 ≤ ip) (his : 1 ≤ is)
    (hℓX : ℓ ≤ X) (hQX : Q ≤ X) (hAX : A ≤ X) (hAeX : Ae ≤ X) (hδX : iδ ≤ X) (hcX : ic ≤ X)
    (hξX : iξ ≤ X) (hpX : ip ≤ X) (hsX : is ≤ X) :
    loopPool ℓ Q A Ae (1 / iδ) (1 / ic) (1 / iξ) (1 / ip) (1 / 2 - 1 / is)
      ≤ 2 ^ 34 * X ^ 16 := by
  have hX : 1 ≤ X := hQ.trans hQX
  rw [loopPool_inv]
  set Θ := loopΘ Q Ae (1 / ic) (1 / iξ) (1 / 2 - 1 / is)
  set W₀ := loopW₀ Q A (1 / iδ)
  have hΘ1 := (loopΘ_le hQ hAe hic hiξ his).1
  have hΘX := loopΘ_le_pow hQ hAe hic hiξ his hQX hAeX hcX hξX hsX
  obtain ⟨hW1, hWX⟩ := loopW₀_le hQ hA hiδ hQX hAX hδX
  have hm : max (log (20 * Θ)) (log (8 * W₀ * (8 * (Q + 1) * iδ))) ≤ 2 ^ 27 * X ^ 13 := by
    have h1 : log (20 * Θ) ≤ 20 * Θ := Real.log_le_self (by positivity)
    have h2 : log (8 * W₀ * (8 * (Q + 1) * iδ)) ≤ 8 * W₀ * (8 * (Q + 1) * iδ) :=
      Real.log_le_self (by positivity)
    have h3 : 8 * W₀ * (8 * (Q + 1) * iδ) ≤ 8 * (2 ^ 19 * X ^ 11) * (8 * (X + X) * X) := by
      gcongr
    have h4 : X ^ 4 ≤ X ^ 13 := pow_le_pow_right₀ hX (by norm_num)
    refine max_le ?_ ?_ <;> nlinarith
  have hl : log ℓ ≤ X ^ 13 :=
    (Real.log_le_self (by linarith)).trans (hℓX.trans (le_self_pow₀ hX (by norm_num)))
  have hl2 : log (48 * (Q + 1) * iδ) ≤ 16 * (5 * X) * (2 * X) * X := by
    refine (Real.log_le_self (by positivity)).trans ?_
    have : 48 * (Q + 1) * iδ ≤ 48 * (2 * X) * X := by gcongr <;> linarith
    have : 96 * X ^ 2 ≤ 160 * X ^ 3 := by nlinarith [mul_le_mul_of_nonneg_left hX (sq_nonneg X)]
    nlinarith
  have hlp : 0 ≤ log (48 * (Q + 1) * iδ) :=
    Real.log_nonneg (one_le_mul_of_one_le_of_one_le (by nlinarith) hiδ)
  have hm0 : 0 ≤ max (log (20 * Θ)) (log (8 * W₀ * (8 * (Q + 1) * iδ))) :=
    le_max_of_le_left (Real.log_nonneg (by linarith))
  have hl0 : 0 ≤ log ℓ := Real.log_nonneg hℓ
  have hb : (50 * (max (log (20 * Θ)) (log (8 * W₀ * (8 * (Q + 1) * iδ))) + 2 * log ℓ) * is ^ 2
      + 20 * is + 28) ≤ 2 ^ 33 * X ^ 15 := by
    have h1 : (50 * (max (log (20 * Θ)) (log (8 * W₀ * (8 * (Q + 1) * iδ))) + 2 * log ℓ) * is ^ 2)
        ≤ 50 * (2 ^ 27 * X ^ 13 + 2 * X ^ 13) * X ^ 2 := by gcongr
    have h2 : X ≤ X ^ 15 := le_self_pow₀ hX (by norm_num)
    have h3 : 1 ≤ X ^ 15 := one_le_pow₀ hX
    have e : X ^ 13 * X ^ 2 = X ^ 15 := by ring
    nlinarith
  have hb2 : 120 * log (48 * (Q + 1) * iδ) * ip ≤ 120 * (16 * (5 * X) * (2 * X) * X)
      * X := by gcongr
  have h5 : X ^ 4 ≤ X ^ 16 := pow_le_pow_right₀ hX (by norm_num)
  have hb0 : 0 ≤ (50 * (max (log (20 * Θ)) (log (8 * W₀ * (8 * (Q + 1) * iδ))) + 2 * log ℓ)
      * is ^ 2 + 20 * is + 28) := by positivity
  have h6 := mul_le_mul hb hpX (by linarith) (by positivity)
  have e : X ^ 15 * X = X ^ 16 := by ring
  nlinarith


lemma loopPool_le_log {ℓ Q A Ae iδ ic iξ ip is X : ℝ} (hℓ : 1 ≤ ℓ) (hQ : 1 ≤ Q) (hA : 1 ≤ A)
    (hAe : 1 ≤ Ae) (hiδ : 2 ≤ iδ) (hic : 1 ≤ ic) (hiξ : 1 ≤ iξ) (hip : 1 ≤ ip) (his : 1 ≤ is)
    (hX : 2 ≤ X) (hℓX : ℓ ≤ 4 * X) (hQX : Q ≤ X) (hAX : A ≤ X) (hAeX : Ae ≤ X) (hδX : iδ ≤ X)
    (hcX : ic ≤ X) (hξX : iξ ≤ X) (hsX : is ≤ X) :
    loopPool ℓ Q A Ae (1 / iδ) (1 / ic) (1 / iξ) (1 / ip) (1 / 2 - 1 / is)
      ≤ 2400 * log X * ip * is ^ 2 + 1200 * log (Q * iδ) * ip := by
  rw [loopPool_inv]
  set Θ := loopΘ Q Ae (1 / ic) (1 / iξ) (1 / 2 - 1 / is)
  set W₀ := loopW₀ Q A (1 / iδ)
  have hQ1 : 1 ≤ Q := hQ
  have hδ1 : 1 ≤ iδ := by linarith
  have hΘ1 := (loopΘ_le hQ1 hAe hic hiξ his).1
  have hΘX := loopΘ_le_pow hQ1 hAe hic hiξ his hQX hAeX hcX hξX hsX
  obtain ⟨hW1, hWX⟩ := loopW₀_le hQ1 hA hδ1 hQX hAX hδX
  have hlX : log 2 ≤ log X := log_two_le_log hX
  have hlX0 : 0 ≤ log X := Real.log_nonneg (by linarith)
  have h1 : log (20 * Θ) ≤ 39 * log X := by
    have hb : 20 * Θ ≤ 2 ^ 24 * X ^ 4 := by linarith
    have := log_le_mul_log_of_le_pow (24 + 4) (by positivity) (hb.trans (by rw [pow_add]; gcongr))
    push_cast at this
    linarith
  have h2 : log (8 * W₀ * (8 * (Q + 1) * iδ)) ≤ 39 * log X := by
    have hb : 8 * W₀ * (8 * (Q + 1) * iδ) ≤ 8 * (2 ^ 19 * X ^ 11) * (8 * (X + X) * X) := by
      gcongr <;> linarith
    have hb' : 8 * W₀ * (8 * (Q + 1) * iδ) ≤ 2 ^ 26 * X ^ 13 := hb.trans_eq (by ring)
    have := log_le_mul_log_of_le_pow (26 + 13) (by positivity)
      (hb'.trans (by rw [pow_add]; gcongr))
    push_cast at this
    linarith
  have h3 : log ℓ ≤ 3 * log X := by
    have hb : ℓ ≤ 2 ^ 2 * X ^ 1 := by linarith
    have := log_le_mul_log_of_le_pow (2 + 1) (by positivity) (hb.trans (by rw [pow_add]; gcongr))
    push_cast at this
    linarith
  have hm : max (log (20 * Θ)) (log (8 * W₀ * (8 * (Q + 1) * iδ))) ≤ 39 * log X := max_le h1 h2
  have hm0 : 0 ≤ max (log (20 * Θ)) (log (8 * W₀ * (8 * (Q + 1) * iδ))) :=
    le_max_of_le_left (Real.log_nonneg (by linarith))
  have hl0 : 0 ≤ log ℓ := Real.log_nonneg hℓ
  have hs2 : 1 ≤ is ^ 2 := one_le_pow₀ his
  have hb : 50 * (max (log (20 * Θ)) (log (8 * W₀ * (8 * (Q + 1) * iδ))) + 2 * log ℓ) * is ^ 2
      + 20 * is + 28 ≤ 2400 * log X * is ^ 2 := by
    have e1 : 50 * (max (log (20 * Θ)) (log (8 * W₀ * (8 * (Q + 1) * iδ))) + 2 * log ℓ) * is ^ 2
        ≤ 50 * (39 * log X + 2 * (3 * log X)) * is ^ 2 := by gcongr
    have e2 : is ≤ is ^ 2 := by nlinarith
    have hl := Real.log_two_gt_d9
    nlinarith
  have hQδ : 2 ≤ Q * iδ := by nlinarith
  have hc : log (48 * (Q + 1) * iδ) ≤ 10 * log (Q * iδ) := by
    have hb : 48 * (Q + 1) * iδ ≤ 2 ^ 8 * (Q * iδ) ^ 2 := by
      have e1 : 48 * (Q + 1) ≤ 160 * Q ^ 2 := by nlinarith
      have e2 : 48 * (Q + 1) * iδ ≤ 160 * Q ^ 2 * iδ :=
        mul_le_mul_of_nonneg_right e1 (by linarith)
      have e3 : 160 * Q ^ 2 * iδ ≤ 160 * Q ^ 2 * iδ ^ 2 :=
        mul_le_mul_of_nonneg_left (by nlinarith) (by positivity)
      nlinarith
    have := log_le_mul_log_of_le_pow (8 + 2) (by positivity) (hb.trans (by rw [pow_add]; gcongr))
    push_cast at this
    linarith
  have hip0 : 0 ≤ ip := by linarith
  have := mul_le_mul_of_nonneg_right hb hip0
  have := mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left hc (by norm_num : (0 : ℝ) ≤ 120))
    hip0
  nlinarith


set_option maxHeartbeats 1000000 in -- the bounds chain through many `gcongr` steps
lemma claim_pref_le {L Q A Ae iδ ic iξ ip is : ℝ} (hL : 0 ≤ L) (hQ : 1 ≤ Q) (hA : 2 ≤ A)
    (hAAe : A ≤ Ae)
    (hiδ : 2 ≤ iδ) (hic : 1 ≤ ic) (hiξ : 1 ≤ iξ) (hip : 1 ≤ ip) (his : 2 ≤ is) :
    0 ≤ (Q + 1) * (6 * (Q + 2)
        * loopCount (L + Q + 4) Q A Ae (1 / iδ) (1 / ic) (1 / iξ) (1 / ip) (1 / 2 - 1 / is))
      ∧ (Q + 1) * (6 * (Q + 2)
        * loopCount (L + Q + 4) Q A Ae (1 / iδ) (1 / ic) (1 / iξ) (1 / ip) (1 / 2 - 1 / is))
        ≤ 2 ^ 74 * (Q ^ 3 * (L + Q) ^ 2 * log ((L + Q) * Ae * iδ * ic * iξ * ip * is)
          * ((ic ^ 2 + iξ ^ 2) * is ^ 8 + Q ^ 4 * Ae ^ 2 * ic ^ 2 * is ^ 4)) := by
  have hA1 : 1 ≤ A := by linarith
  have hAe1 : 1 ≤ Ae := by linarith
  have hδ1 : 1 ≤ iδ := by linarith
  have hc1 : 1 ≤ ic := by linarith
  have hξ1 : 1 ≤ iξ := by linarith
  have hp1 : 1 ≤ ip := by linarith
  have hs1 : 1 ≤ is := by linarith
  have hQ1 : 1 ≤ Q := hQ
  have hLQ : 1 ≤ L + Q := by linarith
  have hl : ∀ x ∈ [L + Q, Ae, iδ, ic, iξ, ip, is], 1 ≤ x := by
    simp [hAe1, hδ1, hc1, hξ1, hp1, hs1, hLQ]
  have hsub (l : List ℝ) (hs : l.Sublist [L + Q, Ae, iδ, ic, iξ, ip, is]) :
      l.prod ≤ (L + Q) * Ae * iδ * ic * iξ * ip * is := by
    simpa [mul_assoc] using prod_le_prod_of_sublist hs hl
  have hLQA := hsub [L + Q, Ae] (by simp)
  have h8 := hsub [Ae, iδ, is] (by simp)
  set X := (L + Q) * Ae * iδ * ic * iξ * ip * is with hXd
  have hLQX : 2 * (L + Q) ≤ X := by simp only [List.prod_cons, List.prod_nil] at hLQA; nlinarith
  have hX8 : 8 ≤ X := by
    simp only [List.prod_cons, List.prod_nil, mul_one] at h8
    have : 4 ≤ Ae * iδ := by nlinarith [hA.trans hAAe]
    nlinarith
  have hX : 2 ≤ X := by linarith
  have hQX : Q ≤ X := by linarith
  have hAeX : Ae ≤ X := by simpa using hsub [Ae] (by simp)
  have hAX : A ≤ X := hAAe.trans hAeX
  have hδX : iδ ≤ X := by simpa using hsub [iδ] (by simp)
  have hcX : ic ≤ X := by simpa using hsub [ic] (by simp)
  have hξX : iξ ≤ X := by simpa using hsub [iξ] (by simp)
  have hpX : ip ≤ X := by simpa using hsub [ip] (by simp)
  have hsX : is ≤ X := by simpa using hsub [is] (by simp)
  set ℓ := L + Q + 4 with hℓd
  have hℓ1 : 1 ≤ ℓ := by linarith
  have hℓ2 : ℓ ≤ 5 * (L + Q) := by linarith
  have hℓX : ℓ ≤ X := by linarith
  rw [loopCount_inv]
  set Θ := loopΘ Q Ae (1 / ic) (1 / iξ) (1 / 2 - 1 / is)
  set PB := loopPool ℓ Q A Ae (1 / iδ) (1 / ic) (1 / iξ) (1 / ip) (1 / 2 - 1 / is)
  obtain ⟨hΘ1, hΘu⟩ := loopΘ_le hQ1 hAe1 hc1 hξ1 hs1
  have hΘX := loopΘ_le_pow hQ1 hAe1 hc1 hξ1 hs1 hQX hAeX hcX hξX hsX
  have hPB1 : 1 ≤ PB := loopPool_ge hℓ1 hQ1 hAe1 hδ1 hc1 hξ1 hp1 hs1
  have hPBX := loopPool_le_pow hℓ1 hQ1 hA1 hAe1 hδ1 hc1 hξ1 hp1 hs1 hℓX hQX hAX hAeX hδX hcX hξX
    hpX hsX
  set Y := 16 * (Q + 4) * PB * Θ * ℓ * (8 * (Q + 1) * iδ) ^ 2 * ip with hYd
  have hY1 : 1 ≤ Y := by
    have h8 : 1 ≤ (8 * (Q + 1) * iδ) ^ 2 := one_le_pow₀ (by nlinarith)
    have hΘ1' : 1 ≤ Θ := hΘ1
    have hl : ∀ x ∈ [16 * (Q + 4), PB, Θ, ℓ, (8 * (Q + 1) * iδ) ^ 2, ip], 1 ≤ x := by
      simp [hPB1, hΘ1', hℓ1, h8, hp1, show 1 ≤ 16 * (Q + 4) by linarith]
    simpa [hYd, mul_assoc] using pow_length_le_prod zero_le_one hl
  have hYX : Y ≤ 2 ^ 68 * X ^ 27 := by
    have h8 : 8 * (Q + 1) * iδ ≤ 16 * X ^ 2 := by nlinarith
    have : Y ≤ 16 * (5 * X) * (2 ^ 34 * X ^ 16) * (2 ^ 19 * X ^ 4) * X * (16 * X ^ 2) ^ 2 * X := by
      rw [hYd]; gcongr <;> linarith
    have e : 16 * (5 * X) * (2 ^ 34 * X ^ 16) * (2 ^ 19 * X ^ 4) * X * (16 * X ^ 2) ^ 2 * X
        = 5 * 2 ^ 65 * X ^ 27 := by ring
    have : (0 : ℝ) ≤ X ^ 27 := by positivity
    linarith
  have hlY : log Y ≤ 95 * log X := by
    have := log_le_mul_log_of_le_pow (68 + 27) (by linarith) (hYX.trans (by rw [pow_add]; gcongr))
    push_cast at this
    linarith
  have hlY0 : 0 ≤ log Y := Real.log_nonneg hY1
  have hG : Θ ^ 2 * is ^ 4 ≤ 2 ^ 36 * ((ic ^ 2 + iξ ^ 2) * is ^ 8
      + Q ^ 4 * Ae ^ 2 * ic ^ 2 * is ^ 4) := by
    have h1 : Θ ^ 2 ≤ (2 ^ 17 * ((ic + iξ) * is ^ 2 + Q ^ 2 * Ae * ic)) ^ 2 :=
      pow_le_pow_left₀ (by linarith) hΘu 2
    have h2 : ((ic + iξ) * is ^ 2 + Q ^ 2 * Ae * ic) ^ 2
        ≤ 2 * ((ic + iξ) ^ 2 * is ^ 4 + Q ^ 4 * Ae ^ 2 * ic ^ 2) := by
      linarith [sq_nonneg ((ic + iξ) * is ^ 2 - Q ^ 2 * Ae * ic)]
    have h3 : (ic + iξ) ^ 2 ≤ 2 * (ic ^ 2 + iξ ^ 2) := by
      linarith [sq_nonneg (ic - iξ)]
    have hs4 : 0 ≤ is ^ 4 := by positivity
    have hs8 : is ^ 4 * is ^ 4 = is ^ 8 := by ring
    have h4 := mul_le_mul_of_nonneg_right h1 hs4
    have h5 := mul_le_mul_of_nonneg_right h2 hs4
    have h6 := mul_le_mul_of_nonneg_right h3 (by positivity : (0 : ℝ) ≤ is ^ 4 * is ^ 4)
    have h7 : 0 ≤ Q ^ 4 * Ae ^ 2 * ic ^ 2 * is ^ 4 := by positivity
    linarith only [h4, h5, h6, h7]
  have hcnt0 : 0 ≤ 161000 * (Q + 2) * Θ ^ 2 * ℓ ^ 2 * log Y * is ^ 4 := by positivity
  constructor
  · positivity
  · have hlX0 : 0 ≤ log X := Real.log_nonneg (by linarith)
    have e : (Q + 1) * (6 * (Q + 2) * (161000 * (Q + 2) * Θ ^ 2 * ℓ ^ 2 * log Y * is ^ 4))
        = 966000 * ((Q + 1) * (Q + 2) ^ 2 * ℓ ^ 2 * log Y) * (Θ ^ 2 * is ^ 4) := by ring
    rw [e]
    have a1 : Q + 1 ≤ 2 * Q := by linarith only [hQ]
    have a2 : Q + 2 ≤ 3 * Q := by linarith only [hQ]
    have h1 : (Q + 1) * (Q + 2) ^ 2 * ℓ ^ 2 * log Y
        ≤ (2 * Q) * (3 * Q) ^ 2 * (5 * (L + Q)) ^ 2 * (95 * log X) := by
      gcongr
    have hG0 : 0 ≤ (ic ^ 2 + iξ ^ 2) * is ^ 8 + Q ^ 4 * Ae ^ 2 * ic ^ 2 * is ^ 4 := by
      positivity
    have hq0 : 0 ≤ (Q + 1) * (Q + 2) ^ 2 * ℓ ^ 2 * log Y := by positivity
    calc 966000 * ((Q + 1) * (Q + 2) ^ 2 * ℓ ^ 2 * log Y) * (Θ ^ 2 * is ^ 4)
        ≤ 966000 * ((2 * Q) * (3 * Q) ^ 2 * (5 * (L + Q)) ^ 2 * (95 * log X))
          * (2 ^ 36 * ((ic ^ 2 + iξ ^ 2) * is ^ 8
            + Q ^ 4 * Ae ^ 2 * ic ^ 2 * is ^ 4)) := by
          gcongr
      _ = (966000 * 2 * 9 * 25 * 95 * 2 ^ 36) * (Q ^ 3 * (L + Q) ^ 2 * log X
          * ((ic ^ 2 + iξ ^ 2) * is ^ 8 + Q ^ 4 * Ae ^ 2 * ic ^ 2 * is ^ 4)) := by ring
      _ ≤ _ := by gcongr; norm_num


lemma claim_suff_le {L Q A Ae iδ ic iξ ip is : ℝ} (hL : 0 ≤ L) (hQ : 1 ≤ Q) (hA : 2 ≤ A)
    (hAAe : A ≤ Ae)
    (hiδ : 2 ≤ iδ) (hic : 1 ≤ ic) (hiξ : 1 ≤ iξ) (hip : 1 ≤ ip) (his : 2 ≤ is) :
    0 ≤ (Q + 1)
        * (loopPool (L + Q + 4) Q A Ae (1 / iδ) (1 / ic) (1 / iξ) (1 / ip) (1 / 2 - 1 / is) + 1)
      ∧ (Q + 1)
        * (loopPool (L + Q + 4) Q A Ae (1 / iδ) (1 / ic) (1 / iξ) (1 / ip) (1 / 2 - 1 / is) + 1)
        ≤ 2 ^ 13 * (Q * (log ((L + Q) * Ae * iδ * ic * iξ * is) * ip * is ^ 2
          + log (Q * iδ) * ip)) := by
  have hA1 : 1 ≤ A := by linarith
  have hAe1 : 1 ≤ Ae := by linarith
  have hδ1 : 1 ≤ iδ := by linarith
  have hc1 : 1 ≤ ic := by linarith
  have hξ1 : 1 ≤ iξ := by linarith
  have hp1 : 1 ≤ ip := by linarith
  have hs1 : 1 ≤ is := by linarith
  have hQ1 : 1 ≤ Q := hQ
  have hLQ : 1 ≤ L + Q := by linarith
  have hl : ∀ x ∈ [L + Q, Ae, iδ, ic, iξ, is], 1 ≤ x := by
    simp [hAe1, hδ1, hc1, hξ1, hs1, hLQ]
  have hsub (l : List ℝ) (hs : l.Sublist [L + Q, Ae, iδ, ic, iξ, is]) :
      l.prod ≤ (L + Q) * Ae * iδ * ic * iξ * is := by
    simpa [mul_assoc] using prod_le_prod_of_sublist hs hl
  have hLQA := hsub [L + Q, Ae] (by simp)
  have h8 := hsub [Ae, iδ, is] (by simp)
  set X := (L + Q) * Ae * iδ * ic * iξ * is with hXd
  have hLQX : 2 * (L + Q) ≤ X := by simp only [List.prod_cons, List.prod_nil] at hLQA; nlinarith
  have hX8 : 8 ≤ X := by
    simp only [List.prod_cons, List.prod_nil, mul_one] at h8
    have : 4 ≤ Ae * iδ := by nlinarith [hA.trans hAAe]
    nlinarith
  have hX : 2 ≤ X := by linarith
  have hQX : Q ≤ X := by linarith
  have hAeX : Ae ≤ X := by simpa using hsub [Ae] (by simp)
  have hAX : A ≤ X := hAAe.trans hAeX
  have hδX : iδ ≤ X := by simpa using hsub [iδ] (by simp)
  have hcX : ic ≤ X := by simpa using hsub [ic] (by simp)
  have hξX : iξ ≤ X := by simpa using hsub [iξ] (by simp)
  have hsX : is ≤ X := by simpa using hsub [is] (by simp)
  have hℓ1 : 1 ≤ L + Q + 4 := by linarith
  have hℓX : L + Q + 4 ≤ 4 * X := by linarith
  have hP := loopPool_le_log hℓ1 hQ hA1 hAe1 hiδ hc1 hξ1 hp1 hs1 hX hℓX hQX hAX hAeX hδX hcX hξX
    hsX
  have hP1 := loopPool_ge (A := A) hℓ1 hQ1 hAe1 hδ1 hc1 hξ1 hp1 hs1
  have hlX : log 2 ≤ log X := log_two_le_log hX
  have hl2 := Real.log_two_gt_d9
  have hQδ : 2 ≤ Q * iδ := by nlinarith
  have hlQ : 0 ≤ log (Q * iδ) := Real.log_nonneg (by linarith)
  have hps : 1 ≤ ip * is ^ 2 := one_le_mul_of_one_le_of_one_le hp1 (one_le_pow₀ hs1)
  have h1 : 1 ≤ 2 * log X * ip * is ^ 2 := by
    have : 1 ≤ 2 * log X := by linarith
    have e : 2 * log X * ip * is ^ 2 = 2 * log X * (ip * is ^ 2) := by ring
    rw [e]; exact one_le_mul_of_one_le_of_one_le this hps
  have hb0 : 0 ≤ log X * ip * is ^ 2 := by
    have : 0 ≤ log X := by linarith
    positivity
  have hc0 : 0 ≤ log (Q * iδ) * ip := by positivity
  constructor
  · positivity
  · have h2 : loopPool (L + Q + 4) Q A Ae (1 / iδ) (1 / ic) (1 / iξ) (1 / ip) (1 / 2 - 1 / is) + 1
        ≤ 2402 * (log X * ip * is ^ 2) + 1200 * (log (Q * iδ) * ip) := by linarith
    have h3 : (Q + 1) ≤ 2 * Q := by linarith
    calc (Q + 1) * (loopPool (L + Q + 4) Q A Ae (1 / iδ) (1 / ic) (1 / iξ) (1 / ip) (1 / 2 - 1 / is)
          + 1)
        ≤ (2 * Q) * (2402 * (log X * ip * is ^ 2) + 1200 * (log (Q * iδ) * ip)) := by
          gcongr
      _ ≤ _ := by nlinarith

lemma claim_probes_le {L Q A Ae iδ ic : ℝ} (hL : 0 ≤ L) (hQ : 1 ≤ Q) (hA : 2 ≤ A)
    (hAAe : A ≤ Ae) (hiδ : 2 ≤ iδ) (hic : 1 ≤ ic) :
    0 ≤ (Q + 1) * loopBudget (L + Q + 4) Q A Ae (1 / iδ) (1 / ic)
      ∧ (Q + 1) * loopBudget (L + Q + 4) Q A Ae (1 / iδ) (1 / ic)
        ≤ 2 ^ 19 * (Q ^ 3 * A * Ae * (L + Q) * ic * log (Q * A * iδ)) := by
  rw [loopBudget_inv]
  set S := Q * ((Q + 2) * A + 1) + 2 * A + 1 with hS
  set Y := Q * A * iδ with hYd
  have hQA : 2 ≤ Q * A := by nlinarith
  have hY : 4 ≤ Y := by rw [hYd]; nlinarith
  have hS1 : 1 ≤ S := by
    have : 0 ≤ Q * ((Q + 2) * A + 1) := by positivity
    rw [hS]; linarith
  have hSQ : S ≤ 7 * (Q ^ 2 * A) := by
    rw [hS]; nlinarith
  have hQ2A : Q ^ 2 * A ≤ Y ^ 2 := by
    rw [hYd]
    have : Q ^ 2 * A ≤ Q ^ 2 * A ^ 2 := by
      have : A ≤ A ^ 2 := by nlinarith
      exact mul_le_mul_of_nonneg_left this (by positivity)
    nlinarith [mul_le_mul_of_nonneg_left (show 1 ≤ iδ ^ 2 by nlinarith)
      (show 0 ≤ Q ^ 2 * A ^ 2 by positivity)]
  set z := 16 * S * (S + 1) * (8 * (Q + 1) * iδ) with hz
  have hz1 : 1 ≤ z := by
    rw [hz]
    exact one_le_mul_of_one_le_of_one_le (by nlinarith) (by nlinarith)
  have hzY : z ≤ 2 ^ 14 * Y ^ 6 := by
    have hQY : Q + 1 ≤ 2 * Y := by rw [hYd]; nlinarith
    have hδY : iδ ≤ Y := by rw [hYd]; nlinarith
    have : z ≤ 16 * (7 * Y ^ 2) * (7 * Y ^ 2 + Y ^ 2) * (8 * (2 * Y) * Y) := by
      rw [hz]; gcongr <;> nlinarith
    nlinarith
  have hlz : log z ≤ 20 * log Y := by
    have hY2 : 2 ≤ Y := by linarith
    have := log_le_mul_log_of_le_pow (14 + 6) (by linarith) (hzY.trans (by rw [pow_add]; gcongr))
    push_cast at this
    linarith
  have hlz0 : 0 ≤ log z := Real.log_nonneg hz1
  have hlY : log 2 ≤ log Y := log_two_le_log (by linarith)
  have hl2 := Real.log_two_gt_d9
  have hAe2 : 2 ≤ Ae := hA.trans hAAe
  have hK : 36 * S * (4 * (Ae + 1) * ic) * log z + S + 1
      ≤ 40400 * (Q ^ 2 * A * Ae * ic * log Y) := by
    have h1 : 36 * S * (4 * (Ae + 1) * ic) * log z ≤ 36 * (7 * (Q ^ 2 * A)) * (4 * (2 * Ae) * ic)
        * (20 * log Y) := by gcongr; linarith
    have h2 : S + 1 ≤ 8 * (Q ^ 2 * A) := by nlinarith
    have h3 : Q ^ 2 * A ≤ Q ^ 2 * A * Ae * ic := by
      have : 1 ≤ Ae * ic := by nlinarith
      have h0 : 0 ≤ Q ^ 2 * A := by positivity
      nlinarith [mul_le_mul_of_nonneg_left this h0]
    have h4 : 8 * (Q ^ 2 * A * Ae * ic) ≤ 12 * (Q ^ 2 * A * Ae * ic * log Y) := by
      have : 0 ≤ Q ^ 2 * A * Ae * ic := by positivity
      nlinarith
    nlinarith
  have hK0 : 0 ≤ 36 * S * (4 * (Ae + 1) * ic) * log z + S + 1 := by positivity
  constructor
  · positivity
  · have hℓ : L + Q + 4 ≤ 5 * (L + Q) := by linarith
    calc (Q + 1) * ((36 * S * (4 * (Ae + 1) * ic) * log z + S + 1) * (L + Q + 4))
        ≤ (2 * Q) * ((40400 * (Q ^ 2 * A * Ae * ic * log Y)) * (5 * (L + Q))) := by
          gcongr <;> linarith
      _ = 404000 * (Q ^ 3 * A * Ae * (L + Q) * ic * log Y) := by ring
      _ ≤ _ := by
          have : 0 ≤ Q ^ 3 * A * Ae * (L + Q) * ic * log Y := by
            have : 0 ≤ log Y := by linarith
            have : 0 ≤ L + Q := by linarith
            positivity
          exact mul_le_mul_of_nonneg_right (by norm_num) this

lemma claim_heldOut_le {Q iδ iξ : ℝ} (hQ : 1 ≤ Q) (hiδ : 2 ≤ iδ) (hiξ : 1 ≤ iξ) :
    (⌈32 * log (16 * (Q + 1) * (Q + 2) / (1 / iδ)) / (1 / iξ) ^ 2⌉₊ : ℝ)
      ≤ 2 ^ 9 * (log (Q * iδ) * iξ ^ 2) := by
  have e : 32 * log (16 * (Q + 1) * (Q + 2) / (1 / iδ)) / (1 / iξ) ^ 2
      = 32 * log (16 * (Q + 1) * (Q + 2) * iδ) * iξ ^ 2 := by
    rw [show 16 * (Q + 1) * (Q + 2) / (1 / iδ) = 16 * (Q + 1) * (Q + 2) * iδ by field_simp]
    field_simp
  rw [e]
  have hQδ2 : 2 ≤ Q * iδ := by nlinarith
  have hb : 16 * (Q + 1) * (Q + 2) * iδ ≤ 2 ^ 7 * (Q * iδ) ^ 2 := by
    have e1 : 16 * (Q + 1) * (Q + 2) ≤ 128 * Q ^ 2 := by nlinarith
    have e2 := mul_le_mul_of_nonneg_right e1 (by linarith : (0 : ℝ) ≤ iδ)
    have e3 : 128 * Q ^ 2 * iδ ≤ 128 * Q ^ 2 * iδ ^ 2 :=
      mul_le_mul_of_nonneg_left (by nlinarith) (by positivity)
    nlinarith
  have hl := log_le_mul_log_of_le_pow (7 + 2) (by positivity) (hb.trans (by rw [pow_add]; gcongr))
  push_cast at hl
  have hl0 : 0 ≤ log (16 * (Q + 1) * (Q + 2) * iδ) := Real.log_nonneg (by nlinarith)
  have hlQ : 1 / 2 ≤ log (Q * iδ) := half_le_log hQδ2
  have hc := Nat.ceil_lt_add_one (show 0 ≤ 32 * log (16 * (Q + 1) * (Q + 2) * iδ) * iξ ^ 2 by
    positivity)
  have hξ2 : 1 ≤ iξ ^ 2 := one_le_pow₀ hiξ
  nlinarith


/-- The log of the summed rates' coordinates, edges crossed at `Ae + Q`, `1/c = 2Riξ` and `ξ/4`,
against that of `P Ae iδ R iξ ip is`. -/
lemma summed_log_le {P Q Ae iδ R iξ ip is : ℝ} (hQ : 1 ≤ Q) (hQP : Q ≤ P) (hAe : 2 ≤ Ae)
    (hiδ : 2 ≤ iδ) (hR : 1 ≤ R) (hiξ : 1 ≤ iξ) (hip : 1 ≤ ip) (his : 1 ≤ is) :
    0 ≤ log (P * (Ae + Q) * iδ * (2 * R * iξ) * (4 * iξ) * ip * is)
      ∧ log (P * (Ae + Q) * iδ * (2 * R * iξ) * (4 * iξ) * ip * is)
        ≤ 4 * log (P * Ae * iδ * R * iξ * ip * is) := by
  have hP : 1 ≤ P := hQ.trans hQP
  set Y := P * Ae * iδ * R * iξ * ip * is with hYd
  have hW : 4 ≤ Ae * iδ * R * ip * is := by
    have h1 : 4 ≤ Ae * iδ := by nlinarith
    have h2 : 4 ≤ Ae * iδ * R := by nlinarith
    have h3 : 4 ≤ Ae * iδ * R * ip := by nlinarith
    nlinarith
  have hPξ : 1 ≤ P * iξ := one_le_mul_of_one_le_of_one_le hP hiξ
  have hYe : Y = (P * iξ) * (Ae * iδ * R * ip * is) := by rw [hYd]; ring
  have hPY : P * iξ ≤ Y := by rw [hYe]; exact le_mul_of_one_le_right (by linarith) (by linarith)
  have hY4 : 4 ≤ Y := by rw [hYe]; nlinarith
  have hX1 : 1 ≤ P * (Ae + Q) * iδ * (2 * R * iξ) * (4 * iξ) * ip * is := by
    have a1 : 1 ≤ P * (Ae + Q) := one_le_mul_of_one_le_of_one_le hP (by linarith)
    have a2 : 1 ≤ P * (Ae + Q) * iδ := one_le_mul_of_one_le_of_one_le a1 (by linarith)
    have a3 : 1 ≤ P * (Ae + Q) * iδ * (2 * R * iξ) :=
      one_le_mul_of_one_le_of_one_le a2 (by nlinarith)
    have a4 : 1 ≤ P * (Ae + Q) * iδ * (2 * R * iξ) * (4 * iξ) :=
      one_le_mul_of_one_le_of_one_le a3 (by linarith)
    exact one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le a4 hip) his
  have hXY : P * (Ae + Q) * iδ * (2 * R * iξ) * (4 * iξ) * ip * is ≤ Y ^ 4 := by
    have hAQ : Ae + Q ≤ 2 * Ae * P := by nlinarith
    have hB : 0 ≤ P * iδ * (2 * R * iξ) * (4 * iξ) * ip * is := by
      have : 0 ≤ P := by linarith
      have : 0 ≤ iδ := by linarith
      have : 0 ≤ R := by linarith
      have : 0 ≤ iξ := by linarith
      have : 0 ≤ ip := by linarith
      have : 0 ≤ is := by linarith
      positivity
    calc P * (Ae + Q) * iδ * (2 * R * iξ) * (4 * iξ) * ip * is
        = (Ae + Q) * (P * iδ * (2 * R * iξ) * (4 * iξ) * ip * is) := by ring
      _ ≤ (2 * Ae * P) * (P * iδ * (2 * R * iξ) * (4 * iξ) * ip * is) :=
          mul_le_mul_of_nonneg_right hAQ hB
      _ = 16 * Y * (P * iξ) := by rw [hYd]; ring
      _ ≤ 16 * Y * Y := mul_le_mul_of_nonneg_left hPY (by linarith)
      _ = 16 * Y ^ 2 := by ring
      _ ≤ Y ^ 2 * Y ^ 2 := mul_le_mul_of_nonneg_right (by nlinarith) (sq_nonneg Y)
      _ = Y ^ 4 := by ring
  have hl := log_le_mul_log_of_le_pow 4 (by linarith) hXY
  push_cast at hl
  exact ⟨Real.log_nonneg hX1, hl⟩

/-- `summed_log_le` for `loopMassInv`'s log, against that of `Q R Ae iδ iξ ip is`. -/
lemma summed_log_mass_le {Q Ae iδ R iξ ip is : ℝ} (hQ : 1 ≤ Q) (hAe : 2 ≤ Ae) (hiδ : 2 ≤ iδ)
    (hR : 1 ≤ R) (hiξ : 1 ≤ iξ) (hip : 1 ≤ ip) (his : 1 ≤ is) :
    0 ≤ log (Q * (Ae + Q) * iδ * (2 * R * iξ) * ip * is)
      ∧ log (Q * (Ae + Q) * iδ * (2 * R * iξ) * ip * is)
        ≤ 3 * log (Q * R * Ae * iδ * iξ * ip * is) := by
  set Y := Q * R * Ae * iδ * iξ * ip * is with hYd
  have hW : 4 ≤ R * Ae * iδ * iξ * ip * is := by
    have h1 : 2 ≤ R * Ae := by nlinarith
    have h2 : 4 ≤ R * Ae * iδ := by nlinarith
    have h3 : 4 ≤ R * Ae * iδ * iξ := by nlinarith
    have h4 : 4 ≤ R * Ae * iδ * iξ * ip := by nlinarith
    nlinarith
  have hYe : Y = Q * (R * Ae * iδ * iξ * ip * is) := by rw [hYd]; ring
  have hQY : Q ≤ Y := by rw [hYe]; exact le_mul_of_one_le_right (by linarith) (by linarith)
  have hY4 : 4 ≤ Y := by rw [hYe]; nlinarith
  have hX1 : 1 ≤ Q * (Ae + Q) * iδ * (2 * R * iξ) * ip * is := by
    have a1 : 1 ≤ Q * (Ae + Q) := one_le_mul_of_one_le_of_one_le hQ (by linarith)
    have a2 : 1 ≤ Q * (Ae + Q) * iδ := one_le_mul_of_one_le_of_one_le a1 (by linarith)
    have a3 : 1 ≤ Q * (Ae + Q) * iδ * (2 * R * iξ) :=
      one_le_mul_of_one_le_of_one_le a2 (by nlinarith)
    exact one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le a3 hip) his
  have hXY : Q * (Ae + Q) * iδ * (2 * R * iξ) * ip * is ≤ Y ^ 3 := by
    have hAQ : Ae + Q ≤ 2 * Ae * Q := by nlinarith
    have hB : 0 ≤ Q * iδ * (2 * R * iξ) * ip * is := by
      have : 0 ≤ Q := by linarith
      have : 0 ≤ iδ := by linarith
      have : 0 ≤ R := by linarith
      have : 0 ≤ iξ := by linarith
      have : 0 ≤ ip := by linarith
      have : 0 ≤ is := by linarith
      positivity
    calc Q * (Ae + Q) * iδ * (2 * R * iξ) * ip * is
        = (Ae + Q) * (Q * iδ * (2 * R * iξ) * ip * is) := by ring
      _ ≤ (2 * Ae * Q) * (Q * iδ * (2 * R * iξ) * ip * is) :=
          mul_le_mul_of_nonneg_right hAQ hB
      _ = 4 * Q * Y := by rw [hYd]; ring
      _ ≤ 4 * Y * Y := by nlinarith
      _ ≤ Y * Y * Y := mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_right hY4 (by linarith))
          (by linarith)
      _ = Y ^ 3 := by ring
  have hl := log_le_mul_log_of_le_pow 3 (by linarith) hXY
  push_cast at hl
  exact ⟨Real.log_nonneg hX1, hl⟩

set_option maxHeartbeats 1000000 in -- the bounds chain through many products
/-- `claim_pref_le` at the summed rates: edges crossed at `Ae + Q`, `1/c = 2Riξ`, and `ξ/4`. -/
lemma rel_pref_le {L Q A Ae iδ R iξ ip is : ℝ} (hL : 0 ≤ L) (hQ : 1 ≤ Q) (hA : 2 ≤ A)
    (hAAe : A ≤ Ae) (hiδ : 2 ≤ iδ) (hR : 1 ≤ R) (hiξ : 1 ≤ iξ) (hip : 1 ≤ ip) (his : 2 ≤ is) :
    0 ≤ (Q + 1) * (6 * (Q + 2) * loopCount (L + Q + 4) Q A (Ae + Q) (1 / iδ) (1 / (2 * R * iξ))
        (1 / (4 * iξ)) (1 / ip) (1 / 2 - 1 / is))
      ∧ (Q + 1) * (6 * (Q + 2) * loopCount (L + Q + 4) Q A (Ae + Q) (1 / iδ) (1 / (2 * R * iξ))
        (1 / (4 * iξ)) (1 / ip) (1 / 2 - 1 / is))
        ≤ 2 ^ 81 * (Q ^ 3 * (L + Q) ^ 2 * R ^ 2 * log ((L + Q) * Ae * iδ * R * iξ * ip * is)
          * iξ ^ 2 * (is ^ 8 + Q ^ 4 * (Ae + Q) ^ 2 * is ^ 4)) := by
  obtain ⟨h0, h⟩ := claim_pref_le (Ae := Ae + Q) (ic := 2 * R * iξ) (iξ := 4 * iξ) hL hQ hA
    (by linarith) hiδ (by nlinarith) (by linarith) hip his
  refine ⟨h0, h.trans ?_⟩
  obtain ⟨hl0, hl⟩ := summed_log_le (P := L + Q) (is := is) hQ (by linarith) (hA.trans hAAe) hiδ
    hR hiξ hip (by linarith)
  have hY0 : 0 ≤ log ((L + Q) * Ae * iδ * R * iξ * ip * is) := by linarith
  set P := R ^ 2 * iξ ^ 2 with hPd
  have hξ2 : 0 ≤ iξ ^ 2 := sq_nonneg _
  have a2 : (4 * iξ) ^ 2 ≤ 16 * P := by
    rw [hPd]
    have := mul_le_mul_of_nonneg_right (one_le_pow₀ hR : (1 : ℝ) ≤ R ^ 2) hξ2
    nlinarith
  have hs8 : 0 ≤ is ^ 8 := by positivity
  have hP0 : 0 ≤ P := by rw [hPd]; positivity
  have hBr : ((2 * R * iξ) ^ 2 + (4 * iξ) ^ 2) * is ^ 8
      + Q ^ 4 * (Ae + Q) ^ 2 * (2 * R * iξ) ^ 2 * is ^ 4
      ≤ 20 * (P * (is ^ 8 + Q ^ 4 * (Ae + Q) ^ 2 * is ^ 4)) := by
    have b1 : (2 * R * iξ) ^ 2 + (4 * iξ) ^ 2 ≤ 20 * P := by
      have : (2 * R * iξ) ^ 2 = 4 * P := by rw [hPd]; ring
      linarith
    have b2 := mul_le_mul_of_nonneg_right b1 hs8
    have e : Q ^ 4 * (Ae + Q) ^ 2 * (2 * R * iξ) ^ 2 * is ^ 4
        = 4 * (P * (Q ^ 4 * (Ae + Q) ^ 2 * is ^ 4)) := by rw [hPd]; ring
    have b3 : 0 ≤ P * (Q ^ 4 * (Ae + Q) ^ 2 * is ^ 4) := by positivity
    nlinarith
  have hBr0 : 0 ≤ ((2 * R * iξ) ^ 2 + (4 * iξ) ^ 2) * is ^ 8
      + Q ^ 4 * (Ae + Q) ^ 2 * (2 * R * iξ) ^ 2 * is ^ 4 := by positivity
  have hK : 0 ≤ Q ^ 3 * (L + Q) ^ 2 := by positivity
  have hT : 0 ≤ Q ^ 3 * (L + Q) ^ 2 * R ^ 2 * log ((L + Q) * Ae * iδ * R * iξ * ip * is)
      * iξ ^ 2 * (is ^ 8 + Q ^ 4 * (Ae + Q) ^ 2 * is ^ 4) := by positivity
  calc 2 ^ 74 * (Q ^ 3 * (L + Q) ^ 2
        * log ((L + Q) * (Ae + Q) * iδ * (2 * R * iξ) * (4 * iξ) * ip * is)
        * (((2 * R * iξ) ^ 2 + (4 * iξ) ^ 2) * is ^ 8
          + Q ^ 4 * (Ae + Q) ^ 2 * (2 * R * iξ) ^ 2 * is ^ 4))
      ≤ 2 ^ 74 * (Q ^ 3 * (L + Q) ^ 2 * (4 * log ((L + Q) * Ae * iδ * R * iξ * ip * is))
        * (20 * (P * (is ^ 8 + Q ^ 4 * (Ae + Q) ^ 2 * is ^ 4)))) := by gcongr
    _ = 80 * 2 ^ 74 * (Q ^ 3 * (L + Q) ^ 2 * R ^ 2 * log ((L + Q) * Ae * iδ * R * iξ * ip * is)
          * iξ ^ 2 * (is ^ 8 + Q ^ 4 * (Ae + Q) ^ 2 * is ^ 4)) := by
        rw [hPd]; ring
    _ ≤ _ := mul_le_mul_of_nonneg_right (by norm_num) hT

/-- `claim_suff_le` at the summed rates. -/
lemma rel_suff_le {L Q A Ae iδ R iξ ip is : ℝ} (hL : 0 ≤ L) (hQ : 1 ≤ Q) (hA : 2 ≤ A)
    (hAAe : A ≤ Ae) (hiδ : 2 ≤ iδ) (hR : 1 ≤ R) (hiξ : 1 ≤ iξ) (hip : 1 ≤ ip) (his : 2 ≤ is) :
    0 ≤ (Q + 1) * (loopPool (L + Q + 4) Q A (Ae + Q) (1 / iδ) (1 / (2 * R * iξ)) (1 / (4 * iξ))
        (1 / ip) (1 / 2 - 1 / is) + 1)
      ∧ (Q + 1) * (loopPool (L + Q + 4) Q A (Ae + Q) (1 / iδ) (1 / (2 * R * iξ)) (1 / (4 * iξ))
        (1 / ip) (1 / 2 - 1 / is) + 1)
        ≤ 2 ^ 15 * (Q * (log ((L + Q) * Ae * iδ * R * iξ * is) * ip * is ^ 2
          + log (Q * iδ) * ip)) := by
  obtain ⟨h0, h⟩ := claim_suff_le (Ae := Ae + Q) (ic := 2 * R * iξ) (iξ := 4 * iξ) hL hQ hA
    (by linarith) hiδ (by nlinarith) (by linarith) hip his
  refine ⟨h0, h.trans ?_⟩
  obtain ⟨hl0, hl⟩ := summed_log_le (P := L + Q) (ip := 1) (is := is) hQ (by linarith)
    (hA.trans hAAe) hiδ hR hiξ le_rfl (by linarith)
  simp only [mul_one] at hl0 hl
  have hQδ : 0 ≤ log (Q * iδ) := Real.log_nonneg (by nlinarith)
  have hps : 0 ≤ ip * is ^ 2 := by
    have : 0 ≤ ip := by linarith
    positivity
  have e1 : log ((L + Q) * (Ae + Q) * iδ * (2 * R * iξ) * (4 * iξ) * is) * ip * is ^ 2
      ≤ 4 * log ((L + Q) * Ae * iδ * R * iξ * is) * ip * is ^ 2 := by
    have := mul_le_mul_of_nonneg_right hl hps
    linarith
  have hY0 : 0 ≤ log ((L + Q) * Ae * iδ * R * iξ * is) * ip * is ^ 2 := by
    have := mul_nonneg (by linarith : 0 ≤ log ((L + Q) * Ae * iδ * R * iξ * is)) hps
    linarith
  have hc0 : 0 ≤ log (Q * iδ) * ip := by positivity
  have hQ0 : 0 ≤ Q := by linarith
  have := mul_le_mul_of_nonneg_left e1 hQ0
  nlinarith

/-- `claim_probes_le` at the summed rates. -/
lemma rel_probes_le {L Q A Ae iδ R iξ : ℝ} (hL : 0 ≤ L) (hQ : 1 ≤ Q) (hA : 2 ≤ A)
    (hAAe : A ≤ Ae) (hiδ : 2 ≤ iδ) (hR : 1 ≤ R) (hiξ : 1 ≤ iξ) :
    0 ≤ (Q + 1) * loopBudget (L + Q + 4) Q A (Ae + Q) (1 / iδ) (1 / (2 * R * iξ))
      ∧ (Q + 1) * loopBudget (L + Q + 4) Q A (Ae + Q) (1 / iδ) (1 / (2 * R * iξ))
        ≤ 2 ^ 20 * (Q ^ 3 * A * (Ae + Q) * (L + Q) * R * iξ * log (Q * A * iδ)) := by
  obtain ⟨h0, h⟩ := claim_probes_le (Ae := Ae + Q) (ic := 2 * R * iξ) hL hQ hA (by linarith) hiδ
    (by nlinarith)
  exact ⟨h0, h.trans_eq (by ring)⟩

set_option maxHeartbeats 8000000 in -- `loopMassTerms` bounds about twenty nested quantities
/-- `loopMassTerms` in the inverted coordinates, `iξ ≤ 2 ic` and `A ≤ Ae`. -/
lemma claim_terms_le {Q A Ae iδ ic iξ ip is : ℝ} {nQ nA : ℕ} (hQ : 1 ≤ Q) (hA : 2 ≤ A)
    (hAAe : A ≤ Ae) (hiδ : 2 ≤ iδ) (hic : 1 ≤ ic) (hiξ : 1 ≤ iξ) (hξc : iξ ≤ 2 * ic) (hip : 1 ≤ ip)
    (his : 2 ≤ is) (hnQ : (nQ : ℝ) ≤ Q) (hnA : (nA : ℝ) ≤ A) :
    (loopMassTerms Q A Ae nQ nA (1 / iδ) (1 / ic) (1 / iξ) (1 / ip) (1 / 2 - 1 / is)).1
        ≤ 2 ^ 123 * (Ae * ic * Q ^ 3 * (ic ^ 3 * (is ^ 11 + Q ^ 6 * Ae ^ 3 * is ^ 5)
          + iδ * ic ^ 2 * (is ^ 4 + Q ^ 4 * Ae ^ 2) + iδ * ip ^ 2 * is ^ 2)
          * log (Q * Ae * iδ * ic * ip * is))
      ∧ (loopMassTerms Q A Ae nQ nA (1 / iδ) (1 / ic) (1 / iξ) (1 / ip) (1 / 2 - 1 / is)).2.1
        ≤ 2 ^ 80 * (Q ^ 3 * iξ ^ 2 * ic ^ 2 * iδ * (is ^ 8 + Q ^ 4 * Ae ^ 2 * is ^ 4)
          * log (Q * Ae * iδ * ic * ip * is) ^ 2)
      ∧ (loopMassTerms Q A Ae nQ nA (1 / iδ) (1 / ic) (1 / iξ) (1 / ip) (1 / 2 - 1 / is)).2.2
        ≤ 2 ^ 40 * (Q ^ 5 * A ^ 2 * Ae ^ 2 * ic ^ 2 * iδ
          * log (Q * Ae * iδ * ic * ip * is) ^ 2) := by
  unfold loopMassTerms
  extract_lets δ' s S n aε Θ K₁ W₀ P₀ R₀ E₁ Kc E E₂ E₃
  have hΘe : Θ = loopΘ Q Ae (1 / ic) (1 / iξ) (1 / 2 - 1 / is) := rfl
  have hW₀e : W₀ = loopW₀ Q A (1 / iδ) := rfl
  have hP₀e : P₀ = loopPool (exp 1) Q A Ae (1 / iδ) (1 / ic) (1 / iξ) (1 / ip)
      (1 / 2 - 1 / is) := by
    unfold loopPool
    rw [Real.log_exp, mul_one]
  have hδ' : δ' = 1 / iδ / (8 * (Q + 1)) := rfl
  have hs : s = 1 / 2 - (1 / 2 - 1 / is) := rfl
  have hS : S = Q * ((Q + 2) * A + 1) + 2 * A + 1 := rfl
  have hn : n = ⌈32 * log (16 * (Q + 1) * (Q + 2) / (1 / iδ)) / (1 / iξ) ^ 2⌉₊ := rfl
  have haε : aε = 4 * (Ae + 1) / (1 / ic) := rfl
  have hK₁ : K₁ = 36 * S * aε * log (16 * S * (S + 1) / δ') + S + 1 := rfl
  have hR₀ : R₀ = 161000 * (Q + 2) * Θ ^ 2
      * (log (16 * (Q + 4) * P₀ * Θ / (δ' ^ 2 * (1 / ip))) + 2) / s ^ 4 := rfl
  have hE₁ : E₁ = (Q + 3) ^ 2 * (2 ^ 20 * Θ * (R₀ + 1) / s
      + 2 ^ 36 * Θ ^ 2 * (13 + 2 * Real.logb 2 (R₀ + 1)) / δ'
      + 2 ^ 16 * (P₀ + 2) / (1 / ip * δ')) := rfl
  have hKc : Kc = (1 + (Q + 1) ^ 3 * A) ^ nQ * (Q + 3) ^ ((nQ + 2) * nA) := rfl
  have hE : E = 1536 * aε * E₁ + 1536 * log (16 * (Q + 1) * Kc / δ') * aε := rfl
  have hE₂ : E₂ = 4 * (Q + 1) * n * (6 * (Q + 2) * R₀ + K₁ + n) / (1 / iδ) := rfl
  have hE₃ : E₃ = 16 * K₁ ^ 2 / δ' := rfl
  clear_value δ' s S n aε Θ K₁ W₀ P₀ R₀ E₁ Kc E E₂ E₃
  have hA1 : 1 ≤ A := by linarith
  have hAe1 : 1 ≤ Ae := by linarith
  have hAe2 : 2 ≤ Ae := hA.trans hAAe
  have hδ1 : 1 ≤ iδ := by linarith
  have hp1 : 1 ≤ ip := hip
  have hs1 : 1 ≤ is := by linarith
  have hl : ∀ x ∈ [Q, Ae, iδ, ic, ip, is], 1 ≤ x := by
    simp [hQ, hAe1, hδ1, hic, hp1, hs1]
  have hsub (l : List ℝ) (hs : l.Sublist [Q, Ae, iδ, ic, ip, is]) :
      l.prod ≤ Q * Ae * iδ * ic * ip * is := by
    simpa [mul_assoc] using prod_le_prod_of_sublist hs hl
  have h8 := hsub [Ae, iδ, is] (by simp)
  set X := Q * Ae * iδ * ic * ip * is with hXd
  have hQX : Q ≤ X := by simpa using hsub [Q] (by simp)
  have hAeX : Ae ≤ X := by simpa using hsub [Ae] (by simp)
  have hAX : A ≤ X := hAAe.trans hAeX
  have hδX : iδ ≤ X := by simpa using hsub [iδ] (by simp)
  have hcX : ic ≤ X := by simpa using hsub [ic] (by simp)
  have hξX : iξ ≤ X := by
    have h := hsub [iδ, ic] (by simp)
    simp only [List.prod_cons, List.prod_nil, mul_one] at h
    nlinarith only [h, hξc, hiδ, hic]
  have hpX : ip ≤ X := by simpa using hsub [ip] (by simp)
  have hsX : is ≤ X := by simpa using hsub [is] (by simp)
  clear_value X
  have hX : 8 ≤ X := by
    simp only [List.prod_cons, List.prod_nil, mul_one] at h8
    have : 4 ≤ Ae * iδ := by nlinarith only [hAe2, hiδ]
    nlinarith only [h8, this, his]
  have hX1 : 1 ≤ X := by linarith
  have hpow : ∀ {a b : ℕ}, a ≤ b → X ^ a ≤ X ^ b := fun h => pow_le_pow_right₀ hX1 h
  have hXp : ∀ a : ℕ, 0 ≤ X ^ a := fun a => by positivity
  have hX1p : ∀ a : ℕ, 1 ≤ X ^ a := fun a => one_le_pow₀ hX1
  -- `log X`, and logs of powers of `X`
  set lX := log X with hlXd
  have hl2 := Real.log_two_gt_d9
  have hlX2 : 2 ≤ lX := by
    rw [hlXd, Real.le_log_iff_exp_le (by linarith)]
    have e2 : Real.exp 2 = Real.exp 1 * Real.exp 1 := by rw [← Real.exp_add]; norm_num
    have := Real.exp_one_lt_d9
    have hex : 0 ≤ Real.exp 1 := (Real.exp_pos 1).le
    nlinarith only [e2, this, hex, hX]
  have hlXp : ∀ {y : ℝ} (a : ℕ), 0 < y → y ≤ X ^ a → log y ≤ a * lX := fun a hy h =>
    log_le_mul_log_of_le_pow a hy h
  have h2X : (2 : ℝ) ^ 3 ≤ X := by norm_num; linarith
  have h2pow : ∀ a : ℕ, (2 : ℝ) ^ (3 * a) ≤ X ^ a := fun a => by
    rw [pow_mul]; exact pow_le_pow_left₀ (by norm_num) h2X a
  -- the rates
  set D := 8 * (Q + 1) * iδ with hDd
  have hD1 : 1 ≤ D := by rw [hDd]; nlinarith only [hQ, hδ1]
  have hDQ : D ≤ 16 * (Q * iδ) := by rw [hDd]; nlinarith only [hQ, hδ1]
  clear_value D
  have hδ'0 : 0 < δ' := by rw [hδ']; positivity
  have hδ'D : ∀ x, x / δ' = x * D := fun x => by rw [hδ', hDd]; field_simp
  have hsis : s = 1 / is := by rw [hs]; ring
  have hsd : ∀ x, x / s = x * is := fun x => by rw [hsis]; field_simp
  have haε' : aε = 4 * (Ae + 1) * ic := by rw [haε]; field_simp
  have haε6 : aε ≤ 6 * (Ae * ic) := by rw [haε']; nlinarith only [hAe2, hic]
  have haε1 : 1 ≤ aε := by rw [haε']; nlinarith only [hAe1, hic]
  -- `Θ`
  obtain ⟨hΘ1, hΘu⟩ := loopΘ_le hQ hAe1 hic hiξ hs1
  rw [← hΘe] at hΘ1 hΘu
  have hΘ : Θ ≤ 2 ^ 18 * (ic * (is ^ 2 + Q ^ 2 * Ae)) := by
    rw [hΘe, loopΘ_inv]
    have hs4 : 4 ≤ is ^ 2 := by nlinarith only [his]
    have hc0 : 0 ≤ ic := by linarith
    have a : 10 * ic + 8 * iξ ≤ 26 * ic := by linarith
    have t1 : (10 * ic + 8 * iξ) * is ^ 2 / 2 ≤ 13 * (ic * is ^ 2) := by nlinarith only [a, hs4]
    have t2 : 8 * (10 * ic + 8 * iξ) ≤ 52 * (ic * is ^ 2) := by nlinarith only [a, hs4, hc0]
    have t3 : 4608 * Q * (Q + 1) * (4 * (Ae + 1) * ic) ≤ 2 ^ 17 * (Q ^ 2 * Ae * ic) := by
      have h1 : Q * (Q + 1) ≤ 2 * Q ^ 2 := by nlinarith only [hQ]
      have h2 : (Ae + 1) * ic ≤ 2 * (Ae * ic) := by nlinarith only [hAe1, hc0]
      have h3 : 0 ≤ Q * (Q + 1) := by positivity
      nlinarith only [mul_le_mul h1 h2 (by positivity) (by positivity), h3,
        (by positivity : 0 ≤ Q ^ 2 * Ae * ic)]
    have t4 : 0 ≤ Q ^ 2 * Ae * ic := by positivity
    have t5 : 0 ≤ ic * is ^ 2 := by positivity
    nlinarith only [t1, t2, t3, t4, t5]
  have hΘ0 : 0 ≤ Θ := by linarith
  have hab0 : 0 ≤ is ^ 2 := by positivity
  have hab1 : 0 ≤ Q ^ 2 * Ae := by positivity
  have hΘ2 : Θ ^ 2 ≤ 2 ^ 37 * (ic ^ 2 * (is ^ 4 + Q ^ 4 * Ae ^ 2)) := by
    have h1 : Θ ^ 2 ≤ (2 ^ 18 * (ic * (is ^ 2 + Q ^ 2 * Ae))) ^ 2 := pow_le_pow_left₀ hΘ0 hΘ 2
    have h2 : (is ^ 2 + Q ^ 2 * Ae) ^ 2 ≤ 2 * (is ^ 4 + Q ^ 4 * Ae ^ 2) := by
      nlinarith only [sq_nonneg (is ^ 2 - Q ^ 2 * Ae)]
    have h3 := mul_le_mul_of_nonneg_left h2 (by positivity : (0 : ℝ) ≤ 2 ^ 36 * ic ^ 2)
    calc Θ ^ 2 ≤ (2 ^ 18 * (ic * (is ^ 2 + Q ^ 2 * Ae))) ^ 2 := h1
      _ = 2 ^ 36 * ic ^ 2 * (is ^ 2 + Q ^ 2 * Ae) ^ 2 := by ring
      _ ≤ 2 ^ 36 * ic ^ 2 * (2 * (is ^ 4 + Q ^ 4 * Ae ^ 2)) := h3
      _ = _ := by ring
  have hΘ3 : Θ ^ 3 ≤ 2 ^ 56 * (ic ^ 3 * (is ^ 6 + Q ^ 6 * Ae ^ 3)) := by
    have h1 : Θ ^ 3 ≤ (2 ^ 18 * (ic * (is ^ 2 + Q ^ 2 * Ae))) ^ 3 := pow_le_pow_left₀ hΘ0 hΘ 3
    have h2 : (is ^ 2 + Q ^ 2 * Ae) ^ 3 ≤ 4 * (is ^ 6 + Q ^ 6 * Ae ^ 3) := by
      nlinarith only [mul_nonneg (add_nonneg hab0 hab1) (sq_nonneg (is ^ 2 - Q ^ 2 * Ae))]
    have h3 := mul_le_mul_of_nonneg_left h2 (by positivity : (0 : ℝ) ≤ 2 ^ 54 * ic ^ 3)
    calc Θ ^ 3 ≤ (2 ^ 18 * (ic * (is ^ 2 + Q ^ 2 * Ae))) ^ 3 := h1
      _ = 2 ^ 54 * ic ^ 3 * (is ^ 2 + Q ^ 2 * Ae) ^ 3 := by ring
      _ ≤ 2 ^ 54 * ic ^ 3 * (4 * (is ^ 6 + Q ^ 6 * Ae ^ 3)) := h3
      _ = _ := by ring
  have hΘX := loopΘ_le_pow hQ hAe1 hic hiξ hs1 hQX hAeX hcX hξX hsX
  rw [← hΘe] at hΘX
  -- `S`
  have hS1 : 1 ≤ S := by
    have : 0 ≤ Q * ((Q + 2) * A + 1) := by positivity
    rw [hS]; linarith
  have hQA1 : Q ≤ Q * A := le_mul_of_one_le_right (by linarith) hA1
  have hQA2 : A ≤ Q * A := le_mul_of_one_le_left (by linarith) hQ
  have hQA3 : Q * A ≤ Q ^ 2 * A := by
    have : Q * A * 1 ≤ Q * A * Q := mul_le_mul_of_nonneg_left hQ (by positivity)
    linarith [show Q * A * Q = Q ^ 2 * A by ring]
  have hQA4 : 1 ≤ Q * A := one_le_mul_of_one_le_of_one_le hQ hA1
  have hSQ : S ≤ 7 * (Q ^ 2 * A) := by rw [hS]; linarith
  have hS8 : S + 1 ≤ 8 * (Q ^ 2 * A) := by linarith
  -- `P₀`
  have hex : 1 ≤ exp 1 := by linarith [Real.add_one_le_exp (1 : ℝ)]
  have he1 : exp 1 ≤ 3 := by have := Real.exp_one_lt_d9; linarith
  have hP₀1 : 1 ≤ P₀ := by
    rw [hP₀e]; exact loopPool_ge hex hQ hAe1 hδ1 hic hiξ hp1 hs1
  have hP₀X : P₀ ≤ 2 ^ 34 * X ^ 16 := by
    rw [hP₀e]
    exact loopPool_le_pow hex hQ hA1 hAe1 hδ1 hic hiξ hp1 hs1 (by linarith) hQX hAX hAeX hδX hcX
      hξX hpX hsX
  have hlQδ : log (Q * iδ) ≤ lX := by
    rw [hlXd]
    refine Real.log_le_log (by positivity) ?_
    simpa [mul_assoc, mul_comm, mul_left_comm] using hsub [Q, iδ] (by simp)
  have hP₀ : P₀ ≤ 3600 * lX * ip * is ^ 2 := by
    have h := loopPool_le_log hex hQ hA1 hAe1 hiδ hic hiξ hp1 hs1 (by linarith) (by linarith) hQX
      hAX hAeX hδX hcX hξX hsX
    rw [← hP₀e] at h
    have hs2 : 1 ≤ is ^ 2 := one_le_pow₀ hs1
    have h2 : 1200 * log (Q * iδ) * ip ≤ 1200 * lX * ip * is ^ 2 := by
      have : 1200 * log (Q * iδ) * ip ≤ 1200 * lX * ip := by gcongr
      have : 1200 * lX * ip ≤ 1200 * lX * ip * is ^ 2 :=
        le_mul_of_one_le_right (by positivity) hs2
      linarith
    linarith
  -- the prefix count
  set Z := 16 * (Q + 4) * P₀ * Θ / (δ' ^ 2 * (1 / ip)) with hZd
  have hZe : Z = 16 * (Q + 4) * P₀ * Θ * D ^ 2 * ip := by
    rw [hZd, show δ' ^ 2 * (1 / ip) = δ' * δ' * (1 / ip) by ring, ← div_div, ← div_div, hδ'D,
      hδ'D]
    field_simp
  clear_value Z
  have hZ1 : 1 ≤ Z := by
    rw [hZe]
    have h1 : 1 ≤ 16 * (Q + 4) * P₀ * Θ :=
      one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le (by linarith) hP₀1) hΘ1
    exact one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le h1 (one_le_pow₀ hD1)) hp1
  have hZX : Z ≤ X ^ 49 := by
    have : Z ≤ 16 * (5 * X) * (2 ^ 34 * X ^ 16) * (2 ^ 19 * X ^ 4) * (16 * X ^ 2) ^ 2 * X := by
      rw [hZe]
      have hDX : D ≤ 16 * X ^ 2 := by
        have : Q * iδ ≤ X ^ 2 := by nlinarith only [hQX, hδX, hQ, hδ1]
        linarith
      gcongr <;> linarith
    have e : 16 * (5 * X) * (2 ^ 34 * X ^ 16) * (2 ^ 19 * X ^ 4) * (16 * X ^ 2) ^ 2 * X
        = 5 * 2 ^ 65 * X ^ 26 := by ring
    have h23 : (2 : ℝ) ^ 68 ≤ X ^ 23 := by
      calc (2 : ℝ) ^ 68 ≤ 2 ^ (3 * 23) := pow_le_pow_right₀ (by norm_num) (by norm_num)
        _ ≤ X ^ 23 := h2pow 23
    have h5 : (5 : ℝ) * 2 ^ 65 ≤ 2 ^ 68 := by norm_num
    calc Z ≤ 16 * (5 * X) * (2 ^ 34 * X ^ 16) * (2 ^ 19 * X ^ 4) * (16 * X ^ 2) ^ 2 * X := this
      _ = 5 * 2 ^ 65 * X ^ 26 := e
      _ ≤ X ^ 23 * X ^ 26 := mul_le_mul_of_nonneg_right (h5.trans h23) (hXp 26)
      _ = X ^ 49 := by ring
  have hlZ : log Z + 2 ≤ 50 * lX := by
    have := hlXp 49 (by linarith) hZX
    push_cast at this
    linarith
  have hlZ0 : 0 ≤ log Z := Real.log_nonneg hZ1
  have hR₀e : R₀ = 161000 * (Q + 2) * Θ ^ 2 * (log Z + 2) * is ^ 4 := by
    rw [hR₀, hsis]; field_simp
  have hR₀0 : 0 ≤ R₀ := by rw [hR₀e]; positivity
  have hlX0 : 0 ≤ lX := by linarith
  set Rs := Q * Θ ^ 2 * is ^ 4 * lX with hRsd
  have hRs1 : 1 ≤ Rs := by
    rw [hRsd]
    have : 1 ≤ Q * Θ ^ 2 * is ^ 4 :=
      one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le hQ (one_le_pow₀ hΘ1))
        (one_le_pow₀ hs1)
    exact one_le_mul_of_one_le_of_one_le this (by linarith)
  have hR₀ : R₀ ≤ 2 ^ 25 * Rs := by
    rw [hR₀e, hRsd]
    have hq : Q + 2 ≤ 3 * Q := by linarith
    calc 161000 * (Q + 2) * Θ ^ 2 * (log Z + 2) * is ^ 4
        ≤ 161000 * (3 * Q) * Θ ^ 2 * (50 * lX) * is ^ 4 := by gcongr
      _ = 161000 * 3 * 50 * (Q * Θ ^ 2 * is ^ 4 * lX) := by ring
      _ ≤ 2 ^ 25 * (Q * Θ ^ 2 * is ^ 4 * lX) := by
          have : 0 ≤ Q * Θ ^ 2 * is ^ 4 * lX := by positivity
          exact mul_le_mul_of_nonneg_right (by norm_num) this
  have hR₀1 : R₀ + 1 ≤ 2 ^ 26 * Rs := by linarith
  have hR₀X : R₀ + 1 ≤ X ^ 36 := by
    have hRsX : Rs ≤ 2 ^ 38 * X ^ 14 := by
      rw [hRsd]
      have hlXX : lX ≤ X := by rw [hlXd]; exact Real.log_le_self (by linarith)
      calc Q * Θ ^ 2 * is ^ 4 * lX ≤ X * (2 ^ 19 * X ^ 4) ^ 2 * X ^ 4 * X := by gcongr
        _ = 2 ^ 38 * X ^ 14 := by ring
    have hK : (2 : ℝ) ^ 26 * 2 ^ 38 ≤ X ^ 22 := by
      have : (2 : ℝ) ^ 26 * 2 ^ 38 ≤ 2 ^ (3 * 22) := by norm_num
      exact this.trans (h2pow 22)
    calc R₀ + 1 ≤ 2 ^ 26 * Rs := hR₀1
      _ ≤ 2 ^ 26 * (2 ^ 38 * X ^ 14) := by gcongr
      _ = 2 ^ 26 * 2 ^ 38 * X ^ 14 := by ring
      _ ≤ X ^ 22 * X ^ 14 := mul_le_mul_of_nonneg_right hK (hXp 14)
      _ = X ^ 36 := by ring
  have hlb : 13 + 2 * Real.logb 2 (R₀ + 1) ≤ 120 * lX := by
    have hlR := hlXp 36 (by linarith) hR₀X
    push_cast at hlR
    have : Real.logb 2 (R₀ + 1) ≤ 53 * lX := by
      rw [Real.logb, div_le_iff₀ (Real.log_pos (by norm_num))]
      have := mul_le_mul_of_nonneg_left hl2.le hlX0
      linarith
    linarith
  have hlb0 : 0 ≤ Real.logb 2 (R₀ + 1) := Real.logb_nonneg (by norm_num) (by linarith)
  -- `E₁`, against `B₁`
  set B₁ := ic ^ 3 * (is ^ 11 + Q ^ 6 * Ae ^ 3 * is ^ 5) + iδ * ic ^ 2 * (is ^ 4 + Q ^ 4 * Ae ^ 2)
    + iδ * ip ^ 2 * is ^ 2 with hB₁d
  have hB0 : 0 ≤ ic ^ 3 * (is ^ 11 + Q ^ 6 * Ae ^ 3 * is ^ 5) := by positivity
  have hB1 : 0 ≤ iδ * ic ^ 2 * (is ^ 4 + Q ^ 4 * Ae ^ 2) := by positivity
  have hB2 : 0 ≤ iδ * ip ^ 2 * is ^ 2 := by positivity
  have hE₁e : E₁ = (Q + 3) ^ 2 * (2 ^ 20 * Θ * (R₀ + 1) * is
      + 2 ^ 36 * Θ ^ 2 * (13 + 2 * Real.logb 2 (R₀ + 1)) * D + 2 ^ 16 * (P₀ + 2) * ip * D) := by
    rw [hE₁, hsd, hδ'D, show 2 ^ 16 * (P₀ + 2) / (1 / ip * δ') = 2 ^ 16 * (P₀ + 2) * ip / δ' by
      field_simp, hδ'D]
  have hE₁ : E₁ ≤ 2 ^ 107 * (Q ^ 3 * B₁ * lX) := by
    rw [hE₁e]
    have hq : (Q + 3) ^ 2 ≤ 16 * Q ^ 2 := by
      have : (Q + 3) ^ 2 ≤ (4 * Q) ^ 2 := pow_le_pow_left₀ (by linarith) (by linarith) 2
      linarith [show (4 * Q) ^ 2 = 16 * Q ^ 2 by ring]
    have ta : 2 ^ 20 * Θ * (R₀ + 1) * is
        ≤ 2 ^ 102 * (Q * (ic ^ 3 * (is ^ 11 + Q ^ 6 * Ae ^ 3 * is ^ 5)) * lX) := by
      have h1 : 2 ^ 20 * Θ * (R₀ + 1) * is ≤ 2 ^ 20 * Θ * (2 ^ 26 * Rs) * is := by gcongr
      have e : 2 ^ 20 * Θ * (2 ^ 26 * Rs) * is = 2 ^ 46 * (Q * Θ ^ 3 * is ^ 5 * lX) := by
        rw [hRsd]; ring
      have h2 : Q * Θ ^ 3 * is ^ 5 * lX
          ≤ Q * (2 ^ 56 * (ic ^ 3 * (is ^ 6 + Q ^ 6 * Ae ^ 3))) * is ^ 5 * lX := by gcongr
      have e2 : Q * (2 ^ 56 * (ic ^ 3 * (is ^ 6 + Q ^ 6 * Ae ^ 3))) * is ^ 5 * lX
          = 2 ^ 56 * (Q * (ic ^ 3 * (is ^ 11 + Q ^ 6 * Ae ^ 3 * is ^ 5)) * lX) := by ring
      have h3 := mul_le_mul_of_nonneg_left h2 (by norm_num : (0 : ℝ) ≤ 2 ^ 46)
      rw [e2] at h3
      rw [e] at h1
      linarith
    have tb : 2 ^ 36 * Θ ^ 2 * (13 + 2 * Real.logb 2 (R₀ + 1)) * D
        ≤ 2 ^ 84 * (Q * (iδ * ic ^ 2 * (is ^ 4 + Q ^ 4 * Ae ^ 2)) * lX) := by
      calc 2 ^ 36 * Θ ^ 2 * (13 + 2 * Real.logb 2 (R₀ + 1)) * D
          ≤ 2 ^ 36 * (2 ^ 37 * (ic ^ 2 * (is ^ 4 + Q ^ 4 * Ae ^ 2))) * (120 * lX)
            * (16 * (Q * iδ)) := by gcongr
        _ = 2 ^ 73 * 1920 * (Q * (iδ * ic ^ 2 * (is ^ 4 + Q ^ 4 * Ae ^ 2)) * lX) := by ring
        _ ≤ _ := by
            have : 0 ≤ Q * (iδ * ic ^ 2 * (is ^ 4 + Q ^ 4 * Ae ^ 2)) * lX := by positivity
            exact mul_le_mul_of_nonneg_right (by norm_num) this
    have tc : 2 ^ 16 * (P₀ + 2) * ip * D ≤ 2 ^ 32 * (Q * (iδ * ip ^ 2 * is ^ 2) * lX) := by
      have hP : P₀ + 2 ≤ 3602 * lX * ip * is ^ 2 := by
        have : 1 ≤ lX * ip * is ^ 2 :=
          one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le (by linarith) hp1)
            (one_le_pow₀ hs1)
        linarith
      calc 2 ^ 16 * (P₀ + 2) * ip * D ≤ 2 ^ 16 * (3602 * lX * ip * is ^ 2) * ip * (16 * (Q * iδ)) :=
            by gcongr
        _ = 2 ^ 20 * 3602 * (Q * (iδ * ip ^ 2 * is ^ 2) * lX) := by ring
        _ ≤ _ := by
            have : 0 ≤ Q * (iδ * ip ^ 2 * is ^ 2) * lX := by positivity
            exact mul_le_mul_of_nonneg_right (by norm_num) this
    have hsum : 2 ^ 20 * Θ * (R₀ + 1) * is + 2 ^ 36 * Θ ^ 2 * (13 + 2 * Real.logb 2 (R₀ + 1)) * D
        + 2 ^ 16 * (P₀ + 2) * ip * D ≤ 2 ^ 103 * (Q * B₁ * lX) := by
      have e : Q * B₁ * lX = Q * (ic ^ 3 * (is ^ 11 + Q ^ 6 * Ae ^ 3 * is ^ 5)) * lX
          + Q * (iδ * ic ^ 2 * (is ^ 4 + Q ^ 4 * Ae ^ 2)) * lX
          + Q * (iδ * ip ^ 2 * is ^ 2) * lX := by rw [hB₁d]; ring
      have h1 : 0 ≤ Q * (ic ^ 3 * (is ^ 11 + Q ^ 6 * Ae ^ 3 * is ^ 5)) * lX := by positivity
      have h2 : 0 ≤ Q * (iδ * ic ^ 2 * (is ^ 4 + Q ^ 4 * Ae ^ 2)) * lX := by positivity
      have h3 : 0 ≤ Q * (iδ * ip ^ 2 * is ^ 2) * lX := by positivity
      rw [e]
      nlinarith only [ta, tb, tc, h1, h2, h3]
    have hsum0 : 0 ≤ 2 ^ 20 * Θ * (R₀ + 1) * is
        + 2 ^ 36 * Θ ^ 2 * (13 + 2 * Real.logb 2 (R₀ + 1)) * D + 2 ^ 16 * (P₀ + 2) * ip * D := by
      positivity
    calc (Q + 3) ^ 2 * (2 ^ 20 * Θ * (R₀ + 1) * is
          + 2 ^ 36 * Θ ^ 2 * (13 + 2 * Real.logb 2 (R₀ + 1)) * D + 2 ^ 16 * (P₀ + 2) * ip * D)
        ≤ (16 * Q ^ 2) * (2 ^ 103 * (Q * B₁ * lX)) := mul_le_mul hq hsum hsum0 (by positivity)
      _ = 2 ^ 107 * (Q ^ 3 * B₁ * lX) := by ring
  have hE₁0 : 0 ≤ E₁ := by
    rw [hE₁e]; positivity
  -- `log Kc`
  have hKc1 : 1 ≤ Kc := by
    rw [hKc]
    exact one_le_mul_of_one_le_of_one_le (one_le_pow₀ (le_add_of_nonneg_right (by positivity)))
      (one_le_pow₀ (by linarith))
  have hlKc : log Kc ≤ 12 * (Q * A * lX) := by
    rw [hKc, Real.log_mul (by positivity) (by positivity), Real.log_pow, Real.log_pow]
    have h1 : log (1 + (Q + 1) ^ 3 * A) ≤ 6 * lX := by
      have hb : 1 + (Q + 1) ^ 3 * A ≤ X ^ 6 := by
        have hq3 : (Q + 1) ^ 3 ≤ 8 * Q ^ 3 := by
          have : (Q + 1) ^ 3 ≤ (2 * Q) ^ 3 := pow_le_pow_left₀ (by linarith) (by linarith) 3
          linarith [show (2 * Q) ^ 3 = 8 * Q ^ 3 by ring]
        have h9 : 1 + (Q + 1) ^ 3 * A ≤ 9 * (Q ^ 3 * A) := by
          have h1 : 1 ≤ Q ^ 3 * A := one_le_mul_of_one_le_of_one_le (one_le_pow₀ hQ) hA1
          have h2 : (Q + 1) ^ 3 * A ≤ 8 * Q ^ 3 * A := mul_le_mul_of_nonneg_right hq3 (by linarith)
          linarith
        have h4 : Q ^ 3 * A ≤ X ^ 4 := by
          calc Q ^ 3 * A ≤ X ^ 3 * X := by gcongr
            _ = X ^ 4 := by ring
        have hX9 : (9 : ℝ) ≤ X ^ 2 := by nlinarith only [hX]
        calc 1 + (Q + 1) ^ 3 * A ≤ 9 * (Q ^ 3 * A) := h9
          _ ≤ X ^ 2 * X ^ 4 := mul_le_mul hX9 h4 (by positivity) (hXp 2)
          _ = X ^ 6 := by ring
      have := hlXp 6 (by positivity) hb
      push_cast at this; linarith
    have h2 : log (Q + 3) ≤ 2 * lX := by
      have hb : Q + 3 ≤ X ^ 2 := by nlinarith only [hQX, hX]
      have := hlXp 2 (by positivity) hb
      push_cast at this; linarith
    have hl1 : 0 ≤ log (1 + (Q + 1) ^ 3 * A) :=
      Real.log_nonneg (le_add_of_nonneg_right (by positivity))
    have hl2' : 0 ≤ log (Q + 3) := Real.log_nonneg (by linarith)
    have h3 : (nQ : ℝ) * log (1 + (Q + 1) ^ 3 * A) ≤ Q * (6 * lX) := by gcongr
    have h4 : (((nQ + 2) * nA : ℕ) : ℝ) * log (Q + 3) ≤ (3 * Q) * A * (2 * lX) := by
      push_cast
      have : (nQ : ℝ) + 2 ≤ 3 * Q := by linarith
      gcongr
    have : Q * (6 * lX) ≤ 6 * (Q * A * lX) := by
      have : Q * lX ≤ Q * A * lX := by
        have h0 : 0 ≤ Q * lX := by positivity
        calc Q * lX = Q * lX * 1 := by ring
          _ ≤ Q * lX * A := mul_le_mul_of_nonneg_left hA1 h0
          _ = Q * A * lX := by ring
      linarith
    linarith
  have hlg : log (16 * (Q + 1) * Kc / δ') ≤ 18 * (Q * A * lX) := by
    rw [hδ'D, show 16 * (Q + 1) * Kc * D = (16 * (Q + 1) * D) * Kc by ring,
      Real.log_mul (by positivity) (by positivity)]
    have h1 : log (16 * (Q + 1) * D) ≤ 6 * lX := by
      have hb : 16 * (Q + 1) * D ≤ X ^ 6 := by
        have h1 : 16 * (Q + 1) * D ≤ 512 * (Q * Q * iδ) := by
          have : Q + 1 ≤ 2 * Q := by linarith
          have : 16 * (Q + 1) * D ≤ 16 * (2 * Q) * (16 * (Q * iδ)) := by gcongr
          linarith
        have h2 : Q * Q * iδ ≤ X ^ 3 := by
          calc Q * Q * iδ ≤ X * X * X := by gcongr
            _ = X ^ 3 := by ring
        have h3 : (512 : ℝ) ≤ X ^ 3 := by
          have : (2 : ℝ) ^ (3 * 3) ≤ X ^ 3 := h2pow 3
          norm_num at this; linarith
        calc 16 * (Q + 1) * D ≤ 512 * (Q * Q * iδ) := h1
          _ ≤ X ^ 3 * X ^ 3 := mul_le_mul h3 h2 (by positivity) (hXp 3)
          _ = X ^ 6 := by ring
      have := hlXp 6 (by positivity) hb
      push_cast at this; linarith
    have : 6 * lX ≤ 6 * (Q * A * lX) := by
      have : lX ≤ Q * A * lX :=
        le_mul_of_one_le_left (by linarith) (one_le_mul_of_one_le_of_one_le hQ hA1)
      linarith
    linarith
  have hlg0 : 0 ≤ log (16 * (Q + 1) * Kc / δ') := by
    rw [hδ'D]
    exact Real.log_nonneg
      (one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le (by linarith) hKc1) hD1)
  -- `E`
  have hEb : E ≤ 2 ^ 123 * (Ae * ic * Q ^ 3 * B₁ * lX) := by
    rw [hE]
    have h1 : 1536 * aε * E₁ ≤ 1536 * (6 * (Ae * ic)) * (2 ^ 107 * (Q ^ 3 * B₁ * lX)) := by
      gcongr
    have h2 : 1536 * log (16 * (Q + 1) * Kc / δ') * aε
        ≤ 1536 * (18 * (Q * A * lX)) * (6 * (Ae * ic)) := by gcongr
    have e2 : Q * A * lX * (Ae * ic) ≤ Ae * ic * Q ^ 3 * B₁ * lX := by
      have hQA : Q * A ≤ Q ^ 3 * B₁ := by
        have hA2 : A ≤ Ae ^ 2 := hAAe.trans (by nlinarith only [hAe1])
        have h1 : Q * A ≤ Q ^ 7 * Ae ^ 2 :=
          mul_le_mul (le_self_pow₀ hQ (by norm_num)) hA2 (by linarith) (by positivity)
        have h2 : Q ^ 4 * Ae ^ 2 ≤ B₁ := by
          rw [hB₁d]
          have : Q ^ 4 * Ae ^ 2 ≤ iδ * ic ^ 2 * (is ^ 4 + Q ^ 4 * Ae ^ 2) := by
            have h1 : 1 ≤ iδ * ic ^ 2 := one_le_mul_of_one_le_of_one_le hδ1 (one_le_pow₀ hic)
            have h2 : Q ^ 4 * Ae ^ 2 ≤ is ^ 4 + Q ^ 4 * Ae ^ 2 := by
              have : 0 ≤ is ^ 4 := by positivity
              linarith
            calc Q ^ 4 * Ae ^ 2 = 1 * (Q ^ 4 * Ae ^ 2) := by ring
              _ ≤ iδ * ic ^ 2 * (is ^ 4 + Q ^ 4 * Ae ^ 2) :=
                  mul_le_mul h1 h2 (by positivity) (by positivity)
          linarith
        calc Q * A ≤ Q ^ 7 * Ae ^ 2 := h1
          _ = Q ^ 3 * (Q ^ 4 * Ae ^ 2) := by ring
          _ ≤ Q ^ 3 * B₁ := mul_le_mul_of_nonneg_left h2 (by positivity)
      calc Q * A * lX * (Ae * ic) = Ae * ic * lX * (Q * A) := by ring
        _ ≤ Ae * ic * lX * (Q ^ 3 * B₁) := mul_le_mul_of_nonneg_left hQA (by positivity)
        _ = Ae * ic * Q ^ 3 * B₁ * lX := by ring
    have e1 : 1536 * (6 * (Ae * ic)) * (2 ^ 107 * (Q ^ 3 * B₁ * lX))
        = 1536 * 6 * 2 ^ 107 * (Ae * ic * Q ^ 3 * B₁ * lX) := by ring
    have hM0 : 0 ≤ Ae * ic * Q ^ 3 * B₁ * lX := by positivity
    nlinarith only [h1, h2, e1, e2, hM0]
  -- `E₂` and `E₃`
  have hn0 : (0 : ℝ) ≤ n := Nat.cast_nonneg _
  have hnb : (n : ℝ) ≤ 2 ^ 9 * (lX * iξ ^ 2) := by
    have h := claim_heldOut_le hQ hiδ hiξ
    rw [← hn] at h
    have hlQ0 : 0 ≤ log (Q * iδ) := Real.log_nonneg (one_le_mul_of_one_le_of_one_le hQ hδ1)
    have : log (Q * iδ) * iξ ^ 2 ≤ lX * iξ ^ 2 := mul_le_mul_of_nonneg_right hlQδ (by positivity)
    linarith
  have hlS : log (16 * S * (S + 1) / δ') ≤ 13 * lX := by
    rw [hδ'D]
    have hb : 16 * S * (S + 1) * D ≤ X ^ 13 := by
      have h1 : 16 * S * (S + 1) * D
          ≤ 16 * (7 * (Q ^ 2 * A)) * (8 * (Q ^ 2 * A)) * (16 * (Q * iδ)) := by gcongr
      have h2 : Q ^ 2 * A ≤ X ^ 3 := by
        calc Q ^ 2 * A ≤ X ^ 2 * X := by gcongr
          _ = X ^ 3 := by ring
      have h3 : Q * iδ ≤ X ^ 2 := by
        calc Q * iδ ≤ X * X := by gcongr
          _ = X ^ 2 := by ring
      have h4 : (16 * 7 * 8 * 16 : ℝ) ≤ X ^ 5 := by
        have : (2 : ℝ) ^ (3 * 5) ≤ X ^ 5 := h2pow 5
        norm_num at this; linarith
      calc 16 * S * (S + 1) * D ≤ 16 * (7 * (Q ^ 2 * A)) * (8 * (Q ^ 2 * A)) * (16 * (Q * iδ)) := h1
        _ = (16 * 7 * 8 * 16) * (Q ^ 2 * A) * (Q ^ 2 * A) * (Q * iδ) := by ring
        _ ≤ X ^ 5 * X ^ 3 * X ^ 3 * X ^ 2 := by gcongr
        _ = X ^ 13 := by ring
    have := hlXp 13 (by positivity) hb
    push_cast at this; linarith
  have hlS0 : 0 ≤ log (16 * S * (S + 1) / δ') := by
    rw [hδ'D]
    exact Real.log_nonneg (one_le_mul_of_one_le_of_one_le (by nlinarith only [hS1]) hD1)
  have hK₁0 : 0 ≤ K₁ := by rw [hK₁]; positivity
  have hK₁b : K₁ ≤ 2 ^ 16 * (Q ^ 2 * A * Ae * ic * lX) := by
    rw [hK₁]
    have h1 : 36 * S * aε * log (16 * S * (S + 1) / δ')
        ≤ 36 * (7 * (Q ^ 2 * A)) * (6 * (Ae * ic)) * (13 * lX) := by gcongr
    have h2 : S + 1 ≤ 8 * (Q ^ 2 * A) := by linarith
    have h3 : Q ^ 2 * A ≤ Q ^ 2 * A * Ae * ic * lX := by
      have : 1 ≤ Ae * ic * lX :=
        one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le hAe1 hic) (by linarith)
      have h0 : 0 ≤ Q ^ 2 * A := by positivity
      calc Q ^ 2 * A = Q ^ 2 * A * 1 := by ring
        _ ≤ Q ^ 2 * A * (Ae * ic * lX) := mul_le_mul_of_nonneg_left this h0
        _ = _ := by ring
    linarith
  have hE₂b : E₂ ≤ 2 ^ 80 * (Q ^ 3 * iξ ^ 2 * ic ^ 2 * iδ * (is ^ 8 + Q ^ 4 * Ae ^ 2 * is ^ 4)
      * lX ^ 2) := by
    rw [hE₂, show ∀ x : ℝ, x / (1 / iδ) = x * iδ from fun x => by field_simp]
    set U := Q ^ 2 * ic ^ 2 * (is ^ 8 + Q ^ 4 * Ae ^ 2 * is ^ 4) * lX with hUd
    have hU0 : 0 ≤ U := by rw [hUd]; positivity
    have hs8 : 1 ≤ is ^ 8 := one_le_pow₀ hs1
    have hQ2 : 1 ≤ Q ^ 2 := one_le_pow₀ hQ
    have hin : 6 * (Q + 2) * R₀ + K₁ + n ≤ 2 ^ 67 * U := by
      have h1 : 6 * (Q + 2) * R₀ ≤ 18 * (2 ^ 62 : ℝ) * U := by
        have hq : 6 * (Q + 2) ≤ 18 * Q := by linarith
        have hR : R₀ ≤ 2 ^ 62 * (Q * ic ^ 2 * (is ^ 8 + Q ^ 4 * Ae ^ 2 * is ^ 4) * lX) := by
          have hΘs : Θ ^ 2 * is ^ 4 ≤ 2 ^ 37 * (ic ^ 2 * (is ^ 8 + Q ^ 4 * Ae ^ 2 * is ^ 4)) := by
            have := mul_le_mul_of_nonneg_right hΘ2 (by positivity : (0 : ℝ) ≤ is ^ 4)
            linarith [show 2 ^ 37 * (ic ^ 2 * (is ^ 4 + Q ^ 4 * Ae ^ 2)) * is ^ 4
              = 2 ^ 37 * (ic ^ 2 * (is ^ 8 + Q ^ 4 * Ae ^ 2 * is ^ 4)) by ring]
          have e : 2 ^ 25 * Rs = 2 ^ 25 * (Q * lX) * (Θ ^ 2 * is ^ 4) := by rw [hRsd]; ring
          have := mul_le_mul_of_nonneg_left hΘs (by positivity : (0 : ℝ) ≤ 2 ^ 25 * (Q * lX))
          calc R₀ ≤ 2 ^ 25 * Rs := hR₀
            _ = 2 ^ 25 * (Q * lX) * (Θ ^ 2 * is ^ 4) := e
            _ ≤ 2 ^ 25 * (Q * lX) * (2 ^ 37 * (ic ^ 2 * (is ^ 8 + Q ^ 4 * Ae ^ 2 * is ^ 4))) := this
            _ = _ := by ring
        calc 6 * (Q + 2) * R₀
            ≤ (18 * Q) * (2 ^ 62 * (Q * ic ^ 2 * (is ^ 8 + Q ^ 4 * Ae ^ 2 * is ^ 4) * lX)) :=
              mul_le_mul hq hR hR₀0 (by positivity)
          _ = 18 * (2 ^ 62 : ℝ) * U := by rw [hUd]; ring
      have h2 : Q ^ 2 * A * Ae * ic * lX ≤ U := by
        rw [hUd]
        have hb : A * Ae * ic ≤ ic ^ 2 * (Q ^ 4 * Ae ^ 2 * is ^ 4) := by
          have a1 : A * Ae ≤ Ae ^ 2 := by nlinarith only [hAAe, hA1, hAe1]
          have a2 : ic ≤ ic ^ 2 := by nlinarith only [hic]
          have a3 : 1 ≤ Q ^ 4 * is ^ 4 :=
            one_le_mul_of_one_le_of_one_le (one_le_pow₀ hQ) (one_le_pow₀ hs1)
          calc A * Ae * ic ≤ Ae ^ 2 * ic ^ 2 := mul_le_mul a1 a2 (by linarith) (by positivity)
            _ = Ae ^ 2 * ic ^ 2 * 1 := by ring
            _ ≤ Ae ^ 2 * ic ^ 2 * (Q ^ 4 * is ^ 4) := mul_le_mul_of_nonneg_left a3 (by positivity)
            _ = _ := by ring
        have hb2 : ic ^ 2 * (Q ^ 4 * Ae ^ 2 * is ^ 4)
            ≤ ic ^ 2 * (is ^ 8 + Q ^ 4 * Ae ^ 2 * is ^ 4) := by
          have : 0 ≤ is ^ 8 := by positivity
          gcongr; linarith
        calc Q ^ 2 * A * Ae * ic * lX = Q ^ 2 * lX * (A * Ae * ic) := by ring
          _ ≤ Q ^ 2 * lX * (ic ^ 2 * (is ^ 8 + Q ^ 4 * Ae ^ 2 * is ^ 4)) :=
              mul_le_mul_of_nonneg_left (hb.trans hb2) (by positivity)
          _ = _ := by ring
      have h3 : lX * iξ ^ 2 ≤ U := by
        rw [hUd]
        have hb : iξ ^ 2 ≤ Q ^ 2 * ic ^ 2 * (is ^ 8 + Q ^ 4 * Ae ^ 2 * is ^ 4) := by
          have a1 : iξ ^ 2 ≤ 4 * ic ^ 2 :=
            (pow_le_pow_left₀ (by linarith) hξc 2).trans_eq (by ring)
          have a2 : 4 ≤ Q ^ 2 * (is ^ 8 + Q ^ 4 * Ae ^ 2 * is ^ 4) := by
            have : 0 ≤ Q ^ 4 * Ae ^ 2 * is ^ 4 := by positivity
            have : 4 ≤ is ^ 8 := by
              have := pow_le_pow_left₀ (by norm_num) his 8
              norm_num at this; linarith
            nlinarith only [hQ2, this, ‹0 ≤ Q ^ 4 * Ae ^ 2 * is ^ 4›]
          calc iξ ^ 2 ≤ 4 * ic ^ 2 := a1
            _ = ic ^ 2 * 4 := by ring
            _ ≤ ic ^ 2 * (Q ^ 2 * (is ^ 8 + Q ^ 4 * Ae ^ 2 * is ^ 4)) :=
                mul_le_mul_of_nonneg_left a2 (by positivity)
            _ = _ := by ring
        calc lX * iξ ^ 2 ≤ lX * (Q ^ 2 * ic ^ 2 * (is ^ 8 + Q ^ 4 * Ae ^ 2 * is ^ 4)) :=
              mul_le_mul_of_nonneg_left hb hlX0
          _ = _ := by ring
      have hK : 18 * (2 ^ 62 : ℝ) + 2 ^ 16 + 2 ^ 9 ≤ (2 : ℝ) ^ 67 := by norm_num
      have := mul_le_mul_of_nonneg_right hK hU0
      linarith
    have hin0 : 0 ≤ 6 * (Q + 2) * R₀ + K₁ + n := by positivity
    calc 4 * (Q + 1) * n * (6 * (Q + 2) * R₀ + K₁ + n) * iδ
        ≤ 4 * (2 * Q) * (2 ^ 9 * (lX * iξ ^ 2)) * (2 ^ 67 * U) * iδ := by
          gcongr; linarith
      _ = 2 ^ 79 * (Q ^ 3 * iξ ^ 2 * ic ^ 2 * iδ * (is ^ 8 + Q ^ 4 * Ae ^ 2 * is ^ 4)
          * lX ^ 2) := by
          rw [hUd]; ring
      _ ≤ _ := by
          have : 0 ≤ Q ^ 3 * iξ ^ 2 * ic ^ 2 * iδ * (is ^ 8 + Q ^ 4 * Ae ^ 2 * is ^ 4)
              * lX ^ 2 := by
            positivity
          linarith
  have hE₃b : E₃ ≤ 2 ^ 40 * (Q ^ 5 * A ^ 2 * Ae ^ 2 * ic ^ 2 * iδ * lX ^ 2) := by
    rw [hE₃, hδ'D]
    calc 16 * K₁ ^ 2 * D ≤ 16 * (2 ^ 16 * (Q ^ 2 * A * Ae * ic * lX)) ^ 2 * (16 * (Q * iδ)) := by
          gcongr
      _ = _ := by ring
  exact ⟨hEb, hE₂b, hE₃b⟩

lemma summed_inv (R ξ : ℝ) : 1 / (2 * R * (1 / ξ)) = ξ / (2 * R) := by
  simp only [one_div, mul_inv, inv_inv]; ring

lemma quarter_inv (ξ : ℝ) : 1 / (4 * (1 / ξ)) = ξ / 4 := by
  simp only [one_div, mul_inv, inv_inv]; ring

/-- `claim_terms_le` at the summed rates. -/
lemma rel_terms_le {Q A Ae iδ R iξ ip is : ℝ} {nQ nA : ℕ} (hQ : 1 ≤ Q) (hA : 2 ≤ A)
    (hAAe : A ≤ Ae) (hiδ : 2 ≤ iδ) (hR : 1 ≤ R) (hiξ : 1 ≤ iξ) (hip : 1 ≤ ip) (his : 2 ≤ is)
    (hnQ : (nQ : ℝ) ≤ Q) (hnA : (nA : ℝ) ≤ A) :
    (loopMassTerms Q A (Ae + Q) nQ nA (1 / iδ) (1 / (2 * R * iξ)) (1 / (4 * iξ)) (1 / ip)
        (1 / 2 - 1 / is)).1
        ≤ 2 ^ 129 * ((Ae + Q) * R * Q ^ 3 * iξ
          * (R ^ 3 * iξ ^ 3 * (is ^ 11 + Q ^ 6 * (Ae + Q) ^ 3 * is ^ 5)
            + R ^ 2 * iξ ^ 2 * iδ * (is ^ 4 + Q ^ 4 * (Ae + Q) ^ 2) + iδ * ip ^ 2 * is ^ 2)
          * log (Q * R * Ae * iδ * iξ * ip * is))
      ∧ (loopMassTerms Q A (Ae + Q) nQ nA (1 / iδ) (1 / (2 * R * iξ)) (1 / (4 * iξ)) (1 / ip)
          (1 / 2 - 1 / is)).2.1
        ≤ 2 ^ 90 * (Q ^ 3 * R ^ 2 * iξ ^ 4 * iδ * (is ^ 8 + Q ^ 4 * (Ae + Q) ^ 2 * is ^ 4)
          * log (Q * R * Ae * iδ * iξ * ip * is) ^ 2)
      ∧ (loopMassTerms Q A (Ae + Q) nQ nA (1 / iδ) (1 / (2 * R * iξ)) (1 / (4 * iξ)) (1 / ip)
          (1 / 2 - 1 / is)).2.2
        ≤ 2 ^ 46 * (Q ^ 5 * A ^ 2 * (Ae + Q) ^ 2 * R ^ 2 * iξ ^ 2 * iδ
          * log (Q * R * Ae * iδ * iξ * ip * is) ^ 2) := by
  obtain ⟨h1, h2, h3⟩ := claim_terms_le (Ae := Ae + Q) (ic := 2 * R * iξ) (iξ := 4 * iξ)
    (nQ := nQ) (nA := nA) hQ hA (by linarith) hiδ (by nlinarith) (by linarith) (by nlinarith) hip
    his hnQ hnA
  obtain ⟨hl0, hl⟩ := summed_log_mass_le (is := is) hQ (hA.trans hAAe) hiδ hR hiξ hip (by linarith)
  set lX := log (Q * (Ae + Q) * iδ * (2 * R * iξ) * ip * is) with hlXd
  set lY := log (Q * R * Ae * iδ * iξ * ip * is) with hlYd
  have hlY0 : 0 ≤ lY := by linarith
  have hl2 : lX ^ 2 ≤ 9 * lY ^ 2 := by
    have := pow_le_pow_left₀ hl0 hl 2
    linarith [show (3 * lY) ^ 2 = 9 * lY ^ 2 by ring]
  have hQ0 : 0 ≤ Q := by linarith
  have hAQ : 0 ≤ Ae + Q := by linarith
  have hR0 : 0 ≤ R := by linarith
  have hξ0 : 0 ≤ iξ := by linarith
  have hδ0 : 0 ≤ iδ := by linarith
  have hp0 : 0 ≤ ip := by linarith
  have hs0 : 0 ≤ is := by linarith
  refine ⟨h1.trans ?_, h2.trans ?_, h3.trans ?_⟩
  · set a := is ^ 11 + Q ^ 6 * (Ae + Q) ^ 3 * is ^ 5 with had
    set b := is ^ 4 + Q ^ 4 * (Ae + Q) ^ 2 with hbd
    have ha : 0 ≤ a := by positivity
    have hb0 : 0 ≤ b := by positivity
    set T := (Ae + Q) * R * Q ^ 3 * iξ
      * (R ^ 3 * iξ ^ 3 * a + R ^ 2 * iξ ^ 2 * iδ * b + iδ * ip ^ 2 * is ^ 2) with hTd
    have hT0 : 0 ≤ T := by positivity
    have hb : (Ae + Q) * (2 * R * iξ) * Q ^ 3
        * ((2 * R * iξ) ^ 3 * a + iδ * (2 * R * iξ) ^ 2 * b + iδ * ip ^ 2 * is ^ 2) ≤ 16 * T := by
      have : 0 ≤ (Ae + Q) * Q ^ 3
          * (8 * R ^ 3 * iξ ^ 3 * iδ * b + 14 * R * iξ * iδ * ip ^ 2 * is ^ 2) := by
        positivity
      rw [hTd]; nlinarith only [this]
    calc _ ≤ 2 ^ 123 * ((16 * T) * (3 * lY)) :=
          mul_le_mul_of_nonneg_left (mul_le_mul hb hl hl0 (by positivity)) (by positivity)
      _ = 2 ^ 123 * 48 * (T * lY) := by ring
      _ ≤ 2 ^ 129 * (T * lY) := mul_le_mul_of_nonneg_right (by norm_num) (mul_nonneg hT0 hlY0)
  · have hP : 0 ≤ Q ^ 3 * (4 * iξ) ^ 2 * (2 * R * iξ) ^ 2 * iδ
        * (is ^ 8 + Q ^ 4 * (Ae + Q) ^ 2 * is ^ 4) := by positivity
    calc _ ≤ 2 ^ 80 * (Q ^ 3 * (4 * iξ) ^ 2 * (2 * R * iξ) ^ 2 * iδ
          * (is ^ 8 + Q ^ 4 * (Ae + Q) ^ 2 * is ^ 4) * (9 * lY ^ 2)) :=
          mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left hl2 hP) (by positivity)
      _ = 2 ^ 80 * 576 * (Q ^ 3 * R ^ 2 * iξ ^ 4 * iδ * (is ^ 8 + Q ^ 4 * (Ae + Q) ^ 2 * is ^ 4)
          * lY ^ 2) := by ring
      _ ≤ _ := mul_le_mul_of_nonneg_right (by norm_num) (by positivity)
  · have hP : 0 ≤ Q ^ 5 * A ^ 2 * (Ae + Q) ^ 2 * (2 * R * iξ) ^ 2 * iδ := by positivity
    calc _ ≤ 2 ^ 40 * (Q ^ 5 * A ^ 2 * (Ae + Q) ^ 2 * (2 * R * iξ) ^ 2 * iδ * (9 * lY ^ 2)) :=
          mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left hl2 hP) (by positivity)
      _ = 2 ^ 40 * 36 * (Q ^ 5 * A ^ 2 * (Ae + Q) ^ 2 * R ^ 2 * iξ ^ 2 * iδ * lY ^ 2) := by ring
      _ ≤ _ := mul_le_mul_of_nonneg_right (by norm_num) (by positivity)

/-- `(L + Q)⁶ E ≤ A^⌈L/2⌉` once `L` passes the tangent bound on `6 log (L + 1)` at `24/log A`. -/
lemma pow_of_tangent {A Q E : ℝ} {L : ℕ} (hA : 2 ≤ A) (hQ : 1 ≤ Q) (hE : 1 ≤ E)
    (hL : 1 + 4 * (log E + 6 * log (Q + 4) + 24) / log A ≤ L) :
    ((L : ℝ) + Q) ^ 6 * E ≤ A ^ ((L + 1) / 2) := by
  have hl2 := Real.log_two_gt_d9
  have ha : log 2 ≤ log A := log_two_le_log hA
  have ha0 : 0 < log A := by linarith
  have hL0 : (0 : ℝ) ≤ L := Nat.cast_nonneg L
  have hk : (L : ℝ) ≤ 2 * (((L + 1) / 2 : ℕ) : ℝ) := by
    exact_mod_cast (show L ≤ 2 * ((L + 1) / 2) by omega)
  have hlE : 0 ≤ log E := Real.log_nonneg hE
  have hlQ : 0 ≤ log (Q + 4) := Real.log_nonneg (by linarith)
  have hL' : 4 * (log E + 6 * log (Q + 4) + 24) ≤ ((L : ℝ) - 1) * log A := by
    have := (div_le_iff₀ ha0).1 (show 4 * (log E + 6 * log (Q + 4) + 24) / log A ≤ (L : ℝ) - 1 by
      linarith)
    linarith
  have hT0 : 0 < 24 / log A := by positivity
  have hL1 : (0 : ℝ) < L + 1 := by linarith
  have htan : log ((L : ℝ) + 1) ≤ log (24 / log A) + ((L : ℝ) + 1) * log A / 24 - 1 := by
    have := Real.log_le_sub_one_of_pos (div_pos hL1 hT0)
    rw [Real.log_div hL1.ne' hT0.ne'] at this
    have e : ((L : ℝ) + 1) / (24 / log A) = ((L : ℝ) + 1) * log A / 24 := by field_simp
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
    calc log (24 / log A) ≤ log 35 := Real.log_le_log hT0 hT35
      _ ≤ 4 := by rw [Real.log_le_iff_le_exp (by norm_num)]; exact h35
  have hLQ : log ((L : ℝ) + Q) ≤ log ((L : ℝ) + 1) + log (Q + 4) := by
    rw [← Real.log_mul hL1.ne' (by linarith)]
    exact Real.log_le_log (by linarith) (by nlinarith)
  have hmain : 6 * log ((L : ℝ) + Q) + log E ≤ (L : ℝ) * log A / 2 := by
    linarith
  have hk' : (L : ℝ) * log A / 2 ≤ (((L + 1) / 2 : ℕ) : ℝ) * log A := by
    have := mul_le_mul_of_nonneg_right hk ha0.le
    linarith
  have hpos : (0 : ℝ) < ((L : ℝ) + Q) ^ 6 * E := by positivity
  have hA0 : (0 : ℝ) < A := by linarith
  rw [← Real.log_le_log_iff hpos (pow_pos hA0 _), Real.log_mul (by positivity) (by linarith),
    Real.log_pow, Real.log_pow]
  have e6 : ((6 : ℕ) : ℝ) = 6 := by norm_num
  rw [e6]
  linarith

set_option maxHeartbeats 1000000 in -- `G`'s bound chains through many `gcongr` steps
/-- `L₀` at a constant `C₀ ≥ 1` against `1 + log (|Q| R/(δ ξ pAP s₀))/log |Σ|`, in the inverted
coordinates. -/
lemma loopL0_le {Q A R iδ iξ ip is C₀ : ℝ} (hQ : 1 ≤ Q) (hA : 2 ≤ A) (hR : 1 ≤ R) (hiδ : 2 ≤ iδ)
    (hiξ : 1 ≤ iξ) (hip : 1 ≤ ip) (his : 2 ≤ is) (hC₀ : 1 ≤ C₀) :
    1 ≤ Q ^ 9 * (A + Q) ^ 4 * R ^ 4 * log (Q * R * A * iδ * iξ * ip * is) ^ 3 * iδ * iξ ^ 4
        * ip ^ 2 * is ^ 11
      ∧ 1 + 4 * (log (C₀ * (Q ^ 9 * (A + Q) ^ 4 * R ^ 4 * log (Q * R * A * iδ * iξ * ip * is) ^ 3
          * iδ * iξ ^ 4 * ip ^ 2 * is ^ 11)) + 6 * log (Q + 4) + 24) / log A
        ≤ (263 + 6 * log C₀) * (1 + log (Q * R * iδ * iξ * ip * is) / log A) := by
  have hl2 := Real.log_two_gt_d9
  have hδ1 : 1 ≤ iδ := by linarith
  have hs1 : 1 ≤ is := by linarith
  have hA1 : 1 ≤ A := by linarith
  set Z := Q * R * iδ * iξ * ip * is with hZd
  have hZ1 : 1 ≤ Z := one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le
    (one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le
      (one_le_mul_of_one_le_of_one_le hQ hR) hδ1) hiξ) hip) hs1
  have hZ4 : 4 ≤ Z := by
    have h1 : 1 ≤ Q * R := one_le_mul_of_one_le_of_one_le hQ hR
    have h2 : 2 ≤ Q * R * iδ := by nlinarith only [h1, hiδ]
    have h3 : 2 ≤ Q * R * iδ * iξ := by nlinarith only [h2, hiξ]
    have h4 : 2 ≤ Q * R * iδ * iξ * ip := by nlinarith only [h3, hip]
    rw [hZd]; nlinarith only [h4, his]
  set Y := Q * R * A * iδ * iξ * ip * is with hYd
  have hYZ : Y = A * Z := by rw [hYd, hZd]; ring
  have hY8 : 8 ≤ Y := by rw [hYZ]; nlinarith
  have hlY1 : 1 ≤ log Y := one_le_log_of_three_le (by linarith)
  have hlYY : log Y ≤ Y := Real.log_le_self (by linarith)
  set G := Q ^ 9 * (A + Q) ^ 4 * R ^ 4 * log Y ^ 3 * iδ * iξ ^ 4 * ip ^ 2 * is ^ 11 with hGd
  have hG1 : 1 ≤ G := by
    rw [hGd]
    have : 1 ≤ log Y ^ 3 := one_le_pow₀ hlY1
    exact one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le
      (one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le
        (one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le
          (one_le_mul_of_one_le_of_one_le (one_le_pow₀ hQ) (one_le_pow₀ (by linarith)))
          (one_le_pow₀ hR)) this) hδ1) (one_le_pow₀ hiξ)) (one_le_pow₀ hip)) (one_le_pow₀ hs1)
  refine ⟨hG1, ?_⟩
  -- `G ≤ 16 A⁷ Z³⁸`
  have hQZ : Q ≤ Z := by
    have : 1 ≤ R * iδ * iξ * ip * is :=
      one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le
        (one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le hR hδ1) hiξ) hip) hs1
    calc Q = Q * 1 := by ring
      _ ≤ Q * (R * iδ * iξ * ip * is) := mul_le_mul_of_nonneg_left this (by linarith)
      _ = Z := by rw [hZd]; ring
  have hRZ : R ≤ Z := by
    have : 1 ≤ Q * iδ * iξ * ip * is :=
      one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le
        (one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le hQ hδ1) hiξ) hip) hs1
    calc R = R * 1 := by ring
      _ ≤ R * (Q * iδ * iξ * ip * is) := mul_le_mul_of_nonneg_left this (by linarith)
      _ = Z := by rw [hZd]; ring
  have hδZ : iδ ≤ Z := by
    have : 1 ≤ Q * R * iξ * ip * is :=
      one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le
        (one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le hQ hR) hiξ) hip) hs1
    calc iδ = iδ * 1 := by ring
      _ ≤ iδ * (Q * R * iξ * ip * is) := mul_le_mul_of_nonneg_left this (by linarith)
      _ = Z := by rw [hZd]; ring
  have hξZ : iξ ≤ Z := by
    have : 1 ≤ Q * R * iδ * ip * is :=
      one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le
        (one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le hQ hR) hδ1) hip) hs1
    calc iξ = iξ * 1 := by ring
      _ ≤ iξ * (Q * R * iδ * ip * is) := mul_le_mul_of_nonneg_left this (by linarith)
      _ = Z := by rw [hZd]; ring
  have hpZ : ip ≤ Z := by
    have : 1 ≤ Q * R * iδ * iξ * is :=
      one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le
        (one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le hQ hR) hδ1) hiξ) hs1
    calc ip = ip * 1 := by ring
      _ ≤ ip * (Q * R * iδ * iξ * is) := mul_le_mul_of_nonneg_left this (by linarith)
      _ = Z := by rw [hZd]; ring
  have hsZ : is ≤ Z := by
    have : 1 ≤ Q * R * iδ * iξ * ip :=
      one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le
        (one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le hQ hR) hδ1) hiξ) hip
    calc is = is * 1 := by ring
      _ ≤ is * (Q * R * iδ * iξ * ip) := mul_le_mul_of_nonneg_left this (by linarith)
      _ = Z := by rw [hZd]; ring
  have hlY2 : log Y ^ 3 ≤ Y ^ 3 := pow_le_pow_left₀ (by linarith) hlYY 3
  have hAQ : A + Q ≤ 2 * A * Z := by
    have h1 : A + Q ≤ 2 * A * Q := by
      nlinarith only [mul_nonneg (sub_nonneg.2 hA1) (sub_nonneg.2 hQ), hA1, hQ]
    have h2 : 2 * A * Q ≤ 2 * A * Z := mul_le_mul_of_nonneg_left hQZ (by linarith)
    linarith
  have hAQ0 : 0 ≤ A + Q := by linarith
  have hGZ : G ≤ 16 * A ^ 7 * Z ^ 38 := by
    rw [hGd]
    calc Q ^ 9 * (A + Q) ^ 4 * R ^ 4 * log Y ^ 3 * iδ * iξ ^ 4 * ip ^ 2 * is ^ 11
        ≤ Z ^ 9 * (2 * A * Z) ^ 4 * Z ^ 4 * Y ^ 3 * Z * Z ^ 4 * Z ^ 2 * Z ^ 11 := by
          gcongr
      _ = 16 * A ^ 7 * Z ^ 38 := by rw [hYZ]; ring
  have hlA : log 2 ≤ log A := log_two_le_log hA
  have hlA0 : 0 < log A := by linarith
  have hlZ0 : 0 ≤ log Z := Real.log_nonneg hZ1
  have hlC : 0 ≤ log C₀ := Real.log_nonneg hC₀
  have hl16 : log 16 ≤ 3 := by
    rw [Real.log_le_iff_le_exp (by norm_num)]
    have e : Real.exp 3 = Real.exp 1 ^ 3 := by rw [← Real.exp_nat_mul]; norm_num
    have h1 : (2.7 : ℝ) ≤ Real.exp 1 := by have := Real.exp_one_gt_d9; linarith
    have := pow_le_pow_left₀ (by norm_num) h1 3
    rw [e]; norm_num at this ⊢; linarith
  have hlG : log G ≤ 3 + 7 * log A + 38 * log Z := by
    have := Real.log_le_log (by linarith) hGZ
    have e : log (16 * A ^ 7 * Z ^ 38) = log 16 + 7 * log A + 38 * log Z := by
      rw [Real.log_mul (by positivity) (by positivity), Real.log_mul (by norm_num) (by positivity),
        Real.log_pow, Real.log_pow]
      push_cast; ring
    linarith
  have hlCG : log (C₀ * G) ≤ log C₀ + 3 + 7 * log A + 38 * log Z := by
    rw [Real.log_mul (by linarith) (by linarith)]; linarith
  have hlQ4 : log (Q + 4) ≤ 2 + log Z := by
    have h5 : Q + 4 ≤ 5 * Z := by linarith
    have := Real.log_le_log (by linarith) h5
    rw [Real.log_mul (by norm_num) (by linarith)] at this
    have : log 5 ≤ 2 := by
      rw [Real.log_le_iff_le_exp (by norm_num)]
      have := Real.add_one_le_exp (2 : ℝ)
      have e : Real.exp 2 = Real.exp 1 * Real.exp 1 := by rw [← Real.exp_add]; norm_num
      have := Real.exp_one_gt_d9
      nlinarith
    linarith
  -- assemble
  have hnum : 4 * (log (C₀ * G) + 6 * log (Q + 4) + 24)
      ≤ 28 * log A + (4 * log C₀ + 156) + 176 * log Z := by linarith
  have hdiv : 4 * (log (C₀ * G) + 6 * log (Q + 4) + 24) / log A
      ≤ 28 + (4 * log C₀ + 156) / log A + 176 * (log Z / log A) := by
    rw [div_le_iff₀ hlA0]
    have e : (28 + (4 * log C₀ + 156) / log A + 176 * (log Z / log A)) * log A
        = 28 * log A + (4 * log C₀ + 156) + 176 * log Z := by field_simp
    rw [e]; exact hnum
  have hc : (4 * log C₀ + 156) / log A ≤ 6 * log C₀ + 234 := by
    rw [div_le_iff₀ hlA0]
    nlinarith [mul_le_mul_of_nonneg_left (show (2 : ℝ) / 3 ≤ log A by linarith)
      (show 0 ≤ 6 * log C₀ + 234 by linarith)]
  have hq : 0 ≤ log Z / log A := div_nonneg hlZ0 hlA0.le
  nlinarith [mul_nonneg hlC hq]

open Asymptotics Filter

/-- `LoopClaimUniform`'s prefix shape against `LoopClaimSimple`'s. -/
lemma loop_pref_dom {r : LoopRun} (hr : r ∈ loopRunRegion) :
    0 ≤ r.Q ^ 3 * (r.L + r.Q) ^ 2 * r.R ^ 2
        * log ((r.L + r.Q) * r.A * r.R / (r.δ * r.ξ * r.pAP * r.s₀)) / r.ξ ^ 2
        * (1 / r.s₀ ^ 8 + (r.A + r.Q) ^ 2 * r.Q ^ 4 / r.s₀ ^ 4)
      ∧ 0 ≤ r.Q ^ 7 * (r.A + r.Q) ^ 2 * (r.L + r.Q) ^ 2 * r.R ^ 2
        * log ((r.L + r.Q) * r.A * r.R / (r.δ * r.ξ * r.pAP * r.s₀)) / (r.ξ ^ 2 * r.s₀ ^ 8)
      ∧ r.Q ^ 3 * (r.L + r.Q) ^ 2 * r.R ^ 2
        * log ((r.L + r.Q) * r.A * r.R / (r.δ * r.ξ * r.pAP * r.s₀)) / r.ξ ^ 2
        * (1 / r.s₀ ^ 8 + (r.A + r.Q) ^ 2 * r.Q ^ 4 / r.s₀ ^ 4)
        ≤ 2 * (r.Q ^ 7 * (r.A + r.Q) ^ 2 * (r.L + r.Q) ^ 2 * r.R ^ 2
          * log ((r.L + r.Q) * r.A * r.R / (r.δ * r.ξ * r.pAP * r.s₀)) / (r.ξ ^ 2 * r.s₀ ^ 8)) := by
  obtain ⟨hQ, hA, hδ, hδ2, hR, hξ, hξ1, hp, hp1, hs, hs1⟩ := hr
  have hQ' : (1 : ℝ) ≤ r.Q := by exact_mod_cast hQ
  have hA' : (2 : ℝ) ≤ r.A := by exact_mod_cast hA
  have hL : (0 : ℝ) ≤ r.L := Nat.cast_nonneg _
  have hx : 1 ≤ (r.L + r.Q) * r.A * r.R / (r.δ * r.ξ * r.pAP * r.s₀) := by
    rw [le_div_iff₀ (by positivity), one_mul]
    have h1 : r.δ * r.ξ * r.pAP * r.s₀ ≤ 1 := by
      have : r.δ * r.ξ ≤ 1 := by nlinarith
      have : r.δ * r.ξ * r.pAP ≤ 1 := by nlinarith
      nlinarith
    have h2 : 1 ≤ (r.L + r.Q) * r.A * r.R := by
      have : 1 ≤ ((r.L : ℝ) + r.Q) * r.A := by nlinarith
      nlinarith
    linarith
  have hlg := Real.log_nonneg hx
  set lg := log ((r.L + r.Q) * r.A * r.R / (r.δ * r.ξ * r.pAP * r.s₀))
  have hK : 0 ≤ r.Q ^ 3 * (r.L + r.Q) ^ 2 * r.R ^ 2 * lg / r.ξ ^ 2 := by positivity
  have hs4 : r.s₀ ^ 4 ≤ 1 := pow_le_one₀ hs.le (by linarith)
  have hQ4 : (1 : ℝ) ≤ (r.A + r.Q) ^ 2 * r.Q ^ 4 :=
    one_le_mul_of_one_le_of_one_le (one_le_pow₀ (by linarith)) (one_le_pow₀ hQ')
  have hb : 1 / r.s₀ ^ 8 + (r.A + r.Q) ^ 2 * r.Q ^ 4 / r.s₀ ^ 4
      ≤ 2 * ((r.A + r.Q) ^ 2 * r.Q ^ 4 / r.s₀ ^ 8) := by
    have h8 : 0 < r.s₀ ^ 8 := by positivity
    have h48 : r.s₀ ^ 8 ≤ r.s₀ ^ 4 := by
      have : r.s₀ ^ 8 = r.s₀ ^ 4 * r.s₀ ^ 4 := by ring
      rw [this]; exact mul_le_of_le_one_right (by positivity) hs4
    have e1 : 1 / r.s₀ ^ 8 ≤ (r.A + r.Q) ^ 2 * r.Q ^ 4 / r.s₀ ^ 8 :=
      div_le_div_of_nonneg_right hQ4 h8.le
    have e2 : (r.A + r.Q) ^ 2 * r.Q ^ 4 / r.s₀ ^ 4 ≤ (r.A + r.Q) ^ 2 * r.Q ^ 4 / r.s₀ ^ 8 :=
      div_le_div_of_nonneg_left (by positivity) h8 h48
    linarith
  have e : r.Q ^ 7 * (r.A + r.Q) ^ 2 * (r.L + r.Q) ^ 2 * r.R ^ 2 * lg / (r.ξ ^ 2 * r.s₀ ^ 8)
      = r.Q ^ 3 * (r.L + r.Q) ^ 2 * r.R ^ 2 * lg / r.ξ ^ 2
        * ((r.A + r.Q) ^ 2 * r.Q ^ 4 / r.s₀ ^ 8) := by
    field_simp
  refine ⟨mul_nonneg hK (by positivity), by rw [e]; positivity, ?_⟩
  rw [e]
  calc r.Q ^ 3 * (r.L + r.Q) ^ 2 * r.R ^ 2 * lg / r.ξ ^ 2
        * (1 / r.s₀ ^ 8 + (r.A + r.Q) ^ 2 * r.Q ^ 4 / r.s₀ ^ 4)
      ≤ r.Q ^ 3 * (r.L + r.Q) ^ 2 * r.R ^ 2 * lg / r.ξ ^ 2
        * (2 * ((r.A + r.Q) ^ 2 * r.Q ^ 4 / r.s₀ ^ 8)) := mul_le_mul_of_nonneg_left hb hK
    _ = _ := by ring

/-- `LoopClaimUniform`'s suffix shape against `LoopClaimSimple`'s. -/
lemma loop_suff_dom {r : LoopRun} (hr : r ∈ loopRunRegion) :
    0 ≤ r.Q * (log ((r.L + r.Q) * r.A * r.R / (r.δ * r.ξ * r.s₀)) / (r.pAP * r.s₀ ^ 2)
        + log (r.Q / r.δ) / r.pAP)
      ∧ 0 ≤ r.Q * log ((r.L + r.Q) * r.A * r.R / (r.δ * r.ξ * r.s₀)) / (r.pAP * r.s₀ ^ 2)
      ∧ r.Q * (log ((r.L + r.Q) * r.A * r.R / (r.δ * r.ξ * r.s₀)) / (r.pAP * r.s₀ ^ 2)
        + log (r.Q / r.δ) / r.pAP)
        ≤ 2 * (r.Q * log ((r.L + r.Q) * r.A * r.R / (r.δ * r.ξ * r.s₀)) / (r.pAP * r.s₀ ^ 2)) := by
  obtain ⟨hQ, hA, hδ, hδ2, hR, hξ, hξ1, hp, hp1, hs, hs1⟩ := hr
  have hQ' : (1 : ℝ) ≤ r.Q := by exact_mod_cast hQ
  have hA' : (2 : ℝ) ≤ r.A := by exact_mod_cast hA
  have hL : (0 : ℝ) ≤ r.L := Nat.cast_nonneg _
  set X1 := (r.L + r.Q) * r.A * r.R / (r.δ * r.ξ * r.s₀) with hX1d
  have hX11 : 1 ≤ X1 := by
    rw [hX1d, le_div_iff₀ (by positivity), one_mul]
    have h1 : r.δ * r.ξ * r.s₀ ≤ 1 := by
      have : r.δ * r.ξ ≤ 1 := by nlinarith
      nlinarith
    have h2 : 1 ≤ (r.L + r.Q) * r.A * r.R := by
      have : 1 ≤ ((r.L : ℝ) + r.Q) * r.A := by nlinarith
      nlinarith
    linarith
  have hQδX : r.Q / r.δ ≤ X1 := by
    rw [hX1d, div_le_div_iff₀ hδ (by positivity)]
    have h1 : (r.Q : ℝ) * (r.ξ * r.s₀) ≤ r.Q := mul_le_of_le_one_right (by linarith) (by nlinarith)
    have h2 : (r.Q : ℝ) ≤ (r.L + r.Q) * r.A * r.R := by
      have : (r.Q : ℝ) ≤ (r.L + r.Q) * r.A := by nlinarith
      nlinarith
    nlinarith [mul_le_mul_of_nonneg_left (h1.trans h2) hδ.le]
  have hl2 : log (r.Q / r.δ) ≤ log X1 := Real.log_le_log (div_pos (by linarith) hδ) hQδX
  have hl10 : 0 ≤ log X1 := Real.log_nonneg hX11
  have hl20 : 0 ≤ log (r.Q / r.δ) := Real.log_nonneg (by rw [le_div_iff₀ hδ]; linarith)
  have hb2 : 1 / r.pAP ≤ 1 / (r.pAP * r.s₀ ^ 2) := by
    apply one_div_le_one_div_of_le (by positivity)
    have : r.s₀ ^ 2 ≤ 1 := by nlinarith
    exact mul_le_of_le_one_right (by positivity) this
  have hf2 : log (r.Q / r.δ) / r.pAP ≤ log X1 / (r.pAP * r.s₀ ^ 2) := by
    rw [div_eq_mul_one_div, div_eq_mul_one_div (log X1)]
    exact mul_le_mul hl2 hb2 (by positivity) hl10
  have hQ0 : (0 : ℝ) ≤ r.Q := by linarith
  refine ⟨by positivity, by positivity, ?_⟩
  have := mul_le_mul_of_nonneg_left (add_le_add (le_refl (log X1 / (r.pAP * r.s₀ ^ 2))) hf2) hQ0
  calc (r.Q : ℝ) * (log X1 / (r.pAP * r.s₀ ^ 2) + log (r.Q / r.δ) / r.pAP)
      ≤ r.Q * (log X1 / (r.pAP * r.s₀ ^ 2) + log X1 / (r.pAP * r.s₀ ^ 2)) := this
    _ = 2 * (r.Q * log X1 / (r.pAP * r.s₀ ^ 2)) := by ring

/-- The `c` the general claim runs `learner_loop_dfa_dist_tol` at, edges crossed at `1/β + |Q|`:
`ε = m/(3(1/β+|Q|+1)N)` at `m = 3ξ/(8R)`. -/
def distC (r : LoopDistRun) : ℝ := r.ξ / (2 * r.R)

/-- The most `D` may hold on any first `k` letters. -/
def distP₀Tol (r : LoopDistRun) : ℝ :=
  1 / max 1 (loopTolInv (r.L + r.Q + 4) r.Q r.A (1 / r.β + r.Q) r.Q r.A r.δ (distC r) (r.ξ / 4)
    r.pAP (1 / 2 - r.s₀))

def distPref (r : LoopDistRun) : ℝ :=
  (r.Q + 1) * (6 * (r.Q + 2) * loopCount (r.L + r.Q + 4) r.Q r.A (1 / r.β + r.Q) r.δ (distC r)
    (r.ξ / 4) r.pAP (1 / 2 - r.s₀))

def distSuff (r : LoopDistRun) : ℝ :=
  (r.Q + 1) * (loopPool (r.L + r.Q + 4) r.Q r.A (1 / r.β + r.Q) r.δ (distC r) (r.ξ / 4) r.pAP
    (1 / 2 - r.s₀) + 1)

def distProbes (r : LoopDistRun) : ℝ :=
  (r.Q + 1) * loopBudget (r.L + r.Q + 4) r.Q r.A (1 / r.β + r.Q) r.δ (distC r)

def distHeldOut (x : LoopDistInputs) : ℕ :=
  ⌈32 * log (16 * ((x.Q : ℝ) + 1) * ((x.Q : ℝ) + 2) / x.δ) / (x.ξ / 4) ^ 2⌉₊

/-- The inverted coordinates of a run in `loopDistRunRegion`. -/
lemma dist_coords {r : LoopDistRun} (hr : r ∈ loopDistRunRegion) :
    0 ≤ (r.L : ℝ) ∧ 1 ≤ (r.Q : ℝ) ∧ 2 ≤ (r.A : ℝ) ∧ (r.A : ℝ) ≤ 1 / r.β ∧ 2 ≤ 1 / r.δ
      ∧ 1 ≤ r.R ∧ 1 ≤ 1 / r.ξ ∧ 1 ≤ 1 / r.pAP ∧ 2 ≤ 1 / r.s₀
      ∧ 0 < distC r ∧ distC r ≤ 1 := by
  obtain ⟨⟨hQ, hA, hδ, hδ2, hR, hξ, hξ1, hp, hp1, hs, hs1⟩, hβ, hβA⟩ := hr
  have hQ' : (1 : ℝ) ≤ r.Q := by exact_mod_cast hQ
  have hA' : (2 : ℝ) ≤ r.A := by exact_mod_cast hA
  refine ⟨Nat.cast_nonneg _, hQ', hA', ?_, (le_div_iff₀ hδ).2 (by linarith), hR,
    (le_div_iff₀ hξ).2 (by linarith), (le_div_iff₀ hp).2 (by linarith),
    (le_div_iff₀ hs).2 (by linarith), by unfold distC; positivity, ?_⟩
  · rw [le_div_iff₀ hβ]; linarith [mul_comm r.β (r.A : ℝ)]
  · unfold distC
    rw [div_le_one (by positivity)]
    linarith

lemma bigO_dist {f g : LoopDistRun → ℝ} (K : ℝ) (hK : 0 < K)
    (h : ∀ r ∈ loopDistRunRegion, 0 ≤ f r ∧ f r ≤ K * g r) :
    f =O[𝓟 loopDistRunRegion] g :=
  isBigO_on K fun r hr => by
    obtain ⟨h0, h1⟩ := h r hr
    exact ⟨h0, nonneg_of_mul_nonneg_right (h0.trans h1) hK, h1⟩

lemma distPref_isBigO : distPref =O[𝓟 loopDistRunRegion] (fun r =>
      r.Q ^ 3 * (r.L + r.Q) ^ 2 * r.R ^ 2
        * log ((r.L + r.Q) * r.R / (r.β * r.δ * r.ξ * r.pAP * r.s₀)) / r.ξ ^ 2
        * (1 / r.s₀ ^ 8 + (1 / r.β + r.Q) ^ 2 * r.Q ^ 4 / r.s₀ ^ 4)) := by
  refine bigO_dist (2 ^ 81) (by norm_num) fun r hr => ?_
  obtain ⟨hL, hQ, hA, hAAe, hiδ, hR, hiξ, hip, his, -, -⟩ := dist_coords hr
  have h := rel_pref_le hL hQ hA hAAe hiδ hR hiξ hip his
  simp only [one_div_one_div, summed_inv, quarter_inv] at h
  exact ⟨h.1, h.2.trans_eq (by ring_nf)⟩

lemma distSuff_isBigO : distSuff =O[𝓟 loopDistRunRegion] (fun r =>
      r.Q * (log ((r.L + r.Q) * r.R / (r.β * r.δ * r.ξ * r.s₀)) / (r.pAP * r.s₀ ^ 2)
        + log (r.Q / r.δ) / r.pAP)) := by
  refine bigO_dist (2 ^ 15) (by norm_num) fun r hr => ?_
  obtain ⟨hL, hQ, hA, hAAe, hiδ, hR, hiξ, hip, his, -, -⟩ := dist_coords hr
  have h := rel_suff_le hL hQ hA hAAe hiδ hR hiξ hip his
  simp only [one_div_one_div, summed_inv, quarter_inv] at h
  exact ⟨h.1, h.2.trans_eq (by ring_nf)⟩

lemma distProbes_isBigO : distProbes =O[𝓟 loopDistRunRegion] (fun r =>
      r.Q ^ 3 * r.A * (1 / r.β + r.Q) * (r.L + r.Q) * r.R / r.ξ * log (r.Q * r.A / r.δ)) := by
  refine bigO_dist (2 ^ 20) (by norm_num) fun r hr => ?_
  obtain ⟨hL, hQ, hA, hAAe, hiδ, hR, hiξ, -, -, -, -⟩ := dist_coords hr
  have h := rel_probes_le hL hQ hA hAAe hiδ hR hiξ
  simp only [one_div_one_div, summed_inv] at h
  exact ⟨h.1, h.2.trans_eq (by ring_nf)⟩

lemma distP₀Tol_pos (r : LoopDistRun) : 0 < distP₀Tol r := by
  unfold distP₀Tol
  exact one_div_pos.2 (lt_of_lt_of_le one_pos (le_max_left _ _))

/-- `LoopClaim`'s `1/p₀Tol` shape, `P = L + |Q|`, `b = 1/β + |Q|`, `lL = log (P Y)` and
`lY = log Y`, in the inverted coordinates. -/
def p₀ShapeInv (P Q A b R iδ iξ ip is lL lY : ℝ) : ℝ :=
  P ^ 4 * lL * lY * (b * R * Q ^ 3 * iξ
      * (R ^ 3 * iξ ^ 3 * (is ^ 11 + Q ^ 6 * b ^ 3 * is ^ 5)
        + R ^ 2 * iξ ^ 2 * iδ * (is ^ 4 + Q ^ 4 * b ^ 2) + iδ * ip ^ 2 * is ^ 2))
    + P ^ 2 * lL * lY ^ 2 * (Q ^ 3 * R ^ 2 * iξ ^ 4 * iδ * (is ^ 8 + Q ^ 4 * b ^ 2 * is ^ 4))
    + P ^ 2 * lY ^ 2 * (Q ^ 5 * A ^ 2 * b ^ 2 * R ^ 2 * iξ ^ 2 * iδ)

lemma le_of_cofactor {t c M : ℝ} (ht : 0 ≤ t) (hc : 1 ≤ c) (h : t * c = M) : t ≤ M := by
  nlinarith [mul_nonneg ht (sub_nonneg.2 hc)]

set_option maxHeartbeats 1000000 in -- eight monomials, each against the largest
/-- Every monomial of `p₀ShapeInv` is at most the largest one. -/
lemma p₀ShapeInv_le {P Q A b R iδ iξ ip is lL lY : ℝ} (hP : 1 ≤ P) (hQ : 1 ≤ Q) (hA : 0 ≤ A)
    (hAb : A ≤ b) (hR : 1 ≤ R) (hiδ : 1 ≤ iδ) (hiξ : 1 ≤ iξ) (hip : 1 ≤ ip) (his : 1 ≤ is)
    (hlY : 1 ≤ lY) (hlL : lY ≤ lL) (hb : 1 ≤ b) :
    p₀ShapeInv P Q A b R iδ iξ ip is lL lY
      ≤ 8 * (P ^ 4 * lL * lY ^ 2 * (Q ^ 9 * b ^ 4 * R ^ 4 * iδ * iξ ^ 4 * ip ^ 2 * is ^ 11)) := by
  have m := @one_le_mul_of_one_le_of_one_le ℝ _
  have p : ∀ {x : ℝ} (n : ℕ), 1 ≤ x → 1 ≤ x ^ n := fun n h => one_le_pow₀ h
  have hlL1 : 1 ≤ lL := hlY.trans hlL
  have hP0 : 0 ≤ P := by linarith
  have hQ0 : 0 ≤ Q := by linarith
  have hR0 : 0 ≤ R := by linarith
  have hb0 : 0 ≤ b := by linarith
  have hδ0 : 0 ≤ iδ := by linarith
  have hξ0 : 0 ≤ iξ := by linarith
  have hp0 : 0 ≤ ip := by linarith
  have hs0 : 0 ≤ is := by linarith
  have hY0 : 0 ≤ lY := by linarith
  have hL0 : 0 ≤ lL := by linarith
  set M := P ^ 4 * lL * lY ^ 2 * (Q ^ 9 * b ^ 4 * R ^ 4 * iδ * iξ ^ 4 * ip ^ 2 * is ^ 11) with hMd
  have t1 : P ^ 4 * lL * lY * (b * R * Q ^ 3 * iξ * (R ^ 3 * iξ ^ 3 * is ^ 11)) ≤ M :=
    le_of_cofactor (by positivity) (c := lY * Q ^ 6 * b ^ 3 * iδ * ip ^ 2)
      (m (m (m (m hlY (p 6 hQ)) (p 3 hb)) hiδ) (p 2 hip)) (by rw [hMd]; ring)
  have t2 : P ^ 4 * lL * lY * (b * R * Q ^ 3 * iξ * (R ^ 3 * iξ ^ 3 * (Q ^ 6 * b ^ 3 * is ^ 5)))
      ≤ M :=
    le_of_cofactor (by positivity) (c := lY * iδ * ip ^ 2 * is ^ 6)
      (m (m (m hlY hiδ) (p 2 hip)) (p 6 his)) (by rw [hMd]; ring)
  have t3 : P ^ 4 * lL * lY * (b * R * Q ^ 3 * iξ * (R ^ 2 * iξ ^ 2 * iδ * is ^ 4)) ≤ M :=
    le_of_cofactor (by positivity) (c := lY * Q ^ 6 * b ^ 3 * R * iξ * ip ^ 2 * is ^ 7)
      (m (m (m (m (m (m hlY (p 6 hQ)) (p 3 hb)) hR) hiξ) (p 2 hip)) (p 7 his))
      (by rw [hMd]; ring)
  have t4 : P ^ 4 * lL * lY * (b * R * Q ^ 3 * iξ * (R ^ 2 * iξ ^ 2 * iδ * (Q ^ 4 * b ^ 2)))
      ≤ M :=
    le_of_cofactor (by positivity) (c := lY * Q ^ 2 * b * R * iξ * ip ^ 2 * is ^ 11)
      (m (m (m (m (m (m hlY (p 2 hQ)) hb) hR) hiξ) (p 2 hip)) (p 11 his)) (by rw [hMd]; ring)
  have t5 : P ^ 4 * lL * lY * (b * R * Q ^ 3 * iξ * (iδ * ip ^ 2 * is ^ 2)) ≤ M :=
    le_of_cofactor (by positivity) (c := lY * Q ^ 6 * b ^ 3 * R ^ 3 * iξ ^ 3 * is ^ 9)
      (m (m (m (m (m hlY (p 6 hQ)) (p 3 hb)) (p 3 hR)) (p 3 hiξ)) (p 9 his)) (by rw [hMd]; ring)
  have t6 : P ^ 2 * lL * lY ^ 2 * (Q ^ 3 * R ^ 2 * iξ ^ 4 * iδ * is ^ 8) ≤ M :=
    le_of_cofactor (by positivity) (c := P ^ 2 * Q ^ 6 * b ^ 4 * R ^ 2 * ip ^ 2 * is ^ 3)
      (m (m (m (m (m (p 2 hP) (p 6 hQ)) (p 4 hb)) (p 2 hR)) (p 2 hip)) (p 3 his))
      (by rw [hMd]; ring)
  have t7 : P ^ 2 * lL * lY ^ 2 * (Q ^ 3 * R ^ 2 * iξ ^ 4 * iδ * (Q ^ 4 * b ^ 2 * is ^ 4)) ≤ M :=
    le_of_cofactor (by positivity) (c := P ^ 2 * Q ^ 2 * b ^ 2 * R ^ 2 * ip ^ 2 * is ^ 7)
      (m (m (m (m (m (p 2 hP) (p 2 hQ)) (p 2 hb)) (p 2 hR)) (p 2 hip)) (p 7 his))
      (by rw [hMd]; ring)
  have t8 : P ^ 2 * lY ^ 2 * (Q ^ 5 * A ^ 2 * b ^ 2 * R ^ 2 * iξ ^ 2 * iδ) ≤ M := by
    have hA2 : A ^ 2 ≤ b ^ 2 := pow_le_pow_left₀ hA hAb 2
    refine le_trans ?_ (le_of_cofactor (t := P ^ 2 * lY ^ 2 * (Q ^ 5 * b ^ 4 * R ^ 2 * iξ ^ 2 * iδ))
      (by positivity) (c := P ^ 2 * lL * Q ^ 4 * R ^ 2 * iξ ^ 2 * ip ^ 2 * is ^ 11)
      (m (m (m (m (m (m (p 2 hP) hlL1) (p 4 hQ)) (p 2 hR)) (p 2 hiξ)) (p 2 hip)) (p 11 his))
      (by rw [hMd]; ring))
    have : 0 ≤ P ^ 2 * lY ^ 2 * Q ^ 5 * b ^ 2 * R ^ 2 * iξ ^ 2 * iδ := by positivity
    nlinarith [mul_le_mul_of_nonneg_left hA2 this]
  have e : p₀ShapeInv P Q A b R iδ iξ ip is lL lY
      = P ^ 4 * lL * lY * (b * R * Q ^ 3 * iξ * (R ^ 3 * iξ ^ 3 * is ^ 11))
        + P ^ 4 * lL * lY * (b * R * Q ^ 3 * iξ * (R ^ 3 * iξ ^ 3 * (Q ^ 6 * b ^ 3 * is ^ 5)))
        + P ^ 4 * lL * lY * (b * R * Q ^ 3 * iξ * (R ^ 2 * iξ ^ 2 * iδ * is ^ 4))
        + P ^ 4 * lL * lY * (b * R * Q ^ 3 * iξ * (R ^ 2 * iξ ^ 2 * iδ * (Q ^ 4 * b ^ 2)))
        + P ^ 4 * lL * lY * (b * R * Q ^ 3 * iξ * (iδ * ip ^ 2 * is ^ 2))
        + P ^ 2 * lL * lY ^ 2 * (Q ^ 3 * R ^ 2 * iξ ^ 4 * iδ * is ^ 8)
        + P ^ 2 * lL * lY ^ 2 * (Q ^ 3 * R ^ 2 * iξ ^ 4 * iδ * (Q ^ 4 * b ^ 2 * is ^ 4))
        + P ^ 2 * lY ^ 2 * (Q ^ 5 * A ^ 2 * b ^ 2 * R ^ 2 * iξ ^ 2 * iδ) := by
    unfold p₀ShapeInv; ring
  rw [e]
  linarith

/-- `LoopClaim`'s shape for `1/p₀Tol`. -/
def p₀Shape (r : LoopDistRun) : ℝ :=
  (r.L + r.Q) ^ 4 * log ((r.L + r.Q) * r.Q * r.R / (r.β * r.δ * r.ξ * r.pAP * r.s₀))
      * log (r.Q * r.R / (r.β * r.δ * r.ξ * r.pAP * r.s₀)) * r.Q ^ 3 * r.R * (1 / r.β + r.Q)
      / r.ξ
      * (r.R ^ 3 / r.ξ ^ 3 * (1 / r.s₀ ^ 11 + r.Q ^ 6 * (1 / r.β + r.Q) ^ 3 / r.s₀ ^ 5)
        + r.R ^ 2 / (r.ξ ^ 2 * r.δ) * (1 / r.s₀ ^ 4 + r.Q ^ 4 * (1 / r.β + r.Q) ^ 2)
        + 1 / (r.δ * r.pAP ^ 2 * r.s₀ ^ 2))
    + (r.L + r.Q) ^ 2 * log ((r.L + r.Q) * r.Q * r.R / (r.β * r.δ * r.ξ * r.pAP * r.s₀))
      * log (r.Q * r.R / (r.β * r.δ * r.ξ * r.pAP * r.s₀)) ^ 2 * r.Q ^ 3 * r.R ^ 2
      / (r.ξ ^ 4 * r.δ) * (1 / r.s₀ ^ 8 + r.Q ^ 4 * (1 / r.β + r.Q) ^ 2 / r.s₀ ^ 4)
    + (r.L + r.Q) ^ 2 * log (r.Q * r.R / (r.β * r.δ * r.ξ * r.pAP * r.s₀)) ^ 2
      * r.Q ^ 5 * r.A ^ 2 * (1 / r.β + r.Q) ^ 2 * r.R ^ 2 / (r.ξ ^ 2 * r.δ)

lemma p₀Shape_eq (r : LoopDistRun) :
    p₀Shape r = p₀ShapeInv (r.L + r.Q) r.Q r.A (1 / r.β + r.Q) r.R (1 / r.δ) (1 / r.ξ)
      (1 / r.pAP) (1 / r.s₀) (log ((r.L + r.Q) * r.Q * r.R / (r.β * r.δ * r.ξ * r.pAP * r.s₀)))
      (log (r.Q * r.R / (r.β * r.δ * r.ξ * r.pAP * r.s₀))) := by
  unfold p₀Shape p₀ShapeInv; ring

set_option maxHeartbeats 1000000 in -- three terms, each matched against its shape
lemma distP₀Tol_isBigO : (fun r => 1 / distP₀Tol r) =O[𝓟 loopDistRunRegion] p₀Shape := by
  refine bigO_dist (2 ^ 140) (by norm_num) fun r hr => ?_
  obtain ⟨hL, hQ, hA, hAAe, hiδ, hR, hiξ, hip, his, hc0, hc1⟩ := dist_coords hr
  obtain ⟨⟨-, -, hδ, hδ2, -, hξ, hξ1, hp, hp1, hs, hs1⟩, hβ, -⟩ := hr
  obtain ⟨h1, h2, h3⟩ := rel_terms_le (nQ := r.Q) (nA := r.A) hQ hA hAAe hiδ hR hiξ hip his le_rfl
    le_rfl
  simp only [one_div_one_div, summed_inv, quarter_inv] at h1 h2 h3
  have hLQ1 : (1 : ℝ) ≤ r.L + r.Q := by linarith
  set ℓ : ℝ := (r.L : ℝ) + r.Q + 4 with hℓd
  have hℓ1 : 1 ≤ ℓ := by linarith
  obtain ⟨hT1, -⟩ := loopTolInv_le (ℓ := ℓ) (Q := r.Q) (A := r.A) (Ae := 1 / r.β + r.Q)
    (δ := r.δ) (c := distC r) (ξ := r.ξ / 4) (pAP := r.pAP) (η₀ := 1 / 2 - r.s₀) r.Q r.A hℓ1
    (by linarith) (by linarith) (by linarith) hδ (by linarith) hc0 hc1 (by positivity) hp hp1
    (by linarith) (by linarith)
  have hinv : 1 / distP₀Tol r = loopTolInv ℓ r.Q r.A (1 / r.β + r.Q) r.Q r.A r.δ (distC r)
      (r.ξ / 4) r.pAP (1 / 2 - r.s₀) := by
    unfold distP₀Tol
    rw [one_div_one_div, max_eq_right hT1]
  rw [hinv]
  refine ⟨by linarith, ?_⟩
  simp only [loopTolInv]
  set t := loopMassTerms r.Q r.A (1 / r.β + r.Q) r.Q r.A r.δ (distC r) (r.ξ / 4) r.pAP
    (1 / 2 - r.s₀)
  obtain ⟨ht1, ht2, ht3⟩ := loopMassTerms_ge (Q := r.Q) (A := r.A) (Ae := 1 / r.β + r.Q)
    (δ := r.δ) (c := distC r) (ξ := r.ξ / 4) (pAP := r.pAP) (η₀ := 1 / 2 - r.s₀) r.Q r.A
    (by linarith) (by linarith) (by linarith) hδ (by linarith) hc0 hc1 (by positivity) hp hp1
    (by linarith) (by linarith)
  set Y : ℝ := r.Q * r.R * (1 / r.β) * (1 / r.δ) * (1 / r.ξ) * (1 / r.pAP) * (1 / r.s₀) with hYd
  have hY8 : 8 ≤ Y := by
    have h1 : 1 ≤ (r.Q : ℝ) * r.R := one_le_mul_of_one_le_of_one_le hQ hR
    have h2 : 2 ≤ (r.Q : ℝ) * r.R * (1 / r.β) := by nlinarith only [h1, hA, hAAe]
    have h3 : 4 ≤ (r.Q : ℝ) * r.R * (1 / r.β) * (1 / r.δ) := by nlinarith only [h2, hiδ]
    have h4 : 4 ≤ (r.Q : ℝ) * r.R * (1 / r.β) * (1 / r.δ) * (1 / r.ξ) := by nlinarith only [h3, hiξ]
    have h5 : 4 ≤ (r.Q : ℝ) * r.R * (1 / r.β) * (1 / r.δ) * (1 / r.ξ) * (1 / r.pAP) := by
      nlinarith only [h4, hip]
    rw [hYd]; nlinarith only [h5, his]
  have hlY : 1 ≤ log Y := one_le_log_of_three_le (by linarith)
  have hlLY : log Y ≤ log ((r.L + r.Q) * Y) :=
    Real.log_le_log (by linarith) (le_mul_of_one_le_left (by linarith) hLQ1)
  have hΛ : 1 + log ℓ ≤ 2 * log ((r.L + r.Q) * Y) := by
    have : log ℓ ≤ log ((r.L + r.Q) * Y) :=
      Real.log_le_log (by positivity) (by rw [hℓd]; nlinarith only [hY8, hLQ1])
    linarith
  have hΛ0 : 0 ≤ 1 + log ℓ := by have := Real.log_nonneg hℓ1; linarith
  have hℓ4 : ℓ ^ 4 ≤ 625 * ((r.L : ℝ) + r.Q) ^ 4 := by
    have := pow_le_pow_left₀ (by positivity) (show ℓ ≤ 5 * ((r.L : ℝ) + r.Q) by linarith) 4
    linarith [show (5 * ((r.L : ℝ) + r.Q)) ^ 4 = 625 * ((r.L : ℝ) + r.Q) ^ 4 by ring]
  have hℓ2 : ℓ ^ 2 ≤ 25 * ((r.L : ℝ) + r.Q) ^ 2 := by
    have := pow_le_pow_left₀ (by positivity) (show ℓ ≤ 5 * ((r.L : ℝ) + r.Q) by linarith) 2
    linarith [show (5 * ((r.L : ℝ) + r.Q)) ^ 2 = 25 * ((r.L : ℝ) + r.Q) ^ 2 by ring]
  set lY := log Y with hlYd
  set lL := log ((r.L + r.Q) * Y) with hlLd
  have hlL0 : 0 ≤ lL := by linarith
  have hb0 : 0 ≤ 1 / r.β + r.Q := by linarith
  have hδ0 : 0 ≤ 1 / r.δ := by linarith
  have hip0 : 0 ≤ 1 / r.pAP := by linarith
  set F₁ := (1 / r.β + r.Q) * r.R * r.Q ^ 3 * (1 / r.ξ)
    * (r.R ^ 3 * (1 / r.ξ) ^ 3 * ((1 / r.s₀) ^ 11 + r.Q ^ 6 * (1 / r.β + r.Q) ^ 3 * (1 / r.s₀) ^ 5)
      + r.R ^ 2 * (1 / r.ξ) ^ 2 * (1 / r.δ) * ((1 / r.s₀) ^ 4 + r.Q ^ 4 * (1 / r.β + r.Q) ^ 2)
      + 1 / r.δ * (1 / r.pAP) ^ 2 * (1 / r.s₀) ^ 2) with hF₁d
  set F₂ := r.Q ^ 3 * r.R ^ 2 * (1 / r.ξ) ^ 4 * (1 / r.δ)
    * ((1 / r.s₀) ^ 8 + r.Q ^ 4 * (1 / r.β + r.Q) ^ 2 * (1 / r.s₀) ^ 4) with hF₂d
  set F₃ := r.Q ^ 5 * r.A ^ 2 * (1 / r.β + r.Q) ^ 2 * r.R ^ 2 * (1 / r.ξ) ^ 2 * (1 / r.δ)
    with hF₃d
  have hF₁ : 0 ≤ F₁ := by rw [hF₁d]; positivity
  have hF₂ : 0 ≤ F₂ := by rw [hF₂d]; positivity
  have hF₃ : 0 ≤ F₃ := by rw [hF₃d]; positivity
  have a1 : ℓ ^ 4 * (1 + log ℓ) * t.1
      ≤ 2 ^ 140 * (((r.L : ℝ) + r.Q) ^ 4 * lL * lY * F₁) := by
    have := mul_le_mul (mul_le_mul hℓ4 hΛ hΛ0 (by positivity)) h1 (zero_le_one.trans ht1)
      (by positivity)
    calc ℓ ^ 4 * (1 + log ℓ) * t.1
        ≤ 625 * ((r.L : ℝ) + r.Q) ^ 4 * (2 * lL) * (2 ^ 129 * (F₁ * lY)) := this
      _ = 1250 * 2 ^ 129 * (((r.L : ℝ) + r.Q) ^ 4 * lL * lY * F₁) := by ring
      _ ≤ _ := mul_le_mul_of_nonneg_right (by norm_num) (by positivity)
  have a2 : ℓ ^ 2 * (1 + log ℓ) * t.2.1
      ≤ 2 ^ 140 * (((r.L : ℝ) + r.Q) ^ 2 * lL * lY ^ 2 * F₂) := by
    have := mul_le_mul (mul_le_mul hℓ2 hΛ hΛ0 (by positivity)) h2 ht2 (by positivity)
    calc ℓ ^ 2 * (1 + log ℓ) * t.2.1
        ≤ 25 * ((r.L : ℝ) + r.Q) ^ 2 * (2 * lL) * (2 ^ 90 * (F₂ * lY ^ 2)) := this
      _ = 50 * 2 ^ 90 * (((r.L : ℝ) + r.Q) ^ 2 * lL * lY ^ 2 * F₂) := by ring
      _ ≤ _ := mul_le_mul_of_nonneg_right (by norm_num) (by positivity)
  have a3 : ℓ ^ 2 * t.2.2 ≤ 2 ^ 140 * (((r.L : ℝ) + r.Q) ^ 2 * lY ^ 2 * F₃) := by
    have := mul_le_mul hℓ2 h3 ht3 (by positivity)
    calc ℓ ^ 2 * t.2.2 ≤ 25 * ((r.L : ℝ) + r.Q) ^ 2 * (2 ^ 46 * (F₃ * lY ^ 2)) := this
      _ = 25 * 2 ^ 46 * (((r.L : ℝ) + r.Q) ^ 2 * lY ^ 2 * F₃) := by ring
      _ ≤ _ := mul_le_mul_of_nonneg_right (by norm_num) (by positivity)
  have e1 : ((r.L : ℝ) + r.Q) * Y
      = (r.L + r.Q) * r.Q * r.R / (r.β * r.δ * r.ξ * r.pAP * r.s₀) := by
    rw [hYd]; field_simp
  have e2 : Y = r.Q * r.R / (r.β * r.δ * r.ξ * r.pAP * r.s₀) := by
    rw [hYd]; field_simp
  have hsum := add_le_add (add_le_add a1 a2) a3
  refine hsum.trans (le_of_eq ?_)
  rw [p₀Shape_eq, ← e1, ← e2, ← hlLd, ← hlYd, hF₁d, hF₂d, hF₃d]
  unfold p₀ShapeInv
  ring

/-- `p₀Shape` against its largest monomial. -/
lemma p₀Shape_le {r : LoopDistRun} (hr : r ∈ loopDistRunRegion) :
    0 ≤ p₀Shape r ∧ p₀Shape r ≤ 8 * ((r.L + r.Q) ^ 4 * r.Q ^ 9 * (1 / r.β + r.Q) ^ 4 * r.R ^ 4
      * log ((r.L + r.Q) * r.Q * r.R / (r.β * r.δ * r.ξ * r.pAP * r.s₀))
      * log (r.Q * r.R / (r.β * r.δ * r.ξ * r.pAP * r.s₀)) ^ 2
      / (r.δ * r.ξ ^ 4 * r.pAP ^ 2 * r.s₀ ^ 11)) := by
  obtain ⟨hL, hQ, hA, hAAe, hiδ, hR, hiξ, hip, his, -, -⟩ := dist_coords hr
  rw [p₀Shape_eq]
  set Y : ℝ := r.Q * r.R / (r.β * r.δ * r.ξ * r.pAP * r.s₀) with hYd
  have hYe : Y = r.Q * r.R * (1 / r.β) * (1 / r.δ) * (1 / r.ξ) * (1 / r.pAP) * (1 / r.s₀) := by
    rw [hYd]; field_simp
  have hY8 : 8 ≤ Y := by
    have h1 : 1 ≤ (r.Q : ℝ) * r.R := one_le_mul_of_one_le_of_one_le hQ hR
    have h2 : 2 ≤ (r.Q : ℝ) * r.R * (1 / r.β) := by nlinarith only [h1, hA, hAAe]
    have h3 : 4 ≤ (r.Q : ℝ) * r.R * (1 / r.β) * (1 / r.δ) := by nlinarith only [h2, hiδ]
    have h4 : 4 ≤ (r.Q : ℝ) * r.R * (1 / r.β) * (1 / r.δ) * (1 / r.ξ) := by nlinarith only [h3, hiξ]
    have h5 : 4 ≤ (r.Q : ℝ) * r.R * (1 / r.β) * (1 / r.δ) * (1 / r.ξ) * (1 / r.pAP) := by
      nlinarith only [h4, hip]
    rw [hYe]; nlinarith only [h5, his]
  have hLQ1 : (1 : ℝ) ≤ r.L + r.Q := by linarith
  have eL : (r.L + r.Q) * r.Q * r.R / (r.β * r.δ * r.ξ * r.pAP * r.s₀) = (r.L + r.Q) * Y := by
    rw [hYd]; ring
  have hlY : 1 ≤ log Y := one_le_log_of_three_le (by linarith)
  have hlL : log Y ≤ log ((r.L + r.Q) * Y) :=
    Real.log_le_log (by linarith) (le_mul_of_one_le_left (by linarith) hLQ1)
  rw [eL]
  have hb : 0 ≤ 1 / r.β + r.Q := by linarith
  have hδ0 : 0 ≤ 1 / r.δ := by linarith
  have hip0 : 0 ≤ 1 / r.pAP := by linarith
  have hlL0 : 0 ≤ log ((r.L + r.Q) * Y) := by linarith
  refine ⟨by unfold p₀ShapeInv; positivity, ?_⟩
  refine (p₀ShapeInv_le hLQ1 hQ (by linarith) (by linarith) hR (by linarith) hiξ hip
    (by linarith) hlY hlL (by linarith)).trans (le_of_eq ?_)
  ring

lemma distHeldOut_isBigO : (fun x => (distHeldOut x : ℝ)) =O[𝓟 loopDistRegion]
    (fun x => log (x.Q / x.δ) / x.ξ ^ 2) := by
  refine isBigO_on (2 ^ 13) fun x hx => ?_
  obtain ⟨⟨hQ, -, hδ, hδ2, -, hξ, hξ1, -, -, -, -⟩, -, -⟩ := hx
  have hQ' : (1 : ℝ) ≤ x.Q := by exact_mod_cast hQ
  have hiδ : 2 ≤ 1 / x.δ := (le_div_iff₀ hδ).2 (by linarith)
  have hiξ : 1 ≤ 1 / x.ξ := (le_div_iff₀ hξ).2 (by linarith)
  have h := claim_heldOut_le hQ' hiδ (by linarith : 1 ≤ 4 * (1 / x.ξ))
  rw [one_div_one_div, quarter_inv] at h
  have hl : 0 ≤ log (x.Q / x.δ) := by
    rw [← mul_one_div]; exact Real.log_nonneg (by nlinarith)
  refine ⟨Nat.cast_nonneg _, div_nonneg hl (sq_nonneg _), h.trans_eq ?_⟩
  field_simp
  ring

end OrthoDFA

end
