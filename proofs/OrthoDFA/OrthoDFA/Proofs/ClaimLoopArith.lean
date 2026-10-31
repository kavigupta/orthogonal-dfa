import OrthoDFA.Claims.Loop
import OrthoDFA.Proofs.LoopDFAArith
import OrthoDFA.Proofs.BigO

/-!
# The learner's costs, asymptotically: the arithmetic

In the inverted coordinates `iδ = 1/δ`, `ic = 1/c`, `iξ = 1/ξ`, `ip = 1/pAP`, `is = 1/s₀`, `loopΘ`
is a polynomial, so `loopCount` is `ℓ² Θ²/s⁴` times the log of a polynomial, `loopPool` a sum of
logs of polynomials, and `loopMassInv` a polynomial times `log² X` (`claim_mass_le`). `Ae ≥ |Σ|` is
the rate edges are crossed at, `1/β` over an input distribution.
-/

noncomputable section

namespace OrthoDFA

open Real

lemma loopΘ_inv (Q A ic iξ is : ℝ) :
    loopΘ Q A (1 / ic) (1 / iξ) (1 / 2 - 1 / is)
      = (4 * (A + 1) * ic + 8 * iξ) * is ^ 2 / 2 + 8 * (4 * (A + 1) * ic + 8 * iξ)
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
      = 1901000 * (Q + 2) * loopΘ Q Ae (1 / ic) (1 / iξ) (1 / 2 - 1 / is) ^ 2 * ℓ ^ 2
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
        ≤ 2 ^ 17 * ((A * ic + iξ) * is ^ 2 + Q ^ 2 * A * ic) := by
  rw [loopΘ_inv]
  have hAc : 1 ≤ A * ic := one_le_mul_of_one_le_of_one_le hA hic
  have hs2 : 1 ≤ is ^ 2 := one_le_pow₀ his
  have hu : 0 ≤ (A * ic + iξ) := by positivity
  have h1 : (4 * (A + 1) * ic + 8 * iξ) * is ^ 2 / 2 ≤ 4 * ((A * ic + iξ) * is ^ 2) := by
    have : 4 * (A + 1) * ic + 8 * iξ ≤ 8 * (A * ic + iξ) := by nlinarith
    nlinarith
  have h2 : 8 * (4 * (A + 1) * ic + 8 * iξ) ≤ 64 * ((A * ic + iξ) * is ^ 2) := by nlinarith
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
  have h : (A * ic + iξ) * is ^ 2 + Q ^ 2 * A * ic ≤ (X * X + X) * X ^ 2 + X ^ 2 * X * X := by
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
        ≤ 2 ^ 75 * (Q ^ 3 * (L + Q) ^ 2 * log ((L + Q) * Ae * iδ * ic * iξ * ip * is)
          * ((Ae ^ 2 * ic ^ 2 + iξ ^ 2) * is ^ 8 + Q ^ 4 * Ae ^ 2 * ic ^ 2 * is ^ 4)) := by
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
  have hG : Θ ^ 2 * is ^ 4 ≤ 2 ^ 36 * ((Ae ^ 2 * ic ^ 2 + iξ ^ 2) * is ^ 8
      + Q ^ 4 * Ae ^ 2 * ic ^ 2 * is ^ 4) := by
    have h1 : Θ ^ 2 ≤ (2 ^ 17 * ((Ae * ic + iξ) * is ^ 2 + Q ^ 2 * Ae * ic)) ^ 2 :=
      pow_le_pow_left₀ (by linarith) hΘu 2
    have h2 : ((Ae * ic + iξ) * is ^ 2 + Q ^ 2 * Ae * ic) ^ 2
        ≤ 2 * ((Ae * ic + iξ) ^ 2 * is ^ 4 + Q ^ 4 * Ae ^ 2 * ic ^ 2) := by
      linarith [sq_nonneg ((Ae * ic + iξ) * is ^ 2 - Q ^ 2 * Ae * ic)]
    have h3 : (Ae * ic + iξ) ^ 2 ≤ 2 * (Ae ^ 2 * ic ^ 2 + iξ ^ 2) := by
      linarith [sq_nonneg (Ae * ic - iξ)]
    have hs4 : 0 ≤ is ^ 4 := by positivity
    have hs8 : is ^ 4 * is ^ 4 = is ^ 8 := by ring
    have h4 := mul_le_mul_of_nonneg_right h1 hs4
    have h5 := mul_le_mul_of_nonneg_right h2 hs4
    have h6 := mul_le_mul_of_nonneg_right h3 (by positivity : (0 : ℝ) ≤ is ^ 4 * is ^ 4)
    have h7 : 0 ≤ Q ^ 4 * Ae ^ 2 * ic ^ 2 * is ^ 4 := by positivity
    linarith only [h4, h5, h6, h7]
  have hcnt0 : 0 ≤ 1901000 * (Q + 2) * Θ ^ 2 * ℓ ^ 2 * log Y * is ^ 4 := by positivity
  constructor
  · positivity
  · have hlX0 : 0 ≤ log X := Real.log_nonneg (by linarith)
    have e : (Q + 1) * (6 * (Q + 2) * (1901000 * (Q + 2) * Θ ^ 2 * ℓ ^ 2 * log Y * is ^ 4))
        = 11406000 * ((Q + 1) * (Q + 2) ^ 2 * ℓ ^ 2 * log Y) * (Θ ^ 2 * is ^ 4) := by ring
    rw [e]
    have a1 : Q + 1 ≤ 2 * Q := by linarith only [hQ]
    have a2 : Q + 2 ≤ 3 * Q := by linarith only [hQ]
    have h1 : (Q + 1) * (Q + 2) ^ 2 * ℓ ^ 2 * log Y
        ≤ (2 * Q) * (3 * Q) ^ 2 * (5 * (L + Q)) ^ 2 * (95 * log X) := by
      gcongr
    have hG0 : 0 ≤ (Ae ^ 2 * ic ^ 2 + iξ ^ 2) * is ^ 8 + Q ^ 4 * Ae ^ 2 * ic ^ 2 * is ^ 4 := by
      positivity
    have hq0 : 0 ≤ (Q + 1) * (Q + 2) ^ 2 * ℓ ^ 2 * log Y := by positivity
    calc 11406000 * ((Q + 1) * (Q + 2) ^ 2 * ℓ ^ 2 * log Y) * (Θ ^ 2 * is ^ 4)
        ≤ 11406000 * ((2 * Q) * (3 * Q) ^ 2 * (5 * (L + Q)) ^ 2 * (95 * log X))
          * (2 ^ 36 * ((Ae ^ 2 * ic ^ 2 + iξ ^ 2) * is ^ 8
            + Q ^ 4 * Ae ^ 2 * ic ^ 2 * is ^ 4)) := by
          gcongr
      _ = (11406000 * 2 * 9 * 25 * 95 * 2 ^ 36) * (Q ^ 3 * (L + Q) ^ 2 * log X
          * ((Ae ^ 2 * ic ^ 2 + iξ ^ 2) * is ^ 8 + Q ^ 4 * Ae ^ 2 * ic ^ 2 * is ^ 4)) := by ring
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


lemma rel_prod {P A iδ R iξ ip is : ℝ} (hP : 1 ≤ P) (hA : 2 ≤ A) (hiδ : 1 ≤ iδ)
    (hR : 1 ≤ R) (hiξ : 1 ≤ iξ) (hip : 1 ≤ ip) (his : 1 ≤ is) :
    2 ≤ P * A * iδ * R * iξ * ip * is ∧ P * iξ ≤ P * A * iδ * R * iξ * ip * is := by
  have a1 : 2 ≤ A * iδ := by nlinarith
  have a2 : 2 ≤ A * iδ * R := by nlinarith
  have a3 : 2 ≤ A * iδ * R * ip := by nlinarith
  have a4 : 2 ≤ A * iδ * R * ip * is := by nlinarith
  have hp : 1 ≤ P * iξ := one_le_mul_of_one_le_of_one_le hP hiξ
  rw [show P * A * iδ * R * iξ * ip * is = (P * iξ) * (A * iδ * R * ip * is) by ring]
  constructor <;> nlinarith

lemma log_le_of_le_sq {Y Z : ℝ} (hY : 2 ≤ Y) (hZ : 0 < Z) (hZY : Z ≤ 8 * Y ^ 2) :
    log Z ≤ 5 * log Y := by
  have h3 : (2 : ℝ) ^ 3 ≤ Y ^ 3 := pow_le_pow_left₀ (by norm_num) hY 3
  have h : Z ≤ Y ^ 5 := hZY.trans (by
    rw [show Y ^ 5 = Y ^ 3 * Y ^ 2 by ring]
    exact mul_le_mul_of_nonneg_right (by norm_num at h3; exact h3) (sq_nonneg Y))
  have := log_le_mul_log_of_le_pow 5 hZ h
  push_cast at this
  exact this

lemma rel_sq_le {P Q A iδ R iξ ip is ic : ℝ} (hQ : 1 ≤ Q) (hQP : Q ≤ P) (hA : 2 ≤ A)
    (hiδ : 1 ≤ iδ) (hR : 1 ≤ R) (hiξ : 1 ≤ iξ) (hip : 1 ≤ ip) (his : 1 ≤ is) (hic : 1 ≤ ic)
    (hicR : ic ≤ 4 * R * Q * iξ) :
    0 ≤ log (P * A * iδ * ic * (2 * iξ) * ip * is)
      ∧ log (P * A * iδ * ic * (2 * iξ) * ip * is) ≤ 5 * log (P * A * iδ * R * iξ * ip * is) := by
  obtain ⟨hY2, hPY⟩ := rel_prod (hQ.trans hQP) hA hiδ hR hiξ hip his
  have hZ1 : 1 ≤ P * A * iδ * ic * (2 * iξ) * ip * is := by
    refine one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le
      (one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le
        (one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le ?_ ?_) ?_) ?_) ?_) ?_) ?_
      <;> linarith
  have hQξ : Q * iξ ≤ P * iξ := by gcongr
  have hZY : P * A * iδ * ic * (2 * iξ) * ip * is ≤ 8 * (P * A * iδ * R * iξ * ip * is) ^ 2 := by
    have h1 : P * A * iδ * ic * (2 * iξ) * ip * is
        ≤ (P * A * iδ * (2 * iξ) * ip * is) * (4 * R * Q * iξ) := by
      rw [show P * A * iδ * ic * (2 * iξ) * ip * is = (P * A * iδ * (2 * iξ) * ip * is) * ic by
        ring]
      exact mul_le_mul_of_nonneg_left hicR (by
        have : 0 ≤ P := by linarith
        have : 0 ≤ A := by linarith
        have : 0 ≤ iδ := by linarith
        have : 0 ≤ iξ := by linarith
        have : 0 ≤ ip := by linarith
        have : 0 ≤ is := by linarith
        positivity)
    have e : (P * A * iδ * (2 * iξ) * ip * is) * (4 * R * Q * iξ)
        = 8 * (Q * iξ) * (P * A * iδ * R * iξ * ip * is) := by ring
    have hY0 : 0 ≤ P * A * iδ * R * iξ * ip * is := by linarith
    nlinarith [mul_le_mul_of_nonneg_right (hQξ.trans hPY) hY0]
  exact ⟨Real.log_nonneg hZ1, log_le_of_le_sq hY2 (by linarith) hZY⟩

set_option maxHeartbeats 1000000 in -- the bounds chain through many products
lemma rel_pref_le {L Q A Ae iδ R iξ ip is ic : ℝ} (hL : 0 ≤ L) (hQ : 1 ≤ Q) (hA : 2 ≤ A)
    (hAAe : A ≤ Ae)
    (hiδ : 2 ≤ iδ) (hR : 1 ≤ R) (hiξ : 1 ≤ iξ) (hip : 1 ≤ ip) (his : 2 ≤ is) (hic : 1 ≤ ic)
    (hicR : ic ≤ 4 * R * Q * iξ) :
    0 ≤ (Q + 1) * (6 * (Q + 2) * loopCount (L + Q + 4) Q A Ae (1 / iδ) (1 / ic) (1 / (2 * iξ))
        (1 / ip) (1 / 2 - 1 / is))
      ∧ (Q + 1) * (6 * (Q + 2) * loopCount (L + Q + 4) Q A Ae (1 / iδ) (1 / ic) (1 / (2 * iξ))
        (1 / ip) (1 / 2 - 1 / is))
        ≤ 2 ^ 82 * (Q ^ 5 * (L + Q) ^ 2 * Ae ^ 2 * R ^ 2
          * log ((L + Q) * Ae * iδ * R * iξ * ip * is) * iξ ^ 2 * (is ^ 8 + Q ^ 4 * is ^ 4)) := by
  obtain ⟨h0, h⟩ := claim_pref_le (iξ := 2 * iξ) hL hQ hA hAAe hiδ hic (by linarith) hip his
  refine ⟨h0, h.trans ?_⟩
  obtain ⟨hl0, hl⟩ := rel_sq_le (P := L + Q) (iδ := iδ) (is := is) hQ (by linarith) (hA.trans hAAe)
    (by linarith) hR hiξ hip (by linarith) hic hicR
  have hY0 : 0 ≤ log ((L + Q) * Ae * iδ * R * iξ * ip * is) := by linarith
  have hic2 : ic ^ 2 ≤ 16 * (R ^ 2 * Q ^ 2 * iξ ^ 2) :=
    (pow_le_pow_left₀ (by linarith) hicR 2).trans_eq (by ring)
  have hQ2 : 1 ≤ Q ^ 2 := one_le_pow₀ hQ
  have hA2 : 1 ≤ Ae ^ 2 := one_le_pow₀ (by linarith)
  have hR2 : 1 ≤ R ^ 2 := one_le_pow₀ hR
  have hξ2 : 0 ≤ iξ ^ 2 := sq_nonneg _
  set P := Q ^ 2 * Ae ^ 2 * R ^ 2 * iξ ^ 2 with hPd
  have a1 : Ae ^ 2 * ic ^ 2 ≤ 16 * P :=
    (mul_le_mul_of_nonneg_left hic2 (sq_nonneg Ae)).trans_eq (by rw [hPd]; ring)
  have a2 : (2 * iξ) ^ 2 ≤ 4 * P := by
    have : 1 ≤ Q ^ 2 * Ae ^ 2 * R ^ 2 :=
      one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le hQ2 hA2) hR2
    rw [hPd]; nlinarith
  have a3 : Q ^ 4 * Ae ^ 2 * ic ^ 2 ≤ 16 * (Q ^ 4 * P) := by
    have := mul_le_mul_of_nonneg_left a1 (by positivity : (0 : ℝ) ≤ Q ^ 4)
    linarith
  have hs8 : 0 ≤ is ^ 8 := by positivity
  have hs4 : 0 ≤ is ^ 4 := by positivity
  have hBr : (Ae ^ 2 * ic ^ 2 + (2 * iξ) ^ 2) * is ^ 8 + Q ^ 4 * Ae ^ 2 * ic ^ 2 * is ^ 4
      ≤ 20 * (P * (is ^ 8 + Q ^ 4 * is ^ 4)) := by
    have b1 := mul_le_mul_of_nonneg_right (add_le_add a1 a2) hs8
    have b2 := mul_le_mul_of_nonneg_right a3 hs4
    have b3 : 0 ≤ Q ^ 4 * P * is ^ 4 := by rw [hPd]; positivity
    linarith
  have hBr0 : 0 ≤ (Ae ^ 2 * ic ^ 2 + (2 * iξ) ^ 2) * is ^ 8 + Q ^ 4 * Ae ^ 2 * ic ^ 2 * is ^ 4 := by
    positivity
  have hK : 0 ≤ Q ^ 3 * (L + Q) ^ 2 := by positivity
  have hT : 0 ≤ Q ^ 5 * (L + Q) ^ 2 * Ae ^ 2 * R ^ 2
      * log ((L + Q) * Ae * iδ * R * iξ * ip * is) * iξ ^ 2 * (is ^ 8 + Q ^ 4 * is ^ 4) := by
    have : 0 ≤ Q ^ 5 * (L + Q) ^ 2 * Ae ^ 2 * R ^ 2 := by positivity
    have : 0 ≤ iξ ^ 2 * (is ^ 8 + Q ^ 4 * is ^ 4) := by positivity
    calc (0 : ℝ) ≤ (Q ^ 5 * (L + Q) ^ 2 * Ae ^ 2 * R ^ 2
          * log ((L + Q) * Ae * iδ * R * iξ * ip * is))
          * (iξ ^ 2 * (is ^ 8 + Q ^ 4 * is ^ 4)) := by positivity
      _ = _ := by ring
  have h1 := mul_le_mul (mul_le_mul_of_nonneg_left hl hK) hBr hBr0 (mul_nonneg hK (by linarith))
  calc 2 ^ 75 * (Q ^ 3 * (L + Q) ^ 2 * log ((L + Q) * Ae * iδ * ic * (2 * iξ) * ip * is)
        * ((Ae ^ 2 * ic ^ 2 + (2 * iξ) ^ 2) * is ^ 8 + Q ^ 4 * Ae ^ 2 * ic ^ 2 * is ^ 4))
      ≤ 2 ^ 75 * (Q ^ 3 * (L + Q) ^ 2 * (5 * log ((L + Q) * Ae * iδ * R * iξ * ip * is))
        * (20 * (P * (is ^ 8 + Q ^ 4 * is ^ 4)))) := by gcongr
    _ = 100 * 2 ^ 75 * (Q ^ 5 * (L + Q) ^ 2 * Ae ^ 2 * R ^ 2
          * log ((L + Q) * Ae * iδ * R * iξ * ip * is) * iξ ^ 2 * (is ^ 8 + Q ^ 4 * is ^ 4)) := by
        rw [hPd]; ring
    _ ≤ _ := mul_le_mul_of_nonneg_right (by norm_num) hT

lemma rel_suff_le {L Q A Ae iδ R iξ ip is ic : ℝ} (hL : 0 ≤ L) (hQ : 1 ≤ Q) (hA : 2 ≤ A)
    (hAAe : A ≤ Ae)
    (hiδ : 2 ≤ iδ) (hR : 1 ≤ R) (hiξ : 1 ≤ iξ) (hip : 1 ≤ ip) (his : 2 ≤ is) (hic : 1 ≤ ic)
    (hicR : ic ≤ 4 * R * Q * iξ) :
    0 ≤ (Q + 1) * (loopPool (L + Q + 4) Q A Ae (1 / iδ) (1 / ic) (1 / (2 * iξ)) (1 / ip)
        (1 / 2 - 1 / is) + 1)
      ∧ (Q + 1) * (loopPool (L + Q + 4) Q A Ae (1 / iδ) (1 / ic) (1 / (2 * iξ)) (1 / ip)
        (1 / 2 - 1 / is) + 1)
        ≤ 2 ^ 16 * (Q * (log ((L + Q) * Ae * iδ * R * iξ * is) * ip * is ^ 2
          + log (Q * iδ) * ip)) := by
  obtain ⟨h0, h⟩ := claim_suff_le (iξ := 2 * iξ) hL hQ hA hAAe hiδ hic (by linarith) hip his
  refine ⟨h0, h.trans ?_⟩
  obtain ⟨hl0, hl⟩ := rel_sq_le (P := L + Q) (iδ := iδ) (ip := 1) (is := is) hQ (by linarith)
    (hA.trans hAAe)
    (by linarith) hR hiξ le_rfl (by linarith) hic hicR
  simp only [mul_one] at hl0 hl
  have hQδ : 0 ≤ log (Q * iδ) := Real.log_nonneg (by nlinarith)
  have hps : 0 ≤ ip * is ^ 2 := by have : 0 ≤ ip := by linarith
                                   positivity
  have e1 : log ((L + Q) * Ae * iδ * ic * (2 * iξ) * is) * ip * is ^ 2
      ≤ 5 * log ((L + Q) * Ae * iδ * R * iξ * is) * ip * is ^ 2 := by
    have := mul_le_mul_of_nonneg_right hl hps
    linarith
  have hY0 : 0 ≤ log ((L + Q) * Ae * iδ * R * iξ * is) * ip * is ^ 2 := by
    have := mul_nonneg (by linarith : 0 ≤ log ((L + Q) * Ae * iδ * R * iξ * is)) hps
    linarith
  have hc0 : 0 ≤ log (Q * iδ) * ip := by positivity
  have hQ0 : 0 ≤ Q := by linarith
  have := mul_le_mul_of_nonneg_left e1 hQ0
  nlinarith

lemma rel_probes_le {L Q A Ae iδ R iξ ic : ℝ} (hL : 0 ≤ L) (hQ : 1 ≤ Q)
    (hA : 2 ≤ A) (hAAe : A ≤ Ae) (hiδ : 2 ≤ iδ) (hic : 1 ≤ ic) (hicR : ic ≤ 4 * R * Q * iξ) :
    0 ≤ (Q + 1) * loopBudget (L + Q + 4) Q A Ae (1 / iδ) (1 / ic)
      ∧ (Q + 1) * loopBudget (L + Q + 4) Q A Ae (1 / iδ) (1 / ic)
        ≤ 2 ^ 21 * (Q ^ 4 * A * Ae * (L + Q) * R * iξ * log (Q * A * iδ)) := by
  obtain ⟨h0, h⟩ := claim_probes_le hL hQ hA hAAe hiδ hic
  refine ⟨h0, h.trans ?_⟩
  have hQA : 1 ≤ Q * A := by nlinarith
  have hl : 0 ≤ log (Q * A * iδ) :=
    Real.log_nonneg (one_le_mul_of_one_le_of_one_le hQA (by linarith))
  have hK : 0 ≤ Q ^ 3 * A * Ae * (L + Q) * log (Q * A * iδ) := by
    have : 0 ≤ L + Q := by linarith
    have : 0 ≤ Ae := by linarith
    have : 0 ≤ A := by linarith
    positivity
  have := mul_le_mul_of_nonneg_left hicR hK
  nlinarith

lemma visitC_bounds {R : ℝ} (L : ℕ) (hR : 0 ≤ R) :
    R ≤ R * (L + 1) / ((L + 1 - (L + 1) / 2 : ℕ) : ℝ)
      ∧ R * (L + 1) / ((L + 1 - (L + 1) / 2 : ℕ) : ℝ) ≤ 2 * R := by
  have h1 : 1 ≤ L + 1 - (L + 1) / 2 := by omega
  have h2 : L + 1 - (L + 1) / 2 ≤ L + 1 := by omega
  have h3 : L + 1 ≤ 2 * (L + 1 - (L + 1) / 2) := by omega
  have r1 : (1 : ℝ) ≤ ((L + 1 - (L + 1) / 2 : ℕ) : ℝ) := by exact_mod_cast h1
  have r2 : ((L + 1 - (L + 1) / 2 : ℕ) : ℝ) ≤ L + 1 := by exact_mod_cast h2
  have r3 : (L : ℝ) + 1 ≤ 2 * ((L + 1 - (L + 1) / 2 : ℕ) : ℝ) := by exact_mod_cast h3
  have hm : (0 : ℝ) < ((L + 1 - (L + 1) / 2 : ℕ) : ℝ) := by linarith
  constructor
  · rw [le_div_iff₀ hm]; nlinarith
  · rw [div_le_iff₀ hm]; nlinarith

set_option maxHeartbeats 8000000 in -- `loopMassInv` bounds about twenty nested quantities
/-- `loopMassInv` in the inverted coordinates, `iξ ≤ ic` and `A ≤ Ae`. -/
lemma claim_mass_le {Q A Ae iδ ic iξ ip is : ℝ} {nQ nA : ℕ} (hQ : 1 ≤ Q) (hA : 2 ≤ A)
    (hAAe : A ≤ Ae) (hiδ : 2 ≤ iδ) (hic : 1 ≤ ic) (hiξ : 1 ≤ iξ) (hξc : iξ ≤ ic) (hip : 1 ≤ ip)
    (his : 2 ≤ is) (hnQ : (nQ : ℝ) ≤ Q) (hnA : (nA : ℝ) ≤ A) :
    loopMassInv Q A Ae nQ nA (1 / iδ) (1 / ic) (1 / iξ) (1 / ip) (1 / 2 - 1 / is)
      ≤ 2 ^ 251 * (Ae ^ 4 * ic ^ 4 * Q ^ 9 * iδ * ip ^ 2 * is ^ 11
        * log (Q * Ae * iδ * ic * ip * is) ^ 2) := by
  unfold loopMassInv
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
  have hR₀ : R₀ = 1901000 * (Q + 2) * Θ ^ 2
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
  have hξX : iξ ≤ X := hξc.trans hcX
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
  set Θ₀ := Ae * ic * Q ^ 2 * is ^ 2 with hΘ₀d
  have hΘ₀1 : 1 ≤ Θ₀ := by
    rw [hΘ₀d]
    exact one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le
      (one_le_mul_of_one_le_of_one_le hAe1 hic) (one_le_pow₀ hQ)) (one_le_pow₀ hs1)
  have hΘ : Θ ≤ 2 ^ 19 * Θ₀ := by
    have h1 : (Ae * ic + iξ) * is ^ 2 ≤ 2 * Θ₀ := by
      have : Ae * ic + iξ ≤ 2 * (Ae * ic * Q ^ 2) := by
        have : ic ≤ Ae * ic := le_mul_of_one_le_left (by linarith) hAe1
        have : Ae * ic ≤ Ae * ic * Q ^ 2 :=
          le_mul_of_one_le_right (by positivity) (one_le_pow₀ hQ)
        linarith
      calc (Ae * ic + iξ) * is ^ 2 ≤ 2 * (Ae * ic * Q ^ 2) * is ^ 2 :=
            mul_le_mul_of_nonneg_right this (by positivity)
        _ = 2 * Θ₀ := by rw [hΘ₀d]; ring
    have h2 : Q ^ 2 * Ae * ic ≤ Θ₀ := by
      calc Q ^ 2 * Ae * ic = Q ^ 2 * Ae * ic * 1 := by ring
        _ ≤ Q ^ 2 * Ae * ic * is ^ 2 :=
            mul_le_mul_of_nonneg_left (one_le_pow₀ hs1) (by positivity)
        _ = Θ₀ := by rw [hΘ₀d]; ring
    calc Θ ≤ 2 ^ 17 * ((Ae * ic + iξ) * is ^ 2 + Q ^ 2 * Ae * ic) := hΘu
      _ ≤ 2 ^ 17 * (2 * Θ₀ + Θ₀) := by gcongr
      _ ≤ 2 ^ 19 * Θ₀ := by linarith
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
  have hR₀e : R₀ = 1901000 * (Q + 2) * Θ ^ 2 * (log Z + 2) * is ^ 4 := by
    rw [hR₀, hsis]; field_simp
  have hR₀0 : 0 ≤ R₀ := by rw [hR₀e]; positivity
  have hR₀ : R₀ ≤ (2 ^ 67 : ℝ) * (Ae ^ 2 * ic ^ 2 * Q ^ 5 * is ^ 8 * lX) := by
    rw [hR₀e]
    have hq : Q + 2 ≤ 3 * Q := by linarith
    have hΘ2 : Θ ^ 2 ≤ (2 ^ 19 * Θ₀) ^ 2 := pow_le_pow_left₀ (by linarith) hΘ 2
    calc 1901000 * (Q + 2) * Θ ^ 2 * (log Z + 2) * is ^ 4
        ≤ 1901000 * (3 * Q) * (2 ^ 19 * Θ₀) ^ 2 * (50 * lX) * is ^ 4 := by gcongr
      _ = 1901000 * 3 * 2 ^ 38 * 50 * (Ae ^ 2 * ic ^ 2 * Q ^ 5 * is ^ 8 * lX) := by rw [hΘ₀d]; ring
      _ ≤ 2 ^ 67 * (Ae ^ 2 * ic ^ 2 * Q ^ 5 * is ^ 8 * lX) := by
          have : 0 ≤ Ae ^ 2 * ic ^ 2 * Q ^ 5 * is ^ 8 * lX := by positivity
          exact mul_le_mul_of_nonneg_right (by norm_num) this
  set Rs := Ae ^ 2 * ic ^ 2 * Q ^ 5 * is ^ 8 * lX with hRsd
  have hRs1 : 1 ≤ Rs := by
    rw [hRsd]
    have : 1 ≤ Ae ^ 2 * ic ^ 2 * Q ^ 5 * is ^ 8 :=
      one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le
        (one_le_mul_of_one_le_of_one_le (one_le_pow₀ hAe1) (one_le_pow₀ hic)) (one_le_pow₀ hQ))
        (one_le_pow₀ hs1)
    exact one_le_mul_of_one_le_of_one_le this (by linarith)
  have hR₀1 : R₀ + 1 ≤ 2 * (2 ^ 67 : ℝ) * Rs := by
    have : 1 ≤ (2 ^ 67 : ℝ) * Rs := one_le_mul_of_one_le_of_one_le (by norm_num) hRs1
    linarith
  have hR₀X : R₀ + 1 ≤ X ^ 82 := by
    have hRsX : Rs ≤ X ^ 18 := by
      rw [hRsd]
      have hlXX : lX ≤ X := by rw [hlXd]; exact Real.log_le_self (by linarith)
      calc Ae ^ 2 * ic ^ 2 * Q ^ 5 * is ^ 8 * lX ≤ X ^ 2 * X ^ 2 * X ^ 5 * X ^ 8 * X := by
            gcongr
        _ = X ^ 18 := by ring
    have hK : 2 * (2 ^ 67 : ℝ) ≤ X ^ 64 := by
      have : 2 * (2 ^ 67 : ℝ) ≤ 2 ^ (3 * 64) := by norm_num
      exact this.trans (h2pow 64)
    calc R₀ + 1 ≤ 2 * (2 ^ 67 : ℝ) * Rs := hR₀1
      _ ≤ X ^ 64 * X ^ 18 := mul_le_mul hK hRsX (by linarith) (hXp 64)
      _ = X ^ 82 := by ring
  have hlb : 13 + 2 * Real.logb 2 (R₀ + 1) ≤ 250 * lX := by
    have hlR := hlXp 82 (by linarith) hR₀X
    push_cast at hlR
    have : Real.logb 2 (R₀ + 1) ≤ 121 * lX := by
      rw [Real.logb, div_le_iff₀ (Real.log_pos (by norm_num))]
      have := mul_le_mul_of_nonneg_left hl2.le (by linarith : (0 : ℝ) ≤ lX)
      linarith
    linarith
  have hlb0 : 0 ≤ Real.logb 2 (R₀ + 1) := Real.logb_nonneg (by norm_num) (by linarith)
  -- `E₁`
  set M₀ := Ae ^ 4 * ic ^ 4 * Q ^ 9 * iδ * ip ^ 2 * is ^ 11 with hM₀d
  have hM₀1 : 1 ≤ M₀ := by
    rw [hM₀d]
    exact one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le
      (one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le
        (one_le_mul_of_one_le_of_one_le (one_le_pow₀ hAe1) (one_le_pow₀ hic)) (one_le_pow₀ hQ))
        hδ1) (one_le_pow₀ hp1)) (one_le_pow₀ hs1)
  have hE₁e : E₁ = (Q + 3) ^ 2 * (2 ^ 20 * Θ * (R₀ + 1) * is
      + 2 ^ 36 * Θ ^ 2 * (13 + 2 * Real.logb 2 (R₀ + 1)) * D + 2 ^ 16 * (P₀ + 2) * ip * D) := by
    rw [hE₁, hsd, hδ'D, show 2 ^ 16 * (P₀ + 2) / (1 / ip * δ') = 2 ^ 16 * (P₀ + 2) * ip / δ' by
      field_simp, hδ'D]
  -- `aε E₁` and the other terms, each against `M₀` (and `lX` or `lX²`)
  have hAeic : Ae * ic * (Ae ^ 3 * ic ^ 3 * Q ^ 9 * iδ * ip ^ 2 * is ^ 11) = M₀ := by
    rw [hM₀d]; ring
  have hE₁ : E₁ ≤ 2 ^ 230 * (Ae ^ 3 * ic ^ 3 * Q ^ 9 * iδ * ip ^ 2 * is ^ 11 * lX) := by
    rw [hE₁e]
    have hq : (Q + 3) ^ 2 ≤ 16 * Q ^ 2 := by
      have : (Q + 3) ^ 2 ≤ (4 * Q) ^ 2 := pow_le_pow_left₀ (by linarith) (by linarith) 2
      linarith [show (4 * Q) ^ 2 = 16 * Q ^ 2 by ring]
    have ha : 2 ^ 20 * Θ * (R₀ + 1) * is
        ≤ 2 ^ 20 * (2 ^ 19 * Θ₀) * (2 * (2 ^ 67 : ℝ) * Rs) * is := by
      gcongr
    have hb : 2 ^ 36 * Θ ^ 2 * (13 + 2 * Real.logb 2 (R₀ + 1)) * D
        ≤ 2 ^ 36 * (2 ^ 19 * Θ₀) ^ 2 * (250 * lX) * (16 * (Q * iδ)) := by
      have hΘ2 : Θ ^ 2 ≤ (2 ^ 19 * Θ₀) ^ 2 := pow_le_pow_left₀ (by linarith) hΘ 2
      gcongr
    have hc : 2 ^ 16 * (P₀ + 2) * ip * D
        ≤ 2 ^ 16 * (3602 * lX * ip * is ^ 2) * ip * (16 * (Q * iδ)) := by
      have : P₀ + 2 ≤ 3602 * lX * ip * is ^ 2 := by
        have : 1 ≤ lX * ip * is ^ 2 :=
          one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le (by linarith) hp1)
            (one_le_pow₀ hs1)
        linarith
      gcongr
    -- each of the three against `Ae³ ic³ Q⁷ iδ ip² is¹¹ lX`
    set T₀ := Ae ^ 3 * ic ^ 3 * Q ^ 7 * iδ * ip ^ 2 * is ^ 11 * lX with hT₀d
    have hT₀0 : 0 ≤ T₀ := by rw [hT₀d]; positivity
    have ta : 2 ^ 20 * (2 ^ 19 * Θ₀) * (2 * (2 ^ 67 : ℝ) * Rs) * is
        ≤ 2 ^ 39 * (2 * (2 ^ 67 : ℝ)) * T₀ := by
      have e : 2 ^ 20 * (2 ^ 19 * Θ₀) * (2 * (2 ^ 67 : ℝ) * Rs) * is
          = 2 ^ 39 * (2 * (2 ^ 67 : ℝ)) * (Ae ^ 3 * ic ^ 3 * Q ^ 7 * is ^ 11 * lX) := by
        rw [hΘ₀d, hRsd]; ring
      rw [e, hT₀d]
      have : Ae ^ 3 * ic ^ 3 * Q ^ 7 * is ^ 11 * lX
          ≤ Ae ^ 3 * ic ^ 3 * Q ^ 7 * iδ * ip ^ 2 * is ^ 11 * lX := by
        have h1 : 1 ≤ iδ * ip ^ 2 := one_le_mul_of_one_le_of_one_le hδ1 (one_le_pow₀ hp1)
        have h0 : 0 ≤ Ae ^ 3 * ic ^ 3 * Q ^ 7 * is ^ 11 * lX := by positivity
        calc Ae ^ 3 * ic ^ 3 * Q ^ 7 * is ^ 11 * lX
            = Ae ^ 3 * ic ^ 3 * Q ^ 7 * is ^ 11 * lX * 1 := by ring
          _ ≤ Ae ^ 3 * ic ^ 3 * Q ^ 7 * is ^ 11 * lX * (iδ * ip ^ 2) :=
              mul_le_mul_of_nonneg_left h1 h0
          _ = _ := by ring
      exact mul_le_mul_of_nonneg_left this (by norm_num)
    have tb : 2 ^ 36 * (2 ^ 19 * Θ₀) ^ 2 * (250 * lX) * (16 * (Q * iδ)) ≤ 2 ^ 82 * 250 * T₀ := by
      have e : 2 ^ 36 * (2 ^ 19 * Θ₀) ^ 2 * (250 * lX) * (16 * (Q * iδ))
          = 2 ^ 78 * 250 * (Ae ^ 2 * ic ^ 2 * Q ^ 5 * iδ * is ^ 4 * lX) := by
        rw [hΘ₀d]; ring
      rw [e, hT₀d]
      have : Ae ^ 2 * ic ^ 2 * Q ^ 5 * iδ * is ^ 4 * lX
          ≤ Ae ^ 3 * ic ^ 3 * Q ^ 7 * iδ * ip ^ 2 * is ^ 11 * lX := by
        have h1 : 1 ≤ Ae * ic * Q ^ 2 * ip ^ 2 * is ^ 7 :=
          one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le
            (one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le hAe1 hic)
              (one_le_pow₀ hQ)) (one_le_pow₀ hp1)) (one_le_pow₀ hs1)
        have h0 : 0 ≤ Ae ^ 2 * ic ^ 2 * Q ^ 5 * iδ * is ^ 4 * lX := by positivity
        have := mul_le_mul_of_nonneg_left h1 h0
        calc _ ≤ Ae ^ 2 * ic ^ 2 * Q ^ 5 * iδ * is ^ 4 * lX * (Ae * ic * Q ^ 2 * ip ^ 2 * is ^ 7) :=
              by linarith
          _ = _ := by ring
      have h16 : (2 : ℝ) ^ 78 * 250 ≤ 2 ^ 82 * 250 := by norm_num
      have h0 : 0 ≤ Ae ^ 2 * ic ^ 2 * Q ^ 5 * iδ * is ^ 4 * lX := by positivity
      exact mul_le_mul h16 this h0 (by norm_num)
    have tc : 2 ^ 16 * (3602 * lX * ip * is ^ 2) * ip * (16 * (Q * iδ))
        ≤ 2 ^ 20 * 3602 * T₀ := by
      have e : 2 ^ 16 * (3602 * lX * ip * is ^ 2) * ip * (16 * (Q * iδ))
          = 2 ^ 20 * 3602 * (Q * iδ * ip ^ 2 * is ^ 2 * lX) := by ring
      rw [e, hT₀d]
      have : Q * iδ * ip ^ 2 * is ^ 2 * lX
          ≤ Ae ^ 3 * ic ^ 3 * Q ^ 7 * iδ * ip ^ 2 * is ^ 11 * lX := by
        have h1 : 1 ≤ Ae ^ 3 * ic ^ 3 * Q ^ 6 * is ^ 9 :=
          one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le
            (one_le_mul_of_one_le_of_one_le (one_le_pow₀ hAe1) (one_le_pow₀ hic)) (one_le_pow₀ hQ))
            (one_le_pow₀ hs1)
        have h0 : 0 ≤ Q * iδ * ip ^ 2 * is ^ 2 * lX := by positivity
        have := mul_le_mul_of_nonneg_left h1 h0
        calc _ ≤ Q * iδ * ip ^ 2 * is ^ 2 * lX * (Ae ^ 3 * ic ^ 3 * Q ^ 6 * is ^ 9) := by linarith
          _ = _ := by ring
      gcongr
    have hsum : 2 ^ 20 * Θ * (R₀ + 1) * is + 2 ^ 36 * Θ ^ 2 * (13 + 2 * Real.logb 2 (R₀ + 1)) * D
        + 2 ^ 16 * (P₀ + 2) * ip * D ≤ 2 ^ 225 * T₀ := by
      have hK : 2 ^ 39 * (2 * (2 ^ 67 : ℝ)) + 2 ^ 82 * 250 + 2 ^ 20 * 3602 ≤ (2 : ℝ) ^ 225 := by
        norm_num
      have := mul_le_mul_of_nonneg_right hK hT₀0
      linarith
    have hsum0 : 0 ≤ 2 ^ 20 * Θ * (R₀ + 1) * is
        + 2 ^ 36 * Θ ^ 2 * (13 + 2 * Real.logb 2 (R₀ + 1)) * D + 2 ^ 16 * (P₀ + 2) * ip * D := by
      positivity
    calc (Q + 3) ^ 2 * (2 ^ 20 * Θ * (R₀ + 1) * is
          + 2 ^ 36 * Θ ^ 2 * (13 + 2 * Real.logb 2 (R₀ + 1)) * D + 2 ^ 16 * (P₀ + 2) * ip * D)
        ≤ (16 * Q ^ 2) * (2 ^ 225 * T₀) := mul_le_mul hq hsum hsum0 (by positivity)
      _ = 2 ^ 229 * (Ae ^ 3 * ic ^ 3 * Q ^ 9 * iδ * ip ^ 2 * is ^ 11 * lX) := by
          rw [hT₀d]; ring
      _ ≤ _ := by
          have : 0 ≤ Ae ^ 3 * ic ^ 3 * Q ^ 9 * iδ * ip ^ 2 * is ^ 11 * lX := by positivity
          linarith
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
  have hEb : E ≤ 2 ^ 245 * (M₀ * lX) := by
    rw [hE]
    have h1 : 1536 * aε * E₁ ≤ 1536 * (6 * (Ae * ic))
        * (2 ^ 230 * (Ae ^ 3 * ic ^ 3 * Q ^ 9 * iδ * ip ^ 2 * is ^ 11 * lX)) := by gcongr
    have h2 : 1536 * log (16 * (Q + 1) * Kc / δ') * aε
        ≤ 1536 * (18 * (Q * A * lX)) * (6 * (Ae * ic)) := by gcongr
    have e1 : 1536 * (6 * (Ae * ic)) * (2 ^ 230 * (Ae ^ 3 * ic ^ 3 * Q ^ 9 * iδ * ip ^ 2 * is ^ 11
        * lX)) = 1536 * 6 * 2 ^ 230 * (M₀ * lX) := by rw [← hAeic]; ring
    have e2 : Q * A * lX * (Ae * ic) ≤ M₀ * lX := by
      have : Q * A * (Ae * ic) ≤ M₀ := by
        rw [hM₀d]
        have h1 : Q * A * (Ae * ic) ≤ Q * Ae * (Ae * ic) := by gcongr
        have h2 : 1 ≤ Ae ^ 2 * ic ^ 3 * Q ^ 8 * iδ * ip ^ 2 * is ^ 11 :=
          one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le
            (one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le
              (one_le_mul_of_one_le_of_one_le (one_le_pow₀ hAe1) (one_le_pow₀ hic))
              (one_le_pow₀ hQ)) hδ1) (one_le_pow₀ hp1)) (one_le_pow₀ hs1)
        have h0 : 0 ≤ Q * Ae * (Ae * ic) := by positivity
        have := mul_le_mul_of_nonneg_left h2 h0
        calc Q * A * (Ae * ic) ≤ Q * Ae * (Ae * ic) * (Ae ^ 2 * ic ^ 3 * Q ^ 8 * iδ * ip ^ 2
              * is ^ 11) := by linarith
          _ = Ae ^ 4 * ic ^ 4 * Q ^ 9 * iδ * ip ^ 2 * is ^ 11 := by ring
      have hlX0 : 0 ≤ lX := by linarith
      calc Q * A * lX * (Ae * ic) = Q * A * (Ae * ic) * lX := by ring
        _ ≤ M₀ * lX := mul_le_mul_of_nonneg_right this hlX0
    have hM0 : 0 ≤ M₀ * lX := by positivity
    linarith
  -- `E₂` and `E₃`
  have hn0 : (0 : ℝ) ≤ n := Nat.cast_nonneg _
  have hnb : (n : ℝ) ≤ 2 ^ 9 * (lX * ic ^ 2) := by
    have h := claim_heldOut_le hQ hiδ hiξ
    rw [← hn] at h
    have h2 : iξ ^ 2 ≤ ic ^ 2 := pow_le_pow_left₀ (by linarith) hξc 2
    have hlQ0 : 0 ≤ log (Q * iδ) := Real.log_nonneg (one_le_mul_of_one_le_of_one_le hQ hδ1)
    have : log (Q * iδ) * iξ ^ 2 ≤ lX * ic ^ 2 := mul_le_mul hlQδ h2 (by positivity) (by linarith)
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
  have hE₂0 : 0 ≤ E₂ := by rw [hE₂]; positivity
  have hE₂b : E₂ ≤ 2 ^ 250 * (M₀ * lX ^ 2) := by
    rw [hE₂, show ∀ x : ℝ, x / (1 / iδ) = x * iδ from fun x => by field_simp]
    set U := Ae ^ 2 * ic ^ 2 * Q ^ 6 * is ^ 8 * lX with hUd
    have hU1 : 1 ≤ U := by
      rw [hUd]
      have : 1 ≤ Ae ^ 2 * ic ^ 2 * Q ^ 6 * is ^ 8 :=
        one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le
          (one_le_mul_of_one_le_of_one_le (one_le_pow₀ hAe1) (one_le_pow₀ hic)) (one_le_pow₀ hQ))
          (one_le_pow₀ hs1)
      exact one_le_mul_of_one_le_of_one_le this (by linarith)
    have hin : 6 * (Q + 2) * R₀ + K₁ + n ≤ 2 ^ 75 * U := by
      have h1 : 6 * (Q + 2) * R₀ ≤ 18 * (2 ^ 67 : ℝ) * U := by
        have : 6 * (Q + 2) ≤ 18 * Q := by linarith
        calc 6 * (Q + 2) * R₀
            ≤ (18 * Q) * ((2 ^ 67 : ℝ) * (Ae ^ 2 * ic ^ 2 * Q ^ 5 * is ^ 8 * lX)) :=
              mul_le_mul this hR₀ hR₀0 (by positivity)
          _ = 18 * (2 ^ 67 : ℝ) * U := by rw [hUd]; ring
      have h2 : Q ^ 2 * A * Ae * ic * lX ≤ U := by
        rw [hUd]
        have h1 : Q ^ 2 * A * Ae * ic * lX ≤ Q ^ 2 * Ae * Ae * ic * lX := by gcongr
        have h2 : 1 ≤ ic * Q ^ 4 * is ^ 8 :=
          one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le hic (one_le_pow₀ hQ))
            (one_le_pow₀ hs1)
        have h0 : 0 ≤ Q ^ 2 * Ae * Ae * ic * lX := by positivity
        have := mul_le_mul_of_nonneg_left h2 h0
        calc _ ≤ Q ^ 2 * Ae * Ae * ic * lX * (ic * Q ^ 4 * is ^ 8) := by linarith
          _ = _ := by ring
      have h3 : lX * ic ^ 2 ≤ U := by
        rw [hUd]
        have h2 : 1 ≤ Ae ^ 2 * Q ^ 6 * is ^ 8 :=
          one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le (one_le_pow₀ hAe1)
            (one_le_pow₀ hQ)) (one_le_pow₀ hs1)
        have h0 : 0 ≤ lX * ic ^ 2 := by positivity
        have := mul_le_mul_of_nonneg_left h2 h0
        calc _ ≤ lX * ic ^ 2 * (Ae ^ 2 * Q ^ 6 * is ^ 8) := by linarith
          _ = _ := by ring
      have hK : 18 * (2 ^ 67 : ℝ) + 2 ^ 16 + 2 ^ 9 ≤ (2 : ℝ) ^ 75 := by norm_num
      have := mul_le_mul_of_nonneg_right hK (by linarith : (0 : ℝ) ≤ U)
      linarith
    have hin0 : 0 ≤ 6 * (Q + 2) * R₀ + K₁ + n := by positivity
    have e : Q * (lX * ic ^ 2) * U * iδ ≤ M₀ * lX ^ 2 := by
      rw [hUd, hM₀d]
      have h1 : 1 ≤ Ae ^ 2 * ic ^ 0 * Q ^ 2 * ip ^ 2 * is ^ 3 := by
        simp only [pow_zero, mul_one]
        exact one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le
          (one_le_mul_of_one_le_of_one_le (one_le_pow₀ hAe1) (one_le_pow₀ hQ)) (one_le_pow₀ hp1))
          (one_le_pow₀ hs1)
      have h0 : 0 ≤ Q * (lX * ic ^ 2) * (Ae ^ 2 * ic ^ 2 * Q ^ 6 * is ^ 8 * lX) * iδ := by
        positivity
      have := mul_le_mul_of_nonneg_left h1 h0
      calc _ ≤ Q * (lX * ic ^ 2) * (Ae ^ 2 * ic ^ 2 * Q ^ 6 * is ^ 8 * lX) * iδ
            * (Ae ^ 2 * ic ^ 0 * Q ^ 2 * ip ^ 2 * is ^ 3) := by linarith
        _ = _ := by ring
    calc 4 * (Q + 1) * n * (6 * (Q + 2) * R₀ + K₁ + n) * iδ
        ≤ 4 * (2 * Q) * (2 ^ 9 * (lX * ic ^ 2)) * (2 ^ 75 * U) * iδ := by
          gcongr; linarith
      _ = 2 ^ 87 * (Q * (lX * ic ^ 2) * U * iδ) := by ring
      _ ≤ 2 ^ 87 * (M₀ * lX ^ 2) := by gcongr
      _ ≤ _ := by
          have : 0 ≤ M₀ * lX ^ 2 := by positivity
          linarith
  have hE₃b : E₃ ≤ 2 ^ 40 * (M₀ * lX ^ 2) := by
    rw [hE₃, hδ'D]
    have h1 : K₁ ^ 2 ≤ (2 ^ 16 * (Q ^ 2 * A * Ae * ic * lX)) ^ 2 := pow_le_pow_left₀ hK₁0 hK₁b 2
    have e : (Q ^ 2 * A * Ae * ic * lX) ^ 2 * (Q * iδ) ≤ M₀ * lX ^ 2 := by
      rw [hM₀d]
      have hA2 : A ^ 2 ≤ Ae ^ 2 := pow_le_pow_left₀ (by linarith) hAAe 2
      have h1 : (Q ^ 2 * A * Ae * ic * lX) ^ 2 * (Q * iδ)
          ≤ Q ^ 5 * Ae ^ 4 * ic ^ 2 * iδ * lX ^ 2 := by
        have e : (Q ^ 2 * A * Ae * ic * lX) ^ 2 * (Q * iδ)
            = A ^ 2 * (Q ^ 5 * Ae ^ 2 * ic ^ 2 * iδ * lX ^ 2) := by ring
        rw [e]
        calc A ^ 2 * (Q ^ 5 * Ae ^ 2 * ic ^ 2 * iδ * lX ^ 2)
            ≤ Ae ^ 2 * (Q ^ 5 * Ae ^ 2 * ic ^ 2 * iδ * lX ^ 2) := by gcongr
          _ = _ := by ring
      have h2 : 1 ≤ ic ^ 2 * Q ^ 4 * ip ^ 2 * is ^ 11 :=
        one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le
          (one_le_mul_of_one_le_of_one_le (one_le_pow₀ hic) (one_le_pow₀ hQ)) (one_le_pow₀ hp1))
          (one_le_pow₀ hs1)
      have h0 : 0 ≤ Q ^ 5 * Ae ^ 4 * ic ^ 2 * iδ * lX ^ 2 := by positivity
      have := mul_le_mul_of_nonneg_left h2 h0
      calc _ ≤ Q ^ 5 * Ae ^ 4 * ic ^ 2 * iδ * lX ^ 2 := h1
        _ ≤ Q ^ 5 * Ae ^ 4 * ic ^ 2 * iδ * lX ^ 2 * (ic ^ 2 * Q ^ 4 * ip ^ 2 * is ^ 11) := by
            linarith
        _ = _ := by ring
    calc 16 * K₁ ^ 2 * D ≤ 16 * (2 ^ 16 * (Q ^ 2 * A * Ae * ic * lX)) ^ 2 * (16 * (Q * iδ)) := by
          gcongr
      _ = 2 ^ 40 * ((Q ^ 2 * A * Ae * ic * lX) ^ 2 * (Q * iδ)) := by ring
      _ ≤ _ := by gcongr
  -- the sum
  have hlX1 : lX ≤ lX ^ 2 := by nlinarith only [hlX2]
  have hM0 : 0 ≤ M₀ := by linarith
  have hEb' : E ≤ 2 ^ 245 * (M₀ * lX ^ 2) :=
    hEb.trans (mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left hlX1 hM0) (by norm_num))
  calc E + E₂ + E₃ ≤ 2 ^ 245 * (M₀ * lX ^ 2) + 2 ^ 250 * (M₀ * lX ^ 2) + 2 ^ 40 * (M₀ * lX ^ 2) :=
        add_le_add (add_le_add hEb' hE₂b) hE₃b
    _ = (2 ^ 245 + 2 ^ 250 + 2 ^ 40) * (M₀ * lX ^ 2) := by ring
    _ ≤ 2 ^ 251 * (M₀ * lX ^ 2) := by
        have hK : (2 : ℝ) ^ 245 + 2 ^ 250 + 2 ^ 40 ≤ 2 ^ 251 := by norm_num
        exact mul_le_mul_of_nonneg_right hK (by positivity)
    _ = _ := by rw [hM₀d, hlXd]

lemma half_inv (ξ : ℝ) : 1 / (2 * (1 / ξ)) = ξ / 2 := by
  simp only [one_div, mul_inv, inv_inv]; ring

/-- `claim_mass_le` at `1/c ≤ 4R|Q|/ξ` and `ξ/2`. -/
lemma rel_mass_le {Q A Ae iδ R iξ ip is ic : ℝ} {nQ nA : ℕ} (hQ : 1 ≤ Q) (hA : 2 ≤ A)
    (hAAe : A ≤ Ae) (hiδ : 2 ≤ iδ) (hR : 1 ≤ R) (hiξ : 1 ≤ iξ) (hip : 1 ≤ ip) (his : 2 ≤ is)
    (hic : 2 * iξ ≤ ic) (hicR : ic ≤ 4 * R * Q * iξ) (hnQ : (nQ : ℝ) ≤ Q) (hnA : (nA : ℝ) ≤ A) :
    loopMassInv Q A Ae nQ nA (1 / iδ) (1 / ic) (1 / (2 * iξ)) (1 / ip) (1 / 2 - 1 / is)
      ≤ 2 ^ 251 * 2304 * (Q ^ 13 * R ^ 4 * log (Q * R * Ae * iδ * iξ * ip * is) ^ 2
        * Ae ^ 4 * iδ * iξ ^ 4 * ip ^ 2 * is ^ 11) := by
  have hAe1 : 1 ≤ Ae := by linarith
  have hδ1 : 1 ≤ iδ := by linarith
  have hs1 : 1 ≤ is := by linarith
  have hic1 : 1 ≤ ic := by linarith
  have h := claim_mass_le hQ hA hAAe hiδ hic1 (by linarith) hic hip his hnQ hnA
  refine h.trans ?_
  set Y := Q * R * Ae * iδ * iξ * ip * is with hYd
  have hY8 : 8 ≤ Y := by
    have h1 : 1 ≤ Q * R := one_le_mul_of_one_le_of_one_le hQ hR
    have h2 : 2 ≤ Q * R * Ae := by nlinarith only [h1, hA, hAAe]
    have h3 : 4 ≤ Q * R * Ae * iδ := by nlinarith only [h2, hiδ]
    have h4 : 4 ≤ Q * R * Ae * iδ * iξ := by nlinarith only [h3, hiξ]
    have h5 : 4 ≤ Q * R * Ae * iδ * iξ * ip := by nlinarith only [h4, hip]
    rw [hYd]; nlinarith only [h5, his]
  have hX1 : 1 ≤ Q * Ae * iδ * ic * ip * is :=
    one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le
      (one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le
        (one_le_mul_of_one_le_of_one_le hQ hAe1) hδ1) hic1) hip) hs1
  have hXY : Q * Ae * iδ * ic * ip * is ≤ Y ^ 3 := by
    have h1 : Q * Ae * iδ * ic * ip * is ≤ 4 * Q * Y := by
      calc Q * Ae * iδ * ic * ip * is ≤ Q * Ae * iδ * (4 * R * Q * iξ) * ip * is := by
            gcongr
        _ = 4 * Q * Y := by rw [hYd]; ring
    have hQY : Q ≤ Y := by
      have : 1 ≤ R * Ae * iδ * iξ * ip * is :=
        one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le
          (one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le
            (one_le_mul_of_one_le_of_one_le hR hAe1) hδ1) hiξ) hip) hs1
      calc Q = Q * 1 := by ring
        _ ≤ Q * (R * Ae * iδ * iξ * ip * is) := mul_le_mul_of_nonneg_left this (by linarith)
        _ = Y := by rw [hYd]; ring
    have h4 : 4 * Q * Y ≤ Y ^ 3 := by
      have : 4 * Q ≤ Y ^ 2 := by nlinarith only [hQY, hY8]
      calc 4 * Q * Y ≤ Y ^ 2 * Y := mul_le_mul_of_nonneg_right this (by linarith)
        _ = Y ^ 3 := by ring
    linarith
  have hl : log (Q * Ae * iδ * ic * ip * is) ≤ 3 * log Y := by
    have := log_le_mul_log_of_le_pow 3 (by linarith) hXY
    push_cast at this; linarith
  have hl0 : 0 ≤ log (Q * Ae * iδ * ic * ip * is) := Real.log_nonneg hX1
  have hl2 : log (Q * Ae * iδ * ic * ip * is) ^ 2 ≤ 9 * log Y ^ 2 := by
    have := pow_le_pow_left₀ hl0 hl 2
    linarith [show (3 * log Y) ^ 2 = 9 * log Y ^ 2 by ring]
  have hc4 : ic ^ 4 ≤ 256 * (R ^ 4 * Q ^ 4 * iξ ^ 4) := by
    have := pow_le_pow_left₀ (by linarith) hicR 4
    linarith [show (4 * R * Q * iξ) ^ 4 = 256 * (R ^ 4 * Q ^ 4 * iξ ^ 4) by ring]
  have hK0 : 0 ≤ Ae ^ 4 * Q ^ 9 * iδ * ip ^ 2 * is ^ 11 := by positivity
  calc 2 ^ 251 * (Ae ^ 4 * ic ^ 4 * Q ^ 9 * iδ * ip ^ 2 * is ^ 11
        * log (Q * Ae * iδ * ic * ip * is) ^ 2)
      = 2 ^ 251 * ((Ae ^ 4 * Q ^ 9 * iδ * ip ^ 2 * is ^ 11) * ic ^ 4
        * log (Q * Ae * iδ * ic * ip * is) ^ 2) := by ring
    _ ≤ 2 ^ 251 * ((Ae ^ 4 * Q ^ 9 * iδ * ip ^ 2 * is ^ 11) * (256 * (R ^ 4 * Q ^ 4 * iξ ^ 4))
        * (9 * log Y ^ 2)) := by gcongr
    _ = 2 ^ 251 * 2304 * (Q ^ 13 * R ^ 4 * log Y ^ 2
        * Ae ^ 4 * iδ * iξ ^ 4 * ip ^ 2 * is ^ 11) := by ring

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

set_option maxHeartbeats 400000 in -- `nlinarith` over the log bounds
/-- `L₀` at a constant `C₀ ≥ 1` against `1 + log (|Q| R/(δ ξ pAP s₀))/log |Σ|`, in the inverted
coordinates. -/
lemma loopL0_le {Q A R iδ iξ ip is C₀ : ℝ} (hQ : 1 ≤ Q) (hA : 2 ≤ A) (hR : 1 ≤ R) (hiδ : 2 ≤ iδ)
    (hiξ : 1 ≤ iξ) (hip : 1 ≤ ip) (his : 2 ≤ is) (hC₀ : 1 ≤ C₀) :
    1 ≤ Q ^ 13 * R ^ 4 * log (Q * R * A * iδ * iξ * ip * is) ^ 2 * A ^ 4 * iδ * iξ ^ 4 * ip ^ 2
        * is ^ 11
      ∧ 1 + 4 * (log (C₀ * (Q ^ 13 * R ^ 4 * log (Q * R * A * iδ * iξ * ip * is) ^ 2 * A ^ 4
          * iδ * iξ ^ 4 * ip ^ 2 * is ^ 11)) + 6 * log (Q + 4) + 24) / log A
        ≤ (241 + 6 * log C₀) * (1 + log (Q * R * iδ * iξ * ip * is) / log A) := by
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
  set G := Q ^ 13 * R ^ 4 * log Y ^ 2 * A ^ 4 * iδ * iξ ^ 4 * ip ^ 2 * is ^ 11 with hGd
  have hG1 : 1 ≤ G := by
    rw [hGd]
    have : 1 ≤ log Y ^ 2 := one_le_pow₀ hlY1
    exact one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le
      (one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le
        (one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le
          (one_le_mul_of_one_le_of_one_le (one_le_pow₀ hQ) (one_le_pow₀ hR)) this)
          (one_le_pow₀ hA1)) hδ1) (one_le_pow₀ hiξ)) (one_le_pow₀ hip)) (one_le_pow₀ hs1)
  refine ⟨hG1, ?_⟩
  -- `G ≤ A⁶ Z³⁷`
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
  have hlY2 : log Y ^ 2 ≤ Y ^ 2 := pow_le_pow_left₀ (by linarith) hlYY 2
  have hGZ : G ≤ A ^ 6 * Z ^ 37 := by
    rw [hGd]
    calc Q ^ 13 * R ^ 4 * log Y ^ 2 * A ^ 4 * iδ * iξ ^ 4 * ip ^ 2 * is ^ 11
        ≤ Z ^ 13 * Z ^ 4 * Y ^ 2 * A ^ 4 * Z * Z ^ 4 * Z ^ 2 * Z ^ 11 := by
          gcongr
      _ = A ^ 6 * Z ^ 37 := by rw [hYZ]; ring
  have hlA : log 2 ≤ log A := log_two_le_log hA
  have hlA0 : 0 < log A := by linarith
  have hlZ0 : 0 ≤ log Z := Real.log_nonneg hZ1
  have hlC : 0 ≤ log C₀ := Real.log_nonneg hC₀
  have hlG : log G ≤ 6 * log A + 37 * log Z := by
    have := Real.log_le_log (by linarith) hGZ
    have e : log (A ^ 6 * Z ^ 37) = 6 * log A + 37 * log Z := by
      rw [Real.log_mul (by positivity) (by positivity), Real.log_pow, Real.log_pow]
      push_cast; ring
    linarith
  have hlCG : log (C₀ * G) ≤ log C₀ + 6 * log A + 37 * log Z := by
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
      ≤ 24 * log A + (4 * log C₀ + 144) + 172 * log Z := by linarith
  have hdiv : 4 * (log (C₀ * G) + 6 * log (Q + 4) + 24) / log A
      ≤ 24 + (4 * log C₀ + 144) / log A + 172 * (log Z / log A) := by
    rw [div_le_iff₀ hlA0]
    have e : (24 + (4 * log C₀ + 144) / log A + 172 * (log Z / log A)) * log A
        = 24 * log A + (4 * log C₀ + 144) + 172 * log Z := by field_simp
    rw [e]; exact hnum
  have hc : (4 * log C₀ + 144) / log A ≤ 6 * log C₀ + 216 := by
    rw [div_le_iff₀ hlA0]
    nlinarith [mul_le_mul_of_nonneg_left (show (2 : ℝ) / 3 ≤ log A by linarith)
      (show 0 ≤ 6 * log C₀ + 216 by linarith)]
  have hq : 0 ≤ log Z / log A := div_nonneg hlZ0 hlA0.le
  nlinarith [mul_nonneg hlC hq]

open Asymptotics Filter

/-- `LoopClaimUniform`'s prefix shape against `LoopClaimSimple`'s. -/
lemma loop_pref_dom {r : LoopRun} (hr : r ∈ loopRunRegion) :
    0 ≤ r.Q ^ 5 * (r.L + r.Q) ^ 2 * r.A ^ 2 * r.R ^ 2
        * log ((r.L + r.Q) * r.A * r.R / (r.δ * r.ξ * r.pAP * r.s₀)) / r.ξ ^ 2
        * (1 / r.s₀ ^ 8 + r.Q ^ 4 / r.s₀ ^ 4)
      ∧ 0 ≤ r.Q ^ 9 * (r.L + r.Q) ^ 2 * r.A ^ 2 * r.R ^ 2
        * log ((r.L + r.Q) * r.A * r.R / (r.δ * r.ξ * r.pAP * r.s₀)) / (r.ξ ^ 2 * r.s₀ ^ 8)
      ∧ r.Q ^ 5 * (r.L + r.Q) ^ 2 * r.A ^ 2 * r.R ^ 2
        * log ((r.L + r.Q) * r.A * r.R / (r.δ * r.ξ * r.pAP * r.s₀)) / r.ξ ^ 2
        * (1 / r.s₀ ^ 8 + r.Q ^ 4 / r.s₀ ^ 4)
        ≤ 2 * (r.Q ^ 9 * (r.L + r.Q) ^ 2 * r.A ^ 2 * r.R ^ 2
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
  have hK : 0 ≤ r.Q ^ 5 * (r.L + r.Q) ^ 2 * r.A ^ 2 * r.R ^ 2 * lg / r.ξ ^ 2 := by positivity
  have hs4 : r.s₀ ^ 4 ≤ 1 := pow_le_one₀ hs.le (by linarith)
  have hQ4 : (1 : ℝ) ≤ r.Q ^ 4 := one_le_pow₀ hQ'
  have hb : 1 / r.s₀ ^ 8 + r.Q ^ 4 / r.s₀ ^ 4 ≤ 2 * (r.Q ^ 4 / r.s₀ ^ 8) := by
    have h8 : 0 < r.s₀ ^ 8 := by positivity
    have h48 : r.s₀ ^ 8 ≤ r.s₀ ^ 4 := by
      have : r.s₀ ^ 8 = r.s₀ ^ 4 * r.s₀ ^ 4 := by ring
      rw [this]; exact mul_le_of_le_one_right (by positivity) hs4
    have e1 : 1 / r.s₀ ^ 8 ≤ r.Q ^ 4 / r.s₀ ^ 8 := div_le_div_of_nonneg_right hQ4 h8.le
    have e2 : r.Q ^ 4 / r.s₀ ^ 4 ≤ r.Q ^ 4 / r.s₀ ^ 8 :=
      div_le_div_of_nonneg_left (by positivity) h8 h48
    linarith
  have e : r.Q ^ 9 * (r.L + r.Q) ^ 2 * r.A ^ 2 * r.R ^ 2 * lg / (r.ξ ^ 2 * r.s₀ ^ 8)
      = r.Q ^ 5 * (r.L + r.Q) ^ 2 * r.A ^ 2 * r.R ^ 2 * lg / r.ξ ^ 2 * (r.Q ^ 4 / r.s₀ ^ 8) := by
    field_simp
  refine ⟨mul_nonneg hK (by positivity), by rw [e]; positivity, ?_⟩
  rw [e]
  calc r.Q ^ 5 * (r.L + r.Q) ^ 2 * r.A ^ 2 * r.R ^ 2 * lg / r.ξ ^ 2
        * (1 / r.s₀ ^ 8 + r.Q ^ 4 / r.s₀ ^ 4)
      ≤ r.Q ^ 5 * (r.L + r.Q) ^ 2 * r.A ^ 2 * r.R ^ 2 * lg / r.ξ ^ 2
        * (2 * (r.Q ^ 4 / r.s₀ ^ 8)) := mul_le_mul_of_nonneg_left hb hK
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

/-- The coverage the general claim runs at, `ξ/(2C|Q|)`, `C = R(L+1)/(L+1−⌈L/2⌉)`. -/
def distC (r : LoopDistRun) : ℝ :=
  r.ξ / (2 * (r.R * (r.L + 1) / ((r.L + 1 - (r.L + 1) / 2 : ℕ) : ℝ)) * r.Q)

/-- The most `D` may hold on any first `k` letters. -/
def distP₀Tol (r : LoopDistRun) : ℝ :=
  1 / ((r.L + r.Q + 4) ^ 6 * max 1
    (loopMassInv r.Q r.A (1 / r.β) r.Q r.A r.δ (distC r) (r.ξ / 2) r.pAP (1 / 2 - r.s₀)))

def distPref (r : LoopDistRun) : ℝ :=
  (r.Q + 1) * (6 * (r.Q + 2)
    * loopCount (r.L + r.Q + 4) r.Q r.A (1 / r.β) r.δ (distC r) (r.ξ / 2) r.pAP (1 / 2 - r.s₀))

def distSuff (r : LoopDistRun) : ℝ :=
  (r.Q + 1)
    * (loopPool (r.L + r.Q + 4) r.Q r.A (1 / r.β) r.δ (distC r) (r.ξ / 2) r.pAP (1 / 2 - r.s₀) + 1)

def distProbes (r : LoopDistRun) : ℝ :=
  (r.Q + 1) * loopBudget (r.L + r.Q + 4) r.Q r.A (1 / r.β) r.δ (distC r)

def distHeldOut (x : LoopDistInputs) : ℕ :=
  ⌈32 * log (16 * ((x.Q : ℝ) + 1) * ((x.Q : ℝ) + 2) / x.δ) / (x.ξ / 2) ^ 2⌉₊

/-- The inverted coordinates of a run in `loopDistRunRegion`. -/
lemma dist_coords {r : LoopDistRun} (hr : r ∈ loopDistRunRegion) :
    0 ≤ (r.L : ℝ) ∧ 1 ≤ (r.Q : ℝ) ∧ 2 ≤ (r.A : ℝ) ∧ (r.A : ℝ) ≤ 1 / r.β ∧ 2 ≤ 1 / r.δ
      ∧ 1 ≤ r.R ∧ 1 ≤ 1 / r.ξ ∧ 1 ≤ 1 / r.pAP ∧ 2 ≤ 1 / r.s₀
      ∧ 2 * (1 / r.ξ) ≤ 1 / distC r ∧ 1 / distC r ≤ 4 * r.R * r.Q * (1 / r.ξ)
      ∧ 0 < distC r ∧ distC r ≤ 1 := by
  obtain ⟨⟨hQ, hA, hδ, hδ2, hR, hξ, hξ1, hp, hp1, hs, hs1⟩, hβ, hβA⟩ := hr
  have hQ' : (1 : ℝ) ≤ r.Q := by exact_mod_cast hQ
  have hA' : (2 : ℝ) ≤ r.A := by exact_mod_cast hA
  obtain ⟨hC1, hC2⟩ := visitC_bounds r.L (by linarith : (0 : ℝ) ≤ r.R)
  set C := r.R * (r.L + 1) / ((r.L + 1 - (r.L + 1) / 2 : ℕ) : ℝ) with hCd
  have hc : distC r = r.ξ / (2 * C * r.Q) := rfl
  have hC0 : 0 < C := by linarith
  have hc0 : 0 < distC r := by rw [hc]; positivity
  have hic : 1 / distC r = 2 * C * r.Q * (1 / r.ξ) := by rw [hc]; field_simp
  refine ⟨Nat.cast_nonneg _, hQ', hA', ?_, (le_div_iff₀ hδ).2 (by linarith), hR,
    (le_div_iff₀ hξ).2 (by linarith), (le_div_iff₀ hp).2 (by linarith),
    (le_div_iff₀ hs).2 (by linarith), ?_, ?_, hc0, ?_⟩
  · rw [le_div_iff₀ hβ]; linarith [mul_comm r.β (r.A : ℝ)]
  · rw [hic]
    have : 2 ≤ 2 * C * r.Q := by nlinarith
    exact mul_le_mul_of_nonneg_right this (by positivity)
  · rw [hic]
    have : 2 * C * r.Q ≤ 4 * r.R * r.Q := by nlinarith
    exact mul_le_mul_of_nonneg_right this (by positivity)
  · rw [hc, div_le_one (by positivity)]
    nlinarith

lemma bigO_dist {f g : LoopDistRun → ℝ} (K : ℝ) (hK : 0 < K)
    (h : ∀ r ∈ loopDistRunRegion, 0 ≤ f r ∧ f r ≤ K * g r) :
    f =O[𝓟 loopDistRunRegion] g :=
  isBigO_on K fun r hr => by
    obtain ⟨h0, h1⟩ := h r hr
    exact ⟨h0, nonneg_of_mul_nonneg_right (h0.trans h1) hK, h1⟩

lemma distPref_isBigO : distPref =O[𝓟 loopDistRunRegion] (fun r =>
      r.Q ^ 5 * (r.L + r.Q) ^ 2 * r.R ^ 2
        * log ((r.L + r.Q) * r.R / (r.β * r.δ * r.ξ * r.pAP * r.s₀)) / (r.β ^ 2 * r.ξ ^ 2)
        * (1 / r.s₀ ^ 8 + r.Q ^ 4 / r.s₀ ^ 4)) := by
  refine bigO_dist (2 ^ 82) (by norm_num) fun r hr => ?_
  obtain ⟨hL, hQ, hA, hAAe, hiδ, hR, hiξ, hip, his, hic, hicR, -, -⟩ := dist_coords hr
  have h := rel_pref_le hL hQ hA hAAe hiδ hR hiξ hip his (by linarith) hicR
  simp only [one_div_one_div, half_inv] at h
  exact ⟨h.1, h.2.trans_eq (by ring_nf)⟩

lemma distSuff_isBigO : distSuff =O[𝓟 loopDistRunRegion] (fun r =>
      r.Q * (log ((r.L + r.Q) * r.R / (r.β * r.δ * r.ξ * r.s₀)) / (r.pAP * r.s₀ ^ 2)
        + log (r.Q / r.δ) / r.pAP)) := by
  refine bigO_dist (2 ^ 16) (by norm_num) fun r hr => ?_
  obtain ⟨hL, hQ, hA, hAAe, hiδ, hR, hiξ, hip, his, hic, hicR, -, -⟩ := dist_coords hr
  have h := rel_suff_le hL hQ hA hAAe hiδ hR hiξ hip his (by linarith) hicR
  simp only [one_div_one_div, half_inv] at h
  exact ⟨h.1, h.2.trans_eq (by ring_nf)⟩

lemma distProbes_isBigO : distProbes =O[𝓟 loopDistRunRegion] (fun r =>
      r.Q ^ 4 * r.A * (r.L + r.Q) * r.R / (r.β * r.ξ) * log (r.Q * r.A / r.δ)) := by
  refine bigO_dist (2 ^ 21) (by norm_num) fun r hr => ?_
  obtain ⟨hL, hQ, hA, hAAe, hiδ, hR, hiξ, hip, his, hic, hicR, -, -⟩ := dist_coords hr
  have h := rel_probes_le hL hQ hA hAAe hiδ (by linarith) hicR
  simp only [one_div_one_div] at h
  exact ⟨h.1, h.2.trans_eq (by ring_nf)⟩

lemma distP₀Tol_pos (r : LoopDistRun) : 0 < distP₀Tol r := by
  unfold distP₀Tol
  have : (0 : ℝ) < (r.L : ℝ) + r.Q + 4 := by positivity
  exact one_div_pos.2 (mul_pos (by positivity) (lt_of_lt_of_le one_pos (le_max_left _ _)))

lemma distP₀Tol_isBigO : (fun r => 1 / distP₀Tol r) =O[𝓟 loopDistRunRegion] (fun r =>
      (r.L + r.Q) ^ 6 * r.Q ^ 13 * r.R ^ 4
        * log (r.Q * r.R / (r.β * r.δ * r.ξ * r.pAP * r.s₀)) ^ 2
        / (r.β ^ 4 * r.δ * r.ξ ^ 4 * r.pAP ^ 2 * r.s₀ ^ 11)) := by
  refine bigO_dist (15625 * (2 ^ 251 * 2304)) (by norm_num) fun r hr => ?_
  obtain ⟨hL, hQ, hA, hAAe, hiδ, hR, hiξ, hip, his, hic, hicR, hc0, hc1⟩ := dist_coords hr
  obtain ⟨⟨-, -, hδ, hδ2, -, hξ, hξ1, hp, hp1, hs, hs1⟩, hβ, -⟩ := hr
  have hM := rel_mass_le (nQ := r.Q) (nA := r.A) hQ hA hAAe hiδ hR hiξ hip his hic hicR le_rfl
    le_rfl
  simp only [one_div_one_div, half_inv] at hM
  have hM1 : 1 ≤ loopMassInv r.Q r.A (1 / r.β) r.Q r.A r.δ (distC r) (r.ξ / 2) r.pAP
      (1 / 2 - r.s₀) :=
    loopMassInv_ge_one _ _ (by linarith) (by linarith) (hA.trans hAAe) hδ (by linarith) hc0 hc1
      (by positivity) hp hp1 (by linarith) (by linarith)
  have hinv : 1 / distP₀Tol r = ((r.L : ℝ) + r.Q + 4) ^ 6 * loopMassInv r.Q r.A (1 / r.β) r.Q r.A
      r.δ (distC r) (r.ξ / 2) r.pAP (1 / 2 - r.s₀) := by
    unfold distP₀Tol
    rw [one_div_one_div, max_eq_right hM1]
  have hℓ : ((r.L : ℝ) + r.Q + 4) ^ 6 ≤ 15625 * ((r.L : ℝ) + r.Q) ^ 6 := by
    have : (r.L : ℝ) + r.Q + 4 ≤ 5 * ((r.L : ℝ) + r.Q) := by linarith
    have := pow_le_pow_left₀ (by positivity) this 6
    linarith [show (5 * ((r.L : ℝ) + r.Q)) ^ 6 = 15625 * ((r.L : ℝ) + r.Q) ^ 6 by ring]
  rw [hinv]
  refine ⟨by positivity, ?_⟩
  calc ((r.L : ℝ) + r.Q + 4) ^ 6 * loopMassInv r.Q r.A (1 / r.β) r.Q r.A r.δ (distC r) (r.ξ / 2)
        r.pAP (1 / 2 - r.s₀)
      ≤ (15625 * ((r.L : ℝ) + r.Q) ^ 6) * (2 ^ 251 * 2304 * ((r.Q : ℝ) ^ 13 * r.R ^ 4
        * log (r.Q * r.R * (1 / r.β) * (1 / r.δ) * (1 / r.ξ) * (1 / r.pAP) * (1 / r.s₀)) ^ 2
        * (1 / r.β) ^ 4 * (1 / r.δ) * (1 / r.ξ) ^ 4 * (1 / r.pAP) ^ 2 * (1 / r.s₀) ^ 11)) :=
        mul_le_mul hℓ hM (by linarith) (by positivity)
    _ = _ := by ring_nf

lemma distHeldOut_isBigO : (fun x => (distHeldOut x : ℝ)) =O[𝓟 loopDistRegion]
    (fun x => log (x.Q / x.δ) / x.ξ ^ 2) := by
  refine isBigO_on (2 ^ 11) fun x hx => ?_
  obtain ⟨⟨hQ, -, hδ, hδ2, -, hξ, hξ1, -, -, -, -⟩, -, -⟩ := hx
  have hQ' : (1 : ℝ) ≤ x.Q := by exact_mod_cast hQ
  have hiδ : 2 ≤ 1 / x.δ := (le_div_iff₀ hδ).2 (by linarith)
  have hiξ : 1 ≤ 1 / x.ξ := (le_div_iff₀ hξ).2 (by linarith)
  have h := claim_heldOut_le hQ' hiδ (by linarith : 1 ≤ 2 * (1 / x.ξ))
  rw [one_div_one_div, half_inv] at h
  have hl : 0 ≤ log (x.Q / x.δ) := by
    rw [← mul_one_div]; exact Real.log_nonneg (by nlinarith)
  refine ⟨Nat.cast_nonneg _, div_nonneg hl (sq_nonneg _), h.trans_eq ?_⟩
  field_simp

end OrthoDFA

end
