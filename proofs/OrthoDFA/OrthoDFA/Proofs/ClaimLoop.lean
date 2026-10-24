import OrthoDFA.Claims.Loop
import OrthoDFA.Proofs.LoopDFA
import OrthoDFA.Proofs.BigO

/-!
# The learner's costs, asymptotically: the proof

`learner_loop_dfa_counts` bounds, for every draw, the loop's reads by `(|Q| + 1)` rounds of
`6 (|Q| + 2) loopCount` prefixes and `loopPool + 1` suffixes, and its probes by
`(|Q| + 1) loopBudget`, all at `ℓ = L + |Q| + 4`. In the inverted coordinates `loopΘ` is a
polynomial, so `loopCount` is `ℓ² Θ²/s⁴` times the log of a polynomial and `loopPool` a sum of
logs of polynomials; `L₀` is the log of a polynomial over `log |Σ|`.
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory Asymptotics Filter Real


lemma log_le_of_le_two_pow_mul {y X : ℝ} (m k : ℕ) (hy : 0 < y) (hX : 2 ≤ X)
    (h : y ≤ 2 ^ m * X ^ k) :
    log y ≤ (m + k) * log X := by
  have hX0 : 0 < X := by linarith
  calc log y ≤ log (2 ^ m * X ^ k) := log_le_log hy h
    _ = m * log 2 + k * log X := by
        rw [log_mul (by positivity) (by positivity), log_pow, log_pow]
    _ ≤ m * log X + k * log X := by gcongr
    _ = (m + k) * log X := by ring

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


lemma loopPool_inv (ℓ Q A iδ ic iξ ip is : ℝ) :
    loopPool ℓ Q A (1 / iδ) (1 / ic) (1 / iξ) (1 / ip) (1 / 2 - 1 / is)
      = (50 * (max (log (20 * loopΘ Q A (1 / ic) (1 / iξ) (1 / 2 - 1 / is)))
          (log (8 * loopW₀ Q A (1 / iδ) * (8 * (Q + 1) * iδ))) + 2 * log ℓ) * is ^ 2
          + 20 * is + 28) * ip
        + 64 * log (16 * (Q + 4) * (Q + 1) * iδ) * ip ^ 2 := by
  unfold loopPool
  dsimp only
  rw [sub_sub_cancel, show 8 * loopW₀ Q A (1 / iδ) / (1 / iδ / (8 * (Q + 1)))
    = 8 * loopW₀ Q A (1 / iδ) * (8 * (Q + 1) * iδ) by field_simp,
    show (Q + 4) / (1 / iδ / (8 * (Q + 1)) / 2) = 16 * (Q + 4) * (Q + 1) * iδ by field_simp; ring]
  field_simp

lemma loopCount_inv (ℓ Q A iδ ic iξ ip is : ℝ) :
    loopCount ℓ Q A (1 / iδ) (1 / ic) (1 / iξ) (1 / ip) (1 / 2 - 1 / is)
      = 161000 * (Q + 2) ^ 2 * loopΘ Q A (1 / ic) (1 / iξ) (1 / 2 - 1 / is) ^ 2 * ℓ ^ 2
        * log (16 * (Q + 4) * loopPool ℓ Q A (1 / iδ) (1 / ic) (1 / iξ) (1 / ip) (1 / 2 - 1 / is)
          * loopΘ Q A (1 / ic) (1 / iξ) (1 / 2 - 1 / is) * ℓ * (8 * (Q + 1) * iδ) ^ 2 * ip)
        * is ^ 4 := by
  unfold loopCount
  dsimp only
  rw [sub_sub_cancel]
  set P := loopPool ℓ Q A (1 / iδ) (1 / ic) (1 / iξ) (1 / ip) (1 / 2 - 1 / is)
  set Θ := loopΘ Q A (1 / ic) (1 / iξ) (1 / 2 - 1 / is)
  rw [show 16 * (Q + 4) * P * Θ * ℓ / ((1 / iδ / (8 * (Q + 1))) ^ 2 * (1 / ip))
    = 16 * (Q + 4) * P * Θ * ℓ * (8 * (Q + 1) * iδ) ^ 2 * ip by field_simp]
  field_simp

lemma loopBudget_inv (ℓ Q A iδ ic : ℝ) :
    loopBudget ℓ Q A (1 / iδ) (1 / ic)
      = (36 * (Q * ((Q + 2) * A + 1) + 2 * A + 1) * (4 * (A + 1) * ic)
        * log (16 * (Q * ((Q + 2) * A + 1) + 2 * A + 1) * (Q * ((Q + 2) * A + 1) + 2 * A + 1 + 1)
          * (8 * (Q + 1) * iδ))
        + (Q * ((Q + 2) * A + 1) + 2 * A + 1) + 1) * ℓ := by
  unfold loopBudget
  dsimp only
  rw [show 4 * (A + 1) / (1 / ic) = 4 * (A + 1) * ic by field_simp]
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

lemma loopΘ_le_pow {Q A ic iξ is X : ℝ} (hQ : 1 ≤ Q) (hA : 1 ≤ A) (hic : 1 ≤ ic) (hiξ : 1 ≤ iξ)
    (his : 1 ≤ is) (hQX : Q ≤ X) (hAX : A ≤ X) (hcX : ic ≤ X) (hξX : iξ ≤ X) (hsX : is ≤ X) :
    loopΘ Q A (1 / ic) (1 / iξ) (1 / 2 - 1 / is) ≤ 2 ^ 19 * X ^ 4 := by
  have hX : 1 ≤ X := hQ.trans hQX
  refine (loopΘ_le hQ hA hic hiξ his).2.trans ?_
  have h : (A * ic + iξ) * is ^ 2 + Q ^ 2 * A * ic ≤ (X * X + X) * X ^ 2 + X ^ 2 * X * X := by
    gcongr
  have h3 : X ^ 3 ≤ X ^ 4 := pow_le_pow_right₀ hX (by norm_num)
  nlinarith

lemma loopW₀_le {Q A iδ X : ℝ} (hQ : 1 ≤ Q) (hA : 1 ≤ A) (hiδ : 1 ≤ iδ) (hQX : Q ≤ X) (hAX : A ≤ X)
    (hδX : iδ ≤ X) :
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


lemma loopPool_ge {ℓ Q A iδ ic iξ ip is : ℝ} (hℓ : 1 ≤ ℓ) (hQ : 1 ≤ Q) (hA : 1 ≤ A) (hiδ : 1 ≤ iδ)
    (hic : 1 ≤ ic) (hiξ : 1 ≤ iξ) (hip : 1 ≤ ip) (his : 1 ≤ is) :
    1 ≤ loopPool ℓ Q A (1 / iδ) (1 / ic) (1 / iξ) (1 / ip) (1 / 2 - 1 / is) := by
  rw [loopPool_inv]
  have hΘ := (loopΘ_le hQ hA hic hiξ his).1
  have h1 : 0 ≤ log (20 * loopΘ Q A (1 / ic) (1 / iξ) (1 / 2 - 1 / is)) :=
    Real.log_nonneg (by linarith)
  have hm : 0 ≤ max (log (20 * loopΘ Q A (1 / ic) (1 / iξ) (1 / 2 - 1 / is)))
      (log (8 * loopW₀ Q A (1 / iδ) * (8 * (Q + 1) * iδ))) := le_max_of_le_left h1
  have h2 : 0 ≤ log ℓ := Real.log_nonneg hℓ
  have h3 : 0 ≤ log (16 * (Q + 4) * (Q + 1) * iδ) :=
    Real.log_nonneg (one_le_mul_of_one_le_of_one_le (by nlinarith) hiδ)
  have h4 : 0 ≤ 64 * log (16 * (Q + 4) * (Q + 1) * iδ) * ip ^ 2 := by positivity
  have h5 : 0 ≤ (50 * (max (log (20 * loopΘ Q A (1 / ic) (1 / iξ) (1 / 2 - 1 / is)))
      (log (8 * loopW₀ Q A (1 / iδ) * (8 * (Q + 1) * iδ))) + 2 * log ℓ) * is ^ 2 + 20 * is) :=
    by positivity
  nlinarith

lemma loopPool_le_pow {ℓ Q A iδ ic iξ ip is X : ℝ} (hℓ : 1 ≤ ℓ) (hQ : 1 ≤ Q) (hA : 1 ≤ A)
    (hiδ : 1 ≤ iδ) (hic : 1 ≤ ic) (hiξ : 1 ≤ iξ) (hip : 1 ≤ ip) (his : 1 ≤ is) (hℓX : ℓ ≤ X)
    (hQX : Q ≤ X) (hAX : A ≤ X) (hδX : iδ ≤ X) (hcX : ic ≤ X) (hξX : iξ ≤ X) (hpX : ip ≤ X)
    (hsX : is ≤ X) :
    loopPool ℓ Q A (1 / iδ) (1 / ic) (1 / iξ) (1 / ip) (1 / 2 - 1 / is) ≤ 2 ^ 34 * X ^ 16 := by
  have hX : 1 ≤ X := hQ.trans hQX
  rw [loopPool_inv]
  set Θ := loopΘ Q A (1 / ic) (1 / iξ) (1 / 2 - 1 / is)
  set W₀ := loopW₀ Q A (1 / iδ)
  have hΘ1 := (loopΘ_le hQ hA hic hiξ his).1
  have hΘX := loopΘ_le_pow hQ hA hic hiξ his hQX hAX hcX hξX hsX
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
  have hl2 : log (16 * (Q + 4) * (Q + 1) * iδ) ≤ 16 * (5 * X) * (2 * X) * X := by
    refine (Real.log_le_self (by positivity)).trans ?_
    gcongr <;> linarith
  have hlp : 0 ≤ log (16 * (Q + 4) * (Q + 1) * iδ) :=
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
  have hb2 : 64 * log (16 * (Q + 4) * (Q + 1) * iδ) * ip ^ 2 ≤ 64 * (16 * (5 * X) * (2 * X) * X)
      * X ^ 2 := by gcongr
  have h5 : X ^ 5 ≤ X ^ 16 := pow_le_pow_right₀ hX (by norm_num)
  have hb0 : 0 ≤ (50 * (max (log (20 * Θ)) (log (8 * W₀ * (8 * (Q + 1) * iδ))) + 2 * log ℓ)
      * is ^ 2 + 20 * is + 28) := by positivity
  have h6 := mul_le_mul hb hpX (by linarith) (by positivity)
  have e : X ^ 15 * X = X ^ 16 := by ring
  nlinarith


lemma loopPool_le_log {ℓ Q A iδ ic iξ ip is X : ℝ} (hℓ : 1 ≤ ℓ) (hQ : 2 ≤ Q) (hA : 1 ≤ A)
    (hiδ : 1 ≤ iδ) (hic : 1 ≤ ic) (hiξ : 1 ≤ iξ) (hip : 1 ≤ ip) (his : 1 ≤ is) (hX : 2 ≤ X)
    (hℓX : ℓ ≤ 4 * X) (hQX : Q ≤ X) (hAX : A ≤ X) (hδX : iδ ≤ X) (hcX : ic ≤ X) (hξX : iξ ≤ X)
    (hsX : is ≤ X) :
    loopPool ℓ Q A (1 / iδ) (1 / ic) (1 / iξ) (1 / ip) (1 / 2 - 1 / is)
      ≤ 2400 * log X * ip * is ^ 2 + 576 * log (Q * iδ) * ip ^ 2 := by
  rw [loopPool_inv]
  set Θ := loopΘ Q A (1 / ic) (1 / iξ) (1 / 2 - 1 / is)
  set W₀ := loopW₀ Q A (1 / iδ)
  have hQ1 : 1 ≤ Q := by linarith
  have hΘ1 := (loopΘ_le hQ1 hA hic hiξ his).1
  have hΘX := loopΘ_le_pow hQ1 hA hic hiξ his hQX hAX hcX hξX hsX
  obtain ⟨hW1, hWX⟩ := loopW₀_le hQ1 hA hiδ hQX hAX hδX
  have hlX : log 2 ≤ log X := log_two_le_log hX
  have hlX0 : 0 ≤ log X := Real.log_nonneg (by linarith)
  have h1 : log (20 * Θ) ≤ 39 * log X := by
    have := log_le_of_le_two_pow_mul (y := 20 * Θ) 24 4 (by positivity) hX (by nlinarith)
    push_cast at this
    nlinarith
  have h2 : log (8 * W₀ * (8 * (Q + 1) * iδ)) ≤ 39 * log X := by
    have hb : 8 * W₀ * (8 * (Q + 1) * iδ) ≤ 8 * (2 ^ 19 * X ^ 11) * (8 * (X + X) * X) := by
      gcongr <;> linarith
    have := log_le_of_le_two_pow_mul (y := 8 * W₀ * (8 * (Q + 1) * iδ)) 26 13 (by positivity) hX
      (by nlinarith)
    push_cast at this
    linarith
  have h3 : log ℓ ≤ 3 * log X := by
    have := log_le_of_le_two_pow_mul (y := ℓ) 2 1 (by positivity) hX (by linarith)
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
  have hc : log (16 * (Q + 4) * (Q + 1) * iδ) ≤ 9 * log (Q * iδ) := by
    have hb : 16 * (Q + 4) * (Q + 1) * iδ ≤ 2 ^ 7 * (Q * iδ) ^ 2 := by
      have e1 : 16 * (Q + 4) * (Q + 1) ≤ 128 * Q ^ 2 := by nlinarith
      have e2 : 16 * (Q + 4) * (Q + 1) * iδ ≤ 128 * Q ^ 2 * iδ :=
        mul_le_mul_of_nonneg_right e1 (by linarith)
      have e3 : 128 * Q ^ 2 * iδ ≤ 128 * Q ^ 2 * iδ ^ 2 :=
        mul_le_mul_of_nonneg_left (by nlinarith) (by positivity)
      nlinarith
    have := log_le_of_le_two_pow_mul 7 2 (by positivity) hQδ hb
    push_cast at this
    linarith
  have hip0 : 0 ≤ ip := by linarith
  have := mul_le_mul_of_nonneg_right hb hip0
  have := mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left hc (by norm_num : (0 : ℝ) ≤ 64))
    (sq_nonneg ip)
  nlinarith


lemma le_of_eq_mul₁ {X c r : ℝ} (hc : 0 ≤ c) (hr : 1 ≤ r) (h : X = c * r) : c ≤ X := by
  rw [h]; exact le_mul_of_one_le_right hc hr

lemma one_le_mul₂ {a b : ℝ} (ha : 1 ≤ a) (hb : 1 ≤ b) : 1 ≤ a * b :=
  one_le_mul_of_one_le_of_one_le ha hb

set_option maxHeartbeats 1000000 in -- the bounds chain through many `gcongr` steps
lemma claim_pref_le {L Q A iδ ic iξ ip is : ℝ} (hL : 2 ≤ L) (hQ : 2 ≤ Q) (hA : 2 ≤ A)
    (hiδ : 2 ≤ iδ) (hic : 2 ≤ ic) (hiξ : 2 ≤ iξ) (hip : 2 ≤ ip) (his : 2 ≤ is) :
    0 ≤ (Q + 1) * (6 * (Q + 2)
        * loopCount (L + Q + 4) Q A (1 / iδ) (1 / ic) (1 / iξ) (1 / ip) (1 / 2 - 1 / is))
      ∧ (Q + 1) * (6 * (Q + 2)
        * loopCount (L + Q + 4) Q A (1 / iδ) (1 / ic) (1 / iξ) (1 / ip) (1 / 2 - 1 / is))
        ≤ 2 ^ 70 * (Q ^ 4 * (L + Q) ^ 2 * log ((L + Q) * A * iδ * ic * iξ * ip * is)
          * ((A ^ 2 * ic ^ 2 + iξ ^ 2) * is ^ 8 + Q ^ 4 * A ^ 2 * ic ^ 2 * is ^ 4)) := by
  set X := (L + Q) * A * iδ * ic * iξ * ip * is with hXd
  have hA1 : 1 ≤ A := by linarith
  have hδ1 : 1 ≤ iδ := by linarith
  have hc1 : 1 ≤ ic := by linarith
  have hξ1 : 1 ≤ iξ := by linarith
  have hp1 : 1 ≤ ip := by linarith
  have hs1 : 1 ≤ is := by linarith
  have hQ1 : 1 ≤ Q := by linarith
  have hLQ : 4 ≤ L + Q := by linarith
  have hr : 2 ≤ A * iδ * ic * iξ * ip * is := by
    have := one_le_mul₂ (one_le_mul₂ (one_le_mul₂ (one_le_mul₂ hδ1 hc1) hξ1) hp1) hs1
    nlinarith
  have hLQX : 2 * (L + Q) ≤ X := by
    rw [hXd, show (L + Q) * A * iδ * ic * iξ * ip * is = (L + Q) * (A * iδ * ic * iξ * ip * is)
      by ring]
    nlinarith
  have hX : 2 ≤ X := by linarith
  have hQX : Q ≤ X := by linarith
  have hAX : A ≤ X := le_of_eq_mul₁ (c := A) (r := (L + Q) * iδ * ic * iξ * ip * is) (by linarith)
    (one_le_mul₂ (one_le_mul₂ (one_le_mul₂ (one_le_mul₂ (one_le_mul₂ (by linarith) hδ1) hc1) hξ1)
      hp1) hs1) (by rw [hXd]; ring)
  have hδX : iδ ≤ X := le_of_eq_mul₁ (c := iδ) (r := (L + Q) * A * ic * iξ * ip * is)
    (by linarith) (one_le_mul₂ (one_le_mul₂ (one_le_mul₂ (one_le_mul₂ (one_le_mul₂ (by linarith)
      hA1) hc1) hξ1) hp1) hs1) (by rw [hXd]; ring)
  have hcX : ic ≤ X := le_of_eq_mul₁ (c := ic) (r := (L + Q) * A * iδ * iξ * ip * is)
    (by linarith) (one_le_mul₂ (one_le_mul₂ (one_le_mul₂ (one_le_mul₂ (one_le_mul₂ (by linarith)
      hA1) hδ1) hξ1) hp1) hs1) (by rw [hXd]; ring)
  have hξX : iξ ≤ X := le_of_eq_mul₁ (c := iξ) (r := (L + Q) * A * iδ * ic * ip * is)
    (by linarith) (one_le_mul₂ (one_le_mul₂ (one_le_mul₂ (one_le_mul₂ (one_le_mul₂ (by linarith)
      hA1) hδ1) hc1) hp1) hs1) (by rw [hXd]; ring)
  have hpX : ip ≤ X := le_of_eq_mul₁ (c := ip) (r := (L + Q) * A * iδ * ic * iξ * is)
    (by linarith) (one_le_mul₂ (one_le_mul₂ (one_le_mul₂ (one_le_mul₂ (one_le_mul₂ (by linarith)
      hA1) hδ1) hc1) hξ1) hs1) (by rw [hXd]; ring)
  have hsX : is ≤ X := le_of_eq_mul₁ (c := is) (r := (L + Q) * A * iδ * ic * iξ * ip)
    (by linarith) (one_le_mul₂ (one_le_mul₂ (one_le_mul₂ (one_le_mul₂ (one_le_mul₂ (by linarith)
      hA1) hδ1) hc1) hξ1) hp1) (by rw [hXd]; ring)
  set ℓ := L + Q + 4 with hℓd
  have hℓ1 : 1 ≤ ℓ := by linarith
  have hℓ2 : ℓ ≤ 2 * (L + Q) := by linarith
  have hℓX : ℓ ≤ X := by linarith
  rw [loopCount_inv]
  set Θ := loopΘ Q A (1 / ic) (1 / iξ) (1 / 2 - 1 / is)
  set PB := loopPool ℓ Q A (1 / iδ) (1 / ic) (1 / iξ) (1 / ip) (1 / 2 - 1 / is)
  obtain ⟨hΘ1, hΘu⟩ := loopΘ_le hQ1 hA1 hc1 hξ1 hs1
  have hΘX := loopΘ_le_pow hQ1 hA1 hc1 hξ1 hs1 hQX hAX hcX hξX hsX
  have hPB1 : 1 ≤ PB := loopPool_ge hℓ1 hQ1 hA1 hδ1 hc1 hξ1 hp1 hs1
  have hPBX := loopPool_le_pow hℓ1 hQ1 hA1 hδ1 hc1 hξ1 hp1 hs1 hℓX hQX hAX hδX hcX hξX hpX hsX
  set Y := 16 * (Q + 4) * PB * Θ * ℓ * (8 * (Q + 1) * iδ) ^ 2 * ip with hYd
  have hY1 : 1 ≤ Y := by
    have h8 : 1 ≤ (8 * (Q + 1) * iδ) ^ 2 := one_le_pow₀ (by nlinarith)
    exact one_le_mul₂ (one_le_mul₂ (one_le_mul₂ (one_le_mul₂ (one_le_mul₂ (by linarith) hPB1) hΘ1)
      hℓ1) h8) hp1
  have hYX : Y ≤ 2 ^ 68 * X ^ 27 := by
    have h8 : 8 * (Q + 1) * iδ ≤ 16 * X ^ 2 := by nlinarith
    have : Y ≤ 16 * (5 * X) * (2 ^ 34 * X ^ 16) * (2 ^ 19 * X ^ 4) * X * (16 * X ^ 2) ^ 2 * X := by
      rw [hYd]; gcongr <;> linarith
    have e : 16 * (5 * X) * (2 ^ 34 * X ^ 16) * (2 ^ 19 * X ^ 4) * X * (16 * X ^ 2) ^ 2 * X
        = 5 * 2 ^ 65 * X ^ 27 := by ring
    have : (0 : ℝ) ≤ X ^ 27 := by positivity
    linarith
  have hlY : log Y ≤ 95 * log X := by
    have := log_le_of_le_two_pow_mul (y := Y) 68 27 (by linarith) hX hYX
    push_cast at this
    linarith
  have hlY0 : 0 ≤ log Y := Real.log_nonneg hY1
  have hG : Θ ^ 2 * is ^ 4 ≤ 2 ^ 36 * ((A ^ 2 * ic ^ 2 + iξ ^ 2) * is ^ 8
      + Q ^ 4 * A ^ 2 * ic ^ 2 * is ^ 4) := by
    have h1 : Θ ^ 2 ≤ (2 ^ 17 * ((A * ic + iξ) * is ^ 2 + Q ^ 2 * A * ic)) ^ 2 :=
      pow_le_pow_left₀ (by linarith) hΘu 2
    have h2 : ((A * ic + iξ) * is ^ 2 + Q ^ 2 * A * ic) ^ 2
        ≤ 2 * ((A * ic + iξ) ^ 2 * is ^ 4 + Q ^ 4 * A ^ 2 * ic ^ 2) := by
      linarith [sq_nonneg ((A * ic + iξ) * is ^ 2 - Q ^ 2 * A * ic)]
    have h3 : (A * ic + iξ) ^ 2 ≤ 2 * (A ^ 2 * ic ^ 2 + iξ ^ 2) := by
      linarith [sq_nonneg (A * ic - iξ)]
    have hs4 : 0 ≤ is ^ 4 := by positivity
    have hs8 : is ^ 4 * is ^ 4 = is ^ 8 := by ring
    have h4 := mul_le_mul_of_nonneg_right h1 hs4
    have h5 := mul_le_mul_of_nonneg_right h2 hs4
    have h6 := mul_le_mul_of_nonneg_right h3 (by positivity : (0 : ℝ) ≤ is ^ 4 * is ^ 4)
    have h7 : 0 ≤ Q ^ 4 * A ^ 2 * ic ^ 2 * is ^ 4 := by positivity
    linarith only [h4, h5, h6, h7]
  have hcnt0 : 0 ≤ 161000 * (Q + 2) ^ 2 * Θ ^ 2 * ℓ ^ 2 * log Y * is ^ 4 := by positivity
  constructor
  · positivity
  · have hlX0 : 0 ≤ log X := Real.log_nonneg (by linarith)
    have e : (Q + 1) * (6 * (Q + 2) * (161000 * (Q + 2) ^ 2 * Θ ^ 2 * ℓ ^ 2 * log Y * is ^ 4))
        = 966000 * ((Q + 1) * (Q + 2) ^ 3 * ℓ ^ 2 * log Y) * (Θ ^ 2 * is ^ 4) := by ring
    rw [e]
    have a1 : Q + 1 ≤ 2 * Q := by linarith only [hQ]
    have a2 : Q + 2 ≤ 2 * Q := by linarith only [hQ]
    have h1 : (Q + 1) * (Q + 2) ^ 3 * ℓ ^ 2 * log Y
        ≤ (2 * Q) * (2 * Q) ^ 3 * (2 * (L + Q)) ^ 2 * (95 * log X) := by
      gcongr
    have hG0 : 0 ≤ (A ^ 2 * ic ^ 2 + iξ ^ 2) * is ^ 8 + Q ^ 4 * A ^ 2 * ic ^ 2 * is ^ 4 := by
      positivity
    have hq0 : 0 ≤ (Q + 1) * (Q + 2) ^ 3 * ℓ ^ 2 * log Y := by positivity
    calc 966000 * ((Q + 1) * (Q + 2) ^ 3 * ℓ ^ 2 * log Y) * (Θ ^ 2 * is ^ 4)
        ≤ 966000 * ((2 * Q) * (2 * Q) ^ 3 * (2 * (L + Q)) ^ 2 * (95 * log X))
          * (2 ^ 36 * ((A ^ 2 * ic ^ 2 + iξ ^ 2) * is ^ 8 + Q ^ 4 * A ^ 2 * ic ^ 2 * is ^ 4)) := by
          gcongr
      _ = (966000 * 2 * 8 * 4 * 95 * 2 ^ 36) * (Q ^ 4 * (L + Q) ^ 2 * log X
          * ((A ^ 2 * ic ^ 2 + iξ ^ 2) * is ^ 8 + Q ^ 4 * A ^ 2 * ic ^ 2 * is ^ 4)) := by ring
      _ ≤ _ := by gcongr; norm_num


lemma claim_suff_le {L Q A iδ ic iξ ip is : ℝ} (hL : 2 ≤ L) (hQ : 2 ≤ Q) (hA : 2 ≤ A)
    (hiδ : 2 ≤ iδ) (hic : 2 ≤ ic) (hiξ : 2 ≤ iξ) (hip : 2 ≤ ip) (his : 2 ≤ is) :
    0 ≤ (Q + 1)
        * (loopPool (L + Q + 4) Q A (1 / iδ) (1 / ic) (1 / iξ) (1 / ip) (1 / 2 - 1 / is) + 1)
      ∧ (Q + 1)
        * (loopPool (L + Q + 4) Q A (1 / iδ) (1 / ic) (1 / iξ) (1 / ip) (1 / 2 - 1 / is) + 1)
        ≤ 2 ^ 13 * (Q * (log ((L + Q) * A * iδ * ic * iξ * is) * ip * is ^ 2
          + log (Q * iδ) * ip ^ 2)) := by
  set X := (L + Q) * A * iδ * ic * iξ * is with hXd
  have hA1 : 1 ≤ A := by linarith
  have hδ1 : 1 ≤ iδ := by linarith
  have hc1 : 1 ≤ ic := by linarith
  have hξ1 : 1 ≤ iξ := by linarith
  have hp1 : 1 ≤ ip := by linarith
  have hs1 : 1 ≤ is := by linarith
  have hQ1 : 1 ≤ Q := by linarith
  have hLQ : 4 ≤ L + Q := by linarith
  have hr : 2 ≤ A * iδ * ic * iξ * is := by
    have := one_le_mul₂ (one_le_mul₂ (one_le_mul₂ hδ1 hc1) hξ1) hs1
    nlinarith
  have hLQX : 2 * (L + Q) ≤ X := by
    rw [hXd, show (L + Q) * A * iδ * ic * iξ * is = (L + Q) * (A * iδ * ic * iξ * is) by ring]
    nlinarith
  have hX : 2 ≤ X := by linarith
  have hQX : Q ≤ X := by linarith
  have hAX : A ≤ X := le_of_eq_mul₁ (c := A) (r := (L + Q) * iδ * ic * iξ * is) (by linarith)
    (one_le_mul₂ (one_le_mul₂ (one_le_mul₂ (one_le_mul₂ (by linarith) hδ1) hc1) hξ1) hs1)
    (by rw [hXd]; ring)
  have hδX : iδ ≤ X := le_of_eq_mul₁ (c := iδ) (r := (L + Q) * A * ic * iξ * is) (by linarith)
    (one_le_mul₂ (one_le_mul₂ (one_le_mul₂ (one_le_mul₂ (by linarith) hA1) hc1) hξ1) hs1)
    (by rw [hXd]; ring)
  have hcX : ic ≤ X := le_of_eq_mul₁ (c := ic) (r := (L + Q) * A * iδ * iξ * is) (by linarith)
    (one_le_mul₂ (one_le_mul₂ (one_le_mul₂ (one_le_mul₂ (by linarith) hA1) hδ1) hξ1) hs1)
    (by rw [hXd]; ring)
  have hξX : iξ ≤ X := le_of_eq_mul₁ (c := iξ) (r := (L + Q) * A * iδ * ic * is) (by linarith)
    (one_le_mul₂ (one_le_mul₂ (one_le_mul₂ (one_le_mul₂ (by linarith) hA1) hδ1) hc1) hs1)
    (by rw [hXd]; ring)
  have hsX : is ≤ X := le_of_eq_mul₁ (c := is) (r := (L + Q) * A * iδ * ic * iξ) (by linarith)
    (one_le_mul₂ (one_le_mul₂ (one_le_mul₂ (one_le_mul₂ (by linarith) hA1) hδ1) hc1) hξ1)
    (by rw [hXd]; ring)
  have hℓ1 : 1 ≤ L + Q + 4 := by linarith
  have hℓX : L + Q + 4 ≤ 4 * X := by linarith
  have hP := loopPool_le_log hℓ1 hQ hA1 hδ1 hc1 hξ1 hp1 hs1 hX hℓX hQX hAX hδX hcX hξX hsX
  have hP1 := loopPool_ge hℓ1 hQ1 hA1 hδ1 hc1 hξ1 hp1 hs1
  have hlX : log 2 ≤ log X := log_two_le_log hX
  have hl2 := Real.log_two_gt_d9
  have hQδ : 2 ≤ Q * iδ := by nlinarith
  have hlQ : 0 ≤ log (Q * iδ) := Real.log_nonneg (by linarith)
  have hps : 1 ≤ ip * is ^ 2 := one_le_mul₂ hp1 (one_le_pow₀ hs1)
  have h1 : 1 ≤ 2 * log X * ip * is ^ 2 := by
    have : 1 ≤ 2 * log X := by linarith
    have e : 2 * log X * ip * is ^ 2 = 2 * log X * (ip * is ^ 2) := by ring
    rw [e]; exact one_le_mul₂ this hps
  have hb0 : 0 ≤ log X * ip * is ^ 2 := by
    have : 0 ≤ log X := by linarith
    positivity
  have hc0 : 0 ≤ log (Q * iδ) * ip ^ 2 := by positivity
  constructor
  · positivity
  · have h2 : loopPool (L + Q + 4) Q A (1 / iδ) (1 / ic) (1 / iξ) (1 / ip) (1 / 2 - 1 / is) + 1
        ≤ 2402 * (log X * ip * is ^ 2) + 576 * (log (Q * iδ) * ip ^ 2) := by linarith
    have h3 : (Q + 1) ≤ 2 * Q := by linarith
    calc (Q + 1) * (loopPool (L + Q + 4) Q A (1 / iδ) (1 / ic) (1 / iξ) (1 / ip) (1 / 2 - 1 / is)
          + 1)
        ≤ (2 * Q) * (2402 * (log X * ip * is ^ 2) + 576 * (log (Q * iδ) * ip ^ 2)) := by
          gcongr
      _ ≤ _ := by nlinarith

lemma claim_probes_le {L Q A iδ ic : ℝ} (hL : 2 ≤ L) (hQ : 2 ≤ Q) (hA : 2 ≤ A) (hiδ : 2 ≤ iδ)
    (hic : 2 ≤ ic) :
    0 ≤ (Q + 1) * loopBudget (L + Q + 4) Q A (1 / iδ) (1 / ic)
      ∧ (Q + 1) * loopBudget (L + Q + 4) Q A (1 / iδ) (1 / ic)
        ≤ 2 ^ 18 * (Q ^ 3 * A ^ 2 * (L + Q) * ic * log (Q * A * iδ)) := by
  rw [loopBudget_inv]
  set S := Q * ((Q + 2) * A + 1) + 2 * A + 1 with hS
  set Y := Q * A * iδ with hYd
  have hQA : 4 ≤ Q * A := by nlinarith
  have hY : 8 ≤ Y := by rw [hYd]; nlinarith
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
    exact one_le_mul₂ (by nlinarith) (by nlinarith)
  have hzY : z ≤ 2 ^ 14 * Y ^ 6 := by
    have hQY : Q + 1 ≤ 2 * Y := by rw [hYd]; nlinarith
    have hδY : iδ ≤ Y := by rw [hYd]; nlinarith
    have : z ≤ 16 * (7 * Y ^ 2) * (7 * Y ^ 2 + Y ^ 2) * (8 * (2 * Y) * Y) := by
      rw [hz]; gcongr <;> nlinarith
    nlinarith
  have hlz : log z ≤ 20 * log Y := by
    have := log_le_of_le_two_pow_mul (y := z) 14 6 (by linarith) (by linarith) hzY
    push_cast at this
    linarith
  have hlz0 : 0 ≤ log z := Real.log_nonneg hz1
  have hlY : log 2 ≤ log Y := log_two_le_log (by linarith)
  have hl2 := Real.log_two_gt_d9
  have hK : 36 * S * (4 * (A + 1) * ic) * log z + S + 1
      ≤ 40400 * (Q ^ 2 * A ^ 2 * ic * log Y) := by
    have h1 : 36 * S * (4 * (A + 1) * ic) * log z ≤ 36 * (7 * (Q ^ 2 * A)) * (4 * (2 * A) * ic)
        * (20 * log Y) := by gcongr; linarith
    have h2 : S + 1 ≤ 8 * (Q ^ 2 * A) := by nlinarith
    have h3 : Q ^ 2 * A ≤ Q ^ 2 * A ^ 2 * ic := by
      have : A ≤ A ^ 2 * ic := by nlinarith
      nlinarith [mul_le_mul_of_nonneg_left this (show 0 ≤ Q ^ 2 by positivity)]
    have h4 : 8 * (Q ^ 2 * A ^ 2 * ic) ≤ 12 * (Q ^ 2 * A ^ 2 * ic * log Y) := by
      have : 0 ≤ Q ^ 2 * A ^ 2 * ic := by positivity
      nlinarith
    nlinarith
  have hK0 : 0 ≤ 36 * S * (4 * (A + 1) * ic) * log z + S + 1 := by positivity
  constructor
  · positivity
  · have hℓ : L + Q + 4 ≤ 2 * (L + Q) := by linarith
    calc (Q + 1) * ((36 * S * (4 * (A + 1) * ic) * log z + S + 1) * (L + Q + 4))
        ≤ (2 * Q) * ((40400 * (Q ^ 2 * A ^ 2 * ic * log Y)) * (2 * (L + Q))) := by
          gcongr <;> linarith
      _ = 161600 * (Q ^ 3 * A ^ 2 * (L + Q) * ic * log Y) := by ring
      _ ≤ _ := by
          have : 0 ≤ Q ^ 3 * A ^ 2 * (L + Q) * ic * log Y := by
            have : 0 ≤ log Y := by linarith
            have : 0 ≤ L + Q := by linarith
            positivity
          nlinarith

lemma claim_heldOut_le {Q iδ iξ : ℝ} (hQ : 2 ≤ Q) (hiδ : 2 ≤ iδ) (hiξ : 2 ≤ iξ) :
    (⌈32 * log (16 * (Q + 1) * (Q + 2) / (1 / iδ)) / (1 / iξ) ^ 2⌉₊ : ℝ)
      ≤ 2 ^ 9 * (log (Q * iδ) * iξ ^ 2) := by
  have e : 32 * log (16 * (Q + 1) * (Q + 2) / (1 / iδ)) / (1 / iξ) ^ 2
      = 32 * log (16 * (Q + 1) * (Q + 2) * iδ) * iξ ^ 2 := by
    rw [show 16 * (Q + 1) * (Q + 2) / (1 / iδ) = 16 * (Q + 1) * (Q + 2) * iδ by field_simp]
    field_simp
  rw [e]
  have hQδ : 4 ≤ Q * iδ := by nlinarith
  have hb : 16 * (Q + 1) * (Q + 2) * iδ ≤ 2 ^ 6 * (Q * iδ) ^ 2 := by
    have e1 : 16 * (Q + 1) * (Q + 2) ≤ 64 * Q ^ 2 := by nlinarith
    have e2 := mul_le_mul_of_nonneg_right e1 (by linarith : (0 : ℝ) ≤ iδ)
    have e3 : 64 * Q ^ 2 * iδ ≤ 64 * Q ^ 2 * iδ ^ 2 :=
      mul_le_mul_of_nonneg_left (by nlinarith) (by positivity)
    nlinarith
  have hl := log_le_of_le_two_pow_mul (y := 16 * (Q + 1) * (Q + 2) * iδ) 6 2 (by positivity)
    (by linarith) hb
  push_cast at hl
  have hl0 : 0 ≤ log (16 * (Q + 1) * (Q + 2) * iδ) := Real.log_nonneg (by nlinarith)
  have hlQ : 1 ≤ log (Q * iδ) := by
    have := log_two_le_log (show 2 ≤ Q * iδ / 2 by linarith)
    rw [Real.log_div (by positivity) (by norm_num)] at this
    have := Real.log_two_gt_d9
    linarith
  have hc := Nat.ceil_lt_add_one (show 0 ≤ 32 * log (16 * (Q + 1) * (Q + 2) * iδ) * iξ ^ 2 by
    positivity)
  have hξ2 : 4 ≤ iξ ^ 2 := by nlinarith
  nlinarith


/-- `LearnerLoopDFA`'s `L₀`, with `nQ` and `nA` standing for `|Q|` and `|Σ|` as exponents. -/
def loopL0 (Q A : ℝ) (nQ nA : ℕ) (δ c ξ pAP η₀ : ℝ) : ℝ :=
  let δ' : ℝ := δ / (8 * (Q + 1))
  let s : ℝ := 1 / 2 - η₀
  let S : ℝ := Q * ((Q + 2) * A + 1) + 2 * A + 1
  let n : ℕ := ⌈32 * Real.log (16 * (Q + 1) * (Q + 2) / δ) / ξ ^ 2⌉₊
  let aε : ℝ := 4 * (A + 1) / c
  let Ce : ℝ := aε + 8 / ξ
  let Θ : ℝ := Ce / (2 * s ^ 2) + 8 * Ce + 4608 * Q * (Q + 1) * aε
  let K₁ : ℝ := 36 * S * aε * Real.log (16 * S * (S + 1) / δ') + S + 1
  let Wm : ℝ := 16 / 5 * S * Real.log (8 * S * (S + 1) / δ') + 3 + (Q + 2) * A * (Q + 1)
  let W₀ : ℝ := Real.exp 1 * Real.sqrt δ' * Wm / 2 + 1
  let P₀ : ℝ := (50 * (max (Real.log (20 * Θ)) (Real.log (8 * W₀ / δ')) + 2) / s ^ 2 + 20 / s
      + 28) / pAP + 64 * Real.log ((Q + 4) / (δ' / 2)) / pAP ^ 2
  let R₀ : ℝ := 161000 * (Q + 2) ^ 2 * Θ ^ 2
    * (Real.log (16 * (Q + 4) * P₀ * Θ / (δ' ^ 2 * pAP)) + 2) / s ^ 4
  let E₁ : ℝ := (Q + 3) ^ 2 * (2 ^ 20 * Θ * (R₀ + 1) / s
    + 2 ^ 36 * Θ ^ 2 * (13 + 2 * Real.logb 2 (R₀ + 1)) / δ' + 2 ^ 16 * (P₀ + 2) / (pAP * δ'))
  let Kc : ℝ := (1 + (Q + 1) ^ 3 * A) ^ nQ * (Q + 3) ^ ((nQ + 2) * nA)
  let E : ℝ := 1536 * aε * E₁ + 1536 * Real.log (16 * (Q + 1) * Kc / δ') * aε
  let E₂ : ℝ := 4 * (Q + 1) * n * (6 * (Q + 2) * R₀ + K₁ + n) / δ
  let E₃ : ℝ := 16 * K₁ ^ 2 / δ'
  let T : ℝ := 2 * (Real.log (E + E₂ + E₃) + 12) / Real.log A + Q + 4
  2 * (Real.log (E + E₂ + E₃) + 6 * Real.log T) / Real.log A
    + 144 * (Real.log T - 2) / (Real.log A * (T * Real.log A - 12))

set_option maxHeartbeats 4000000 in -- `L₀` bounds about twenty nested quantities
lemma claim_L0_le {Q A iδ ic iξ ip is : ℝ} (hQ : 8 ≤ Q) (hA : 2 ≤ A) (hiδ : 2 ≤ iδ)
    (hic : 2 ≤ ic) (hiξ : 2 ≤ iξ) (hip : 2 ≤ ip) (his : 2 ≤ is) :
    |loopL0 Q A ⌊Q⌋₊ ⌊A⌋₊ (1 / iδ) (1 / ic) (1 / iξ) (1 / ip) (1 / 2 - 1 / is)|
      ≤ 1000 * |log (Q * A * iδ * ic * iξ * ip * is) / log A| := by
  unfold loopL0
  extract_lets δ' s S n aε Ce Θ K₁ Wm W₀ P₀ R₀ E₁ Kc E E₂ E₃ T
  have hΘe : Θ = loopΘ Q A (1 / ic) (1 / iξ) (1 / 2 - 1 / is) := rfl
  have hW₀e : W₀ = loopW₀ Q A (1 / iδ) := rfl
  have hP₀e : P₀ = loopPool (exp 1) Q A (1 / iδ) (1 / ic) (1 / iξ) (1 / ip) (1 / 2 - 1 / is) := by
    unfold loopPool
    rw [Real.log_exp, mul_one]
    rfl
  have hδ' : δ' = 1 / iδ / (8 * (Q + 1)) := rfl
  have hs : s = 1 / 2 - (1 / 2 - 1 / is) := rfl
  have hS : S = Q * ((Q + 2) * A + 1) + 2 * A + 1 := rfl
  have hn : n = ⌈32 * log (16 * (Q + 1) * (Q + 2) / (1 / iδ)) / (1 / iξ) ^ 2⌉₊ := rfl
  have haε : aε = 4 * (A + 1) / (1 / ic) := rfl
  have hK₁ : K₁ = 36 * S * aε * log (16 * S * (S + 1) / δ') + S + 1 := rfl
  have hR₀ : R₀ = 161000 * (Q + 2) ^ 2 * Θ ^ 2
      * (log (16 * (Q + 4) * P₀ * Θ / (δ' ^ 2 * (1 / ip))) + 2) / s ^ 4 := rfl
  have hE₁ : E₁ = (Q + 3) ^ 2 * (2 ^ 20 * Θ * (R₀ + 1) / s
      + 2 ^ 36 * Θ ^ 2 * (13 + 2 * Real.logb 2 (R₀ + 1)) / δ'
      + 2 ^ 16 * (P₀ + 2) / (1 / ip * δ')) := rfl
  have hKc : Kc = (1 + (Q + 1) ^ 3 * A) ^ ⌊Q⌋₊ * (Q + 3) ^ ((⌊Q⌋₊ + 2) * ⌊A⌋₊) := rfl
  have hE : E = 1536 * aε * E₁ + 1536 * log (16 * (Q + 1) * Kc / δ') * aε := rfl
  have hE₂ : E₂ = 4 * (Q + 1) * n * (6 * (Q + 2) * R₀ + K₁ + n) / (1 / iδ) := rfl
  have hE₃ : E₃ = 16 * K₁ ^ 2 / δ' := rfl
  have hT : T = 2 * (log (E + E₂ + E₃) + 12) / log A + Q + 4 := rfl
  clear_value δ' s S n aε Ce Θ K₁ Wm W₀ P₀ R₀ E₁ Kc E E₂ E₃ T
  set X := Q * A * iδ * ic * iξ * ip * is with hXd
  have hQ1 : 1 ≤ Q := by linarith
  have hA1 : 1 ≤ A := by linarith
  have hδ1 : 1 ≤ iδ := by linarith
  have hc1 : 1 ≤ ic := by linarith
  have hξ1 : 1 ≤ iξ := by linarith
  have hp1 : 1 ≤ ip := by linarith
  have hs1 : 1 ≤ is := by linarith
  have hQX : Q ≤ X := le_of_eq_mul₁ (c := Q) (r := A * iδ * ic * iξ * ip * is) (by linarith)
    (one_le_mul₂ (one_le_mul₂ (one_le_mul₂ (one_le_mul₂ (one_le_mul₂ hA1 hδ1) hc1) hξ1) hp1) hs1)
    (by rw [hXd]; ring)
  have hAX : A ≤ X := le_of_eq_mul₁ (c := A) (r := Q * iδ * ic * iξ * ip * is) (by linarith)
    (one_le_mul₂ (one_le_mul₂ (one_le_mul₂ (one_le_mul₂ (one_le_mul₂ hQ1 hδ1) hc1) hξ1) hp1) hs1)
    (by rw [hXd]; ring)
  have hδX : iδ ≤ X := le_of_eq_mul₁ (c := iδ) (r := Q * A * ic * iξ * ip * is) (by linarith)
    (one_le_mul₂ (one_le_mul₂ (one_le_mul₂ (one_le_mul₂ (one_le_mul₂ hQ1 hA1) hc1) hξ1) hp1) hs1)
    (by rw [hXd]; ring)
  have hcX : ic ≤ X := le_of_eq_mul₁ (c := ic) (r := Q * A * iδ * iξ * ip * is) (by linarith)
    (one_le_mul₂ (one_le_mul₂ (one_le_mul₂ (one_le_mul₂ (one_le_mul₂ hQ1 hA1) hδ1) hξ1) hp1) hs1)
    (by rw [hXd]; ring)
  have hξX : iξ ≤ X := le_of_eq_mul₁ (c := iξ) (r := Q * A * iδ * ic * ip * is) (by linarith)
    (one_le_mul₂ (one_le_mul₂ (one_le_mul₂ (one_le_mul₂ (one_le_mul₂ hQ1 hA1) hδ1) hc1) hp1) hs1)
    (by rw [hXd]; ring)
  have hpX : ip ≤ X := le_of_eq_mul₁ (c := ip) (r := Q * A * iδ * ic * iξ * is) (by linarith)
    (one_le_mul₂ (one_le_mul₂ (one_le_mul₂ (one_le_mul₂ (one_le_mul₂ hQ1 hA1) hδ1) hc1) hξ1) hs1)
    (by rw [hXd]; ring)
  have hsX : is ≤ X := le_of_eq_mul₁ (c := is) (r := Q * A * iδ * ic * iξ * ip) (by linarith)
    (one_le_mul₂ (one_le_mul₂ (one_le_mul₂ (one_le_mul₂ (one_le_mul₂ hQ1 hA1) hδ1) hc1) hξ1) hp1)
    (by rw [hXd]; ring)
  clear_value X
  have hX : 8 ≤ X := by linarith
  have hX1 : 1 ≤ X := by linarith
  have hpow : ∀ {a b : ℕ}, a ≤ b → X ^ a ≤ X ^ b := fun h => pow_le_pow_right₀ hX1 h
  have hXp : ∀ a : ℕ, 0 ≤ X ^ a := fun a => by positivity
  have hX1p : ∀ a : ℕ, 1 ≤ X ^ a := fun a => one_le_pow₀ hX1
  -- the rates
  set D := 8 * (Q + 1) * iδ with hDd
  have hD1 : 1 ≤ D := by rw [hDd]; nlinarith only [hQ1, hδ1]
  have hDX : D ≤ 16 * X ^ 2 := by
    rw [hDd]; nlinarith only [hQX, hδX, hQ1, hδ1, hX1]
  clear_value D
  have hδ'0 : 0 < δ' := by rw [hδ']; positivity
  have hδ'D : ∀ x, x / δ' = x * D := fun x => by rw [hδ', hDd]; field_simp
  have hsis : s = 1 / is := by rw [hs]; ring
  have hsd : ∀ x, x / s = x * is := fun x => by rw [hsis]; field_simp
  have haε' : aε = 4 * (A + 1) * ic := by rw [haε]; field_simp
  have haε1 : 1 ≤ aε := by rw [haε']; nlinarith only [hA1, hc1]
  have haεX : aε ≤ 8 * X ^ 2 := by rw [haε']; nlinarith only [hAX, hcX, hA1, hc1]
  obtain ⟨hΘ1, -⟩ := loopΘ_le hQ1 hA1 hc1 hξ1 hs1
  have hΘX := loopΘ_le_pow hQ1 hA1 hc1 hξ1 hs1 hQX hAX hcX hξX hsX
  rw [← hΘe] at hΘ1 hΘX
  have hS1 : 1 ≤ S := by
    have : 0 ≤ Q * ((Q + 2) * A + 1) := by positivity
    rw [hS]; linarith
  have hSX : S ≤ 7 * X ^ 3 := by
    have : S ≤ X * ((X + 2 * X) * X + X) + 2 * X + X := by
      rw [hS]; gcongr <;> linarith
    nlinarith only [this, hpow (show 2 ≤ 3 by norm_num), hpow (show 1 ≤ 3 by norm_num), hX1]
  have hex : 1 ≤ exp 1 := by linarith [Real.add_one_le_exp (1 : ℝ)]
  have he1 : exp 1 ≤ 3 := by have := Real.exp_one_lt_d9; linarith
  have hP₀1 : 1 ≤ P₀ := by
    rw [hP₀e]; exact loopPool_ge hex hQ1 hA1 hδ1 hc1 hξ1 hp1 hs1
  have hP₀X : P₀ ≤ 2 ^ 34 * X ^ 16 := by
    rw [hP₀e]
    exact loopPool_le_pow hex hQ1 hA1 hδ1 hc1 hξ1 hp1 hs1 (by linarith) hQX hAX hδX hcX hξX hpX hsX
  -- the prefix count
  set Z := 16 * (Q + 4) * P₀ * Θ / (δ' ^ 2 * (1 / ip)) with hZd
  have hZe : Z = 16 * (Q + 4) * P₀ * Θ * D ^ 2 * ip := by
    rw [hZd, show δ' ^ 2 * (1 / ip) = δ' * δ' * (1 / ip) by ring, ← div_div, ← div_div, hδ'D,
      hδ'D]
    field_simp
  clear_value Z
  have hZ1 : 1 ≤ Z := by
    rw [hZe]
    exact one_le_mul₂ (one_le_mul₂ (one_le_mul₂ (one_le_mul₂ (by linarith) hP₀1) hΘ1)
      (one_le_pow₀ hD1)) hp1
  have hZX : Z ≤ 2 ^ 68 * X ^ 26 := by
    have : Z ≤ 16 * (5 * X) * (2 ^ 34 * X ^ 16) * (2 ^ 19 * X ^ 4) * (16 * X ^ 2) ^ 2 * X := by
      rw [hZe]; gcongr <;> linarith
    have e : 16 * (5 * X) * (2 ^ 34 * X ^ 16) * (2 ^ 19 * X ^ 4) * (16 * X ^ 2) ^ 2 * X
        = 5 * 2 ^ 65 * X ^ 26 := by ring
    linarith only [this, e, hXp 26]
  have hlZ0 : 0 ≤ log Z := Real.log_nonneg hZ1
  have hlZ : log Z + 2 ≤ 2 ^ 69 * X ^ 26 := by
    have := Real.log_le_self (show 0 ≤ Z by linarith)
    linarith only [this, hZX, hX1p 26]
  have hR₀e : R₀ = 161000 * (Q + 2) ^ 2 * Θ ^ 2 * (log Z + 2) * is ^ 4 := by
    rw [hR₀, hsis]; field_simp
  have hR₀0 : 0 ≤ R₀ := by rw [hR₀e]; positivity
  have hR₀X : R₀ + 1 ≤ 2 ^ 128 * X ^ 40 := by
    have : R₀ ≤ 161000 * (2 * X) ^ 2 * (2 ^ 19 * X ^ 4) ^ 2 * (2 ^ 69 * X ^ 26) * X ^ 4 := by
      rw [hR₀e]; gcongr <;> linarith
    have e : 161000 * (2 * X) ^ 2 * (2 ^ 19 * X ^ 4) ^ 2 * (2 ^ 69 * X ^ 26) * X ^ 4
        = 644000 * 2 ^ 107 * X ^ 40 := by ring
    linarith only [this, e, hX1p 40]
  have hlb : Real.logb 2 (R₀ + 1) ≤ 2 * (R₀ + 1) := by
    rw [Real.logb, div_le_iff₀ (Real.log_pos (by norm_num))]
    have := Real.log_le_self (show 0 ≤ R₀ + 1 by linarith)
    have := Real.log_two_gt_d9
    nlinarith only [this, hR₀0, ‹log (R₀ + 1) ≤ R₀ + 1›]
  have hlb0 : 0 ≤ Real.logb 2 (R₀ + 1) := Real.logb_nonneg (by norm_num) (by linarith)
  -- `E₁`
  have hE₁e : E₁ = (Q + 3) ^ 2 * (2 ^ 20 * Θ * (R₀ + 1) * is
      + 2 ^ 36 * Θ ^ 2 * (13 + 2 * Real.logb 2 (R₀ + 1)) * D + 2 ^ 16 * (P₀ + 2) * ip * D) := by
    rw [hE₁, hsd, hδ'D, show 2 ^ 16 * (P₀ + 2) / (1 / ip * δ') = 2 ^ 16 * (P₀ + 2) * ip / δ' by
      field_simp, hδ'D]
  have hE₁1 : 1 ≤ E₁ := by
    rw [hE₁e]
    have h1 : 1 ≤ 2 ^ 20 * Θ * (R₀ + 1) * is := one_le_mul₂ (one_le_mul₂ (one_le_mul₂
      (by norm_num) hΘ1) (by linarith)) hs1
    have h2 : 0 ≤ 2 ^ 36 * Θ ^ 2 * (13 + 2 * Real.logb 2 (R₀ + 1)) * D := by positivity
    have h3 : 0 ≤ 2 ^ 16 * (P₀ + 2) * ip * D := by positivity
    exact one_le_mul₂ (one_le_pow₀ (by linarith)) (by linarith)
  have hE₁X : E₁ ≤ 2 ^ 212 * X ^ 52 := by
    have ha : 2 ^ 20 * Θ * (R₀ + 1) * is ≤ 2 ^ 20 * (2 ^ 19 * X ^ 4) * (2 ^ 128 * X ^ 40) * X := by
      gcongr
    have hlb' : 13 + 2 * Real.logb 2 (R₀ + 1) ≤ 2 ^ 131 * X ^ 40 := by
      linarith only [hlb, hR₀X, hX1p 40]
    have hb : 2 ^ 36 * Θ ^ 2 * (13 + 2 * Real.logb 2 (R₀ + 1)) * D
        ≤ 2 ^ 36 * (2 ^ 19 * X ^ 4) ^ 2 * (2 ^ 131 * X ^ 40) * (16 * X ^ 2) := by
      gcongr
    have hP₀2 : P₀ + 2 ≤ 2 ^ 35 * X ^ 16 := by linarith only [hP₀X, hX1p 16]
    have hc : 2 ^ 16 * (P₀ + 2) * ip * D ≤ 2 ^ 16 * (2 ^ 35 * X ^ 16) * X * (16 * X ^ 2) := by
      gcongr
    have hq : (Q + 3) ^ 2 ≤ (2 * X) ^ 2 := by gcongr; linarith
    rw [hE₁e]
    have e1 : 2 ^ 20 * (2 ^ 19 * X ^ 4) * (2 ^ 128 * X ^ 40) * X = 2 ^ 167 * X ^ 45 := by ring
    have e2 : 2 ^ 36 * (2 ^ 19 * X ^ 4) ^ 2 * (2 ^ 131 * X ^ 40) * (16 * X ^ 2)
        = 2 ^ 209 * X ^ 50 := by ring
    have e3 : 2 ^ 16 * (2 ^ 35 * X ^ 16) * X * (16 * X ^ 2) = 2 ^ 55 * X ^ 19 := by ring
    have hsum : 2 ^ 20 * Θ * (R₀ + 1) * is + 2 ^ 36 * Θ ^ 2 * (13 + 2 * Real.logb 2 (R₀ + 1)) * D
        + 2 ^ 16 * (P₀ + 2) * ip * D ≤ 2 ^ 210 * X ^ 50 := by
      linarith only [ha, hb, hc, e1, e2, e3, hpow (show 45 ≤ 50 by norm_num),
        hpow (show 19 ≤ 50 by norm_num), hXp 50]
    calc (Q + 3) ^ 2 * (2 ^ 20 * Θ * (R₀ + 1) * is
          + 2 ^ 36 * Θ ^ 2 * (13 + 2 * Real.logb 2 (R₀ + 1)) * D + 2 ^ 16 * (P₀ + 2) * ip * D)
        ≤ (2 * X) ^ 2 * (2 ^ 210 * X ^ 50) := by gcongr
      _ = 2 ^ 212 * X ^ 52 := by ring
  -- `E`
  have hKc1 : 1 ≤ Kc := by
    rw [hKc]
    exact one_le_mul₂ (one_le_pow₀ (le_add_of_nonneg_right (by positivity)))
      (one_le_pow₀ (by linarith))
  have hfQ : (⌊Q⌋₊ : ℝ) ≤ Q := Nat.floor_le (by linarith)
  have hfA : (⌊A⌋₊ : ℝ) ≤ A := Nat.floor_le (by linarith)
  have hlKc : log Kc ≤ 15 * X ^ 5 := by
    rw [hKc, Real.log_mul (by positivity) (by positivity), Real.log_pow, Real.log_pow]
    have h1 : log (1 + (Q + 1) ^ 3 * A) ≤ 1 + (2 * X) ^ 3 * X := by
      refine (Real.log_le_self (by positivity)).trans ?_
      gcongr <;> linarith
    have h2 : log (Q + 3) ≤ 2 * X := (Real.log_le_self (by positivity)).trans (by linarith)
    have hl1 : 0 ≤ log (1 + (Q + 1) ^ 3 * A) :=
      Real.log_nonneg (le_add_of_nonneg_right (by positivity))
    have hl2 : 0 ≤ log (Q + 3) := Real.log_nonneg (by linarith)
    have h3 : (⌊Q⌋₊ : ℝ) * log (1 + (Q + 1) ^ 3 * A) ≤ X * (1 + (2 * X) ^ 3 * X) := by
      gcongr; linarith
    have h4 : (((⌊Q⌋₊ + 2) * ⌊A⌋₊ : ℕ) : ℝ) * log (Q + 3) ≤ (X + 2) * X * (2 * X) := by
      push_cast
      gcongr <;> linarith
    have e : X * (1 + (2 * X) ^ 3 * X) + (X + 2) * X * (2 * X)
        = X + 8 * X ^ 5 + 2 * X ^ 3 + 4 * X ^ 2 := by ring
    linarith only [h3, h4, e, hpow (show 1 ≤ 5 by norm_num), hpow (show 3 ≤ 5 by norm_num),
      hpow (show 2 ≤ 5 by norm_num)]
  have hlg : log (16 * (Q + 1) * Kc / δ') ≤ 2 ^ 10 * X ^ 5 := by
    rw [hδ'D, show 16 * (Q + 1) * Kc * D = (16 * (Q + 1) * D) * Kc by ring,
      Real.log_mul (by positivity) (by positivity)]
    have h1 : log (16 * (Q + 1) * D) ≤ 16 * (2 * X) * (16 * X ^ 2) := by
      refine (Real.log_le_self (by positivity)).trans ?_
      gcongr; linarith
    have e : 16 * (2 * X) * (16 * X ^ 2) = 512 * X ^ 3 := by ring
    linarith only [h1, e, hlKc, hpow (show 3 ≤ 5 by norm_num), hXp 5]
  have hlg0 : 0 ≤ log (16 * (Q + 1) * Kc / δ') := by
    rw [hδ'D]
    exact Real.log_nonneg (one_le_mul₂ (one_le_mul₂ (by linarith) hKc1) hD1)
  have hE1 : 1 ≤ E := by
    rw [hE]
    have : 1 ≤ 1536 * aε * E₁ := one_le_mul₂ (one_le_mul₂ (by norm_num) haε1) hE₁1
    have : 0 ≤ 1536 * log (16 * (Q + 1) * Kc / δ') * aε := by positivity
    linarith
  have hEX : E ≤ 2 ^ 227 * X ^ 54 := by
    rw [hE]
    have h1 : 1536 * aε * E₁ ≤ 1536 * (8 * X ^ 2) * (2 ^ 212 * X ^ 52) := by gcongr
    have h2 : 1536 * log (16 * (Q + 1) * Kc / δ') * aε ≤ 1536 * (2 ^ 10 * X ^ 5) * (8 * X ^ 2) := by
      gcongr
    have e1 : 1536 * (8 * X ^ 2) * (2 ^ 212 * X ^ 52) = 12288 * 2 ^ 212 * X ^ 54 := by ring
    have e2 : 1536 * (2 ^ 10 * X ^ 5) * (8 * X ^ 2) = 12288 * 2 ^ 10 * X ^ 7 := by ring
    linarith only [h1, h2, e1, e2, hpow (show 7 ≤ 54 by norm_num), hXp 54]
  -- `E₂` and `E₃`
  have hn0 : (0 : ℝ) ≤ n := Nat.cast_nonneg _
  have hnX : (n : ℝ) ≤ 2 ^ 12 * X ^ 5 := by
    have e : 32 * log (16 * (Q + 1) * (Q + 2) / (1 / iδ)) / (1 / iξ) ^ 2
        = 32 * log (16 * (Q + 1) * (Q + 2) * iδ) * iξ ^ 2 := by
      rw [show 16 * (Q + 1) * (Q + 2) / (1 / iδ) = 16 * (Q + 1) * (Q + 2) * iδ by field_simp]
      field_simp
    have hl : log (16 * (Q + 1) * (Q + 2) * iδ) ≤ 16 * (2 * X) * (2 * X) * X := by
      refine (Real.log_le_self (by positivity)).trans ?_
      gcongr <;> linarith
    have hl0 : 0 ≤ log (16 * (Q + 1) * (Q + 2) * iδ) :=
      Real.log_nonneg (one_le_mul₂ (by nlinarith only [hQ1]) hδ1)
    have hc := Nat.ceil_lt_add_one (show 0 ≤ 32 * log (16 * (Q + 1) * (Q + 2) * iδ) * iξ ^ 2 by
      positivity)
    rw [hn, e]
    have h1 : 32 * log (16 * (Q + 1) * (Q + 2) * iδ) * iξ ^ 2
        ≤ 32 * (16 * (2 * X) * (2 * X) * X) * X ^ 2 := by gcongr
    have e2 : 32 * (16 * (2 * X) * (2 * X) * X) * X ^ 2 = 2048 * X ^ 5 := by ring
    linarith only [hc, h1, e2, hX1p 5]
  have hlS : log (16 * S * (S + 1) / δ') ≤ 2 ^ 14 * X ^ 8 := by
    rw [hδ'D]
    refine (Real.log_le_self (by positivity)).trans ?_
    have : 16 * S * (S + 1) * D ≤ 16 * (7 * X ^ 3) * (7 * X ^ 3 + X ^ 3) * (16 * X ^ 2) := by
      gcongr; linarith [hX1p 3]
    have e : 16 * (7 * X ^ 3) * (7 * X ^ 3 + X ^ 3) * (16 * X ^ 2) = 14336 * X ^ 8 := by ring
    linarith only [this, e, hXp 8]
  have hlS0 : 0 ≤ log (16 * S * (S + 1) / δ') := by
    rw [hδ'D]
    exact Real.log_nonneg (one_le_mul₂ (by nlinarith only [hS1]) hD1)
  have hK₁0 : 0 ≤ K₁ := by rw [hK₁]; positivity
  have hK₁X : K₁ ≤ 2 ^ 26 * X ^ 13 := by
    rw [hK₁]
    have h1 : 36 * S * aε * log (16 * S * (S + 1) / δ')
        ≤ 36 * (7 * X ^ 3) * (8 * X ^ 2) * (2 ^ 14 * X ^ 8) := by gcongr
    have e : 36 * (7 * X ^ 3) * (8 * X ^ 2) * (2 ^ 14 * X ^ 8) = 2016 * 2 ^ 14 * X ^ 13 := by ring
    linarith only [h1, e, hSX, hpow (show 3 ≤ 13 by norm_num), hX1p 13]
  have hE₂0 : 0 ≤ E₂ := by
    rw [hE₂]; positivity
  have hE₂X : E₂ ≤ 2 ^ 147 * X ^ 48 := by
    rw [hE₂, show ∀ x : ℝ, x / (1 / iδ) = x * iδ from fun x => by field_simp]
    have hin : 6 * (Q + 2) * R₀ + K₁ + n ≤ 2 ^ 132 * X ^ 41 := by
      have h1 : 6 * (Q + 2) * R₀ ≤ 6 * (2 * X) * (2 ^ 128 * X ^ 40) := by
        gcongr <;> linarith
      have e : 6 * (2 * X) * (2 ^ 128 * X ^ 40) = 12 * 2 ^ 128 * X ^ 41 := by ring
      linarith only [h1, e, hK₁X, hnX, hpow (show 13 ≤ 41 by norm_num),
        hpow (show 5 ≤ 41 by norm_num), hXp 41]
    have h0 : 0 ≤ 6 * (Q + 2) * R₀ + K₁ + n := by positivity
    calc 4 * (Q + 1) * n * (6 * (Q + 2) * R₀ + K₁ + n) * iδ
        ≤ 4 * (2 * X) * (2 ^ 12 * X ^ 5) * (2 ^ 132 * X ^ 41) * X := by
          gcongr; linarith
      _ = 2 ^ 147 * X ^ 48 := by ring
  have hE₃0 : 0 ≤ E₃ := by rw [hE₃]; positivity
  have hE₃X : E₃ ≤ 2 ^ 60 * X ^ 28 := by
    rw [hE₃, hδ'D]
    calc 16 * K₁ ^ 2 * D ≤ 16 * (2 ^ 26 * X ^ 13) ^ 2 * (16 * X ^ 2) := by gcongr
      _ = 2 ^ 60 * X ^ 28 := by ring
  -- the sum and `T`
  set Es := E + E₂ + E₃ with hEsd
  have hEs1 : 1 ≤ Es := by linarith
  have hEsX : Es ≤ 2 ^ 228 * X ^ 54 := by
    linarith only [hEX, hE₂X, hE₃X, hpow (show 48 ≤ 54 by norm_num),
      hpow (show 28 ≤ 54 by norm_num), hXp 54]
  have hlEs : log Es ≤ 282 * log X := by
    have := log_le_of_le_two_pow_mul (y := Es) 228 54 (by linarith) (by linarith) hEsX
    push_cast at this
    linarith
  have hlEs0 : 0 ≤ log Es := Real.log_nonneg hEs1
  have hlA : log 2 ≤ log A := log_two_le_log hA
  have hl2 := Real.log_two_gt_d9
  have hlA0 : 0 < log A := by linarith
  have hlX0 : 0 < log X := Real.log_pos (by linarith)
  have hlXX : log X ≤ X := Real.log_le_self (by linarith)
  have hT12 : 12 ≤ T := by
    rw [hT]
    have : 0 ≤ 2 * (log Es + 12) / log A := by positivity
    linarith
  have hTX : T ≤ 2 ^ 10 * X := by
    rw [hT]
    have h1 : 2 * (log Es + 12) / log A ≤ 2 * (282 * X + 12 * X) / (69 / 100) := by
      gcongr
      · linarith
      · linarith
      · linarith
    linarith
  have hlT : log T ≤ 11 * log X := by
    have := log_le_of_le_two_pow_mul (y := T) (X := X) 10 1 (by linarith) (by linarith)
      (by linarith)
    push_cast at this
    linarith
  have hlT2 : 2 ≤ log T := by
    rw [Real.le_log_iff_exp_le (by linarith)]
    have : Real.exp 2 = Real.exp 1 * Real.exp 1 := by rw [← Real.exp_add]; norm_num
    nlinarith only [this, he1, hex, hT12]
  have hTA : 12 ≤ T * log A - 12 := by
    have e : T * log A = 2 * (log Es + 12) + (Q + 4) * log A := by
      rw [hT]; field_simp; ring
    have : 0 ≤ (Q + 4) * log A := by positivity
    linarith
  clear_value Es
  -- the threshold
  have hden : 0 < log A * (T * log A - 12) := by positivity
  have h1 : 2 * (log Es + 6 * log T) / log A ≤ 696 * (log X / log A) := by
    rw [mul_div_assoc']
    exact div_le_div_of_nonneg_right (by linarith) hlA0.le
  have h2 : 144 * (log T - 2) / (log A * (T * log A - 12)) ≤ 132 * (log X / log A) := by
    rw [div_le_iff₀ hden]
    have e : 132 * (log X / log A) * (log A * (T * log A - 12))
        = 132 * log X * (T * log A - 12) := by
      field_simp
    rw [e]
    nlinarith only [hlT, hlT2, hTA, hlX0]
  have h0a : 0 ≤ 2 * (log Es + 6 * log T) / log A := by
    have : 0 ≤ log T := by linarith
    positivity
  have h0b : 0 ≤ 144 * (log T - 2) / (log A * (T * log A - 12)) :=
    div_nonneg (by linarith) hden.le
  rw [abs_of_nonneg (by linarith), abs_of_nonneg (by positivity)]
  linarith

lemma abs_le_mul_abs_of {w g C : ℝ} (hC : 0 ≤ C) (h0 : 0 ≤ w) (h : w ≤ C * g) : |w| ≤ C * |g| := by
  rw [abs_of_nonneg h0]
  exact h.trans (mul_le_mul_of_nonneg_left (le_abs_self g) hC)

def claimL0 : LoopVec → ℝ
  | (Q, A, iδ, ic, iξ, ip, is) =>
    loopL0 Q A ⌊Q⌋₊ ⌊A⌋₊ (1 / iδ) (1 / ic) (1 / iξ) (1 / ip) (1 / 2 - 1 / is)

def claimPref : ℝ × LoopVec → ℝ
  | (L, Q, A, iδ, ic, iξ, ip, is) =>
    (Q + 1) * (6 * (Q + 2)
      * loopCount (L + Q + 4) Q A (1 / iδ) (1 / ic) (1 / iξ) (1 / ip) (1 / 2 - 1 / is))

def claimSuff : ℝ × LoopVec → ℝ
  | (L, Q, A, iδ, ic, iξ, ip, is) =>
    (Q + 1) * (loopPool (L + Q + 4) Q A (1 / iδ) (1 / ic) (1 / iξ) (1 / ip) (1 / 2 - 1 / is) + 1)

def claimProbes : ℝ × LoopVec → ℝ
  | (L, Q, A, iδ, ic, _, _, _) => (Q + 1) * loopBudget (L + Q + 4) Q A (1 / iδ) (1 / ic)

def claimHeldOut : LoopVec → ℕ
  | (Q, _, iδ, _, iξ, _, _) => ⌈32 * log (16 * (Q + 1) * (Q + 2) / (1 / iδ)) / (1 / iξ) ^ 2⌉₊

theorem loopClaim_holds : LoopClaim := by
  refine ⟨claimL0, claimPref, claimSuff, claimProbes, claimHeldOut, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · refine isBigO_of_ge (8, 2, 2, 2, 2, 2, 2) 1000 fun b hb => ?_
    obtain ⟨Q, A, iδ, ic, iξ, ip, is⟩ := b
    simp only [Prod.mk_le_mk] at hb
    obtain ⟨hQ, hA, hiδ, hic, hiξ, hip, his⟩ := hb
    exact claim_L0_le hQ hA hiδ hic hiξ hip his
  · refine isBigO_of_ge (2, 2, 2, 2, 2, 2, 2, 2) (2 ^ 70) fun b hb => ?_
    obtain ⟨L, Q, A, iδ, ic, iξ, ip, is⟩ := b
    simp only [Prod.mk_le_mk] at hb
    obtain ⟨hL, hQ, hA, hiδ, hic, hiξ, hip, his⟩ := hb
    obtain ⟨h0, h⟩ := claim_pref_le hL hQ hA hiδ hic hiξ hip his
    exact abs_le_mul_abs_of (by positivity) h0 h
  · refine isBigO_of_ge (2, 2, 2, 2, 2, 2, 2, 2) (2 ^ 13) fun b hb => ?_
    obtain ⟨L, Q, A, iδ, ic, iξ, ip, is⟩ := b
    simp only [Prod.mk_le_mk] at hb
    obtain ⟨hL, hQ, hA, hiδ, hic, hiξ, hip, his⟩ := hb
    obtain ⟨h0, h⟩ := claim_suff_le hL hQ hA hiδ hic hiξ hip his
    exact abs_le_mul_abs_of (by positivity) h0 h
  · refine isBigO_of_ge (2, 2, 2, 2, 2, 2, 2, 2) (2 ^ 18) fun b hb => ?_
    obtain ⟨L, Q, A, iδ, ic, iξ, ip, is⟩ := b
    simp only [Prod.mk_le_mk] at hb
    obtain ⟨hL, hQ, hA, hiδ, hic, -, -, -⟩ := hb
    obtain ⟨h0, h⟩ := claim_probes_le hL hQ hA hiδ hic
    exact abs_le_mul_abs_of (by positivity) h0 h
  · refine isBigO_of_ge (2, 2, 2, 2, 2, 2, 2) (2 ^ 9) fun b hb => ?_
    obtain ⟨Q, A, iδ, ic, iξ, ip, is⟩ := b
    simp only [Prod.mk_le_mk] at hb
    obtain ⟨hQ, -, hiδ, -, hiξ, -, -⟩ := hb
    exact abs_le_mul_abs_of (by positivity) (Nat.cast_nonneg _) (claim_heldOut_le hQ hiδ hiξ)
  · intro α _ _ _ σ _ Ω _ μ _ M O L η₀ c γ δ pAP ξ v k L₁ hL hη hη₀ hc hδ hδ1 hpAP hξ hA2 hpAPu
      hceil hL₀
    have hL₀' : loopL0 (Fintype.card σ) (Fintype.card α) (Fintype.card σ) (Fintype.card α) δ c ξ
        pAP η₀ ≤ L := by
      have e : claimL0 v = loopL0 (Fintype.card σ) (Fintype.card α) (Fintype.card σ)
          (Fintype.card α) δ c ξ pAP η₀ := by
        simp only [v, claimL0, Nat.floor_natCast, one_div_one_div, sub_sub_cancel]
      rw [← e]
      exact hL₀
    obtain ⟨settings, hw, -, -, hcnt, hprob⟩ := learner_loop_dfa_counts M O L η₀ c γ δ pAP ξ hL hη
      hη₀ hc hδ hδ1 hpAP hξ hA2 hpAPu hceil hL₀'
    refine ⟨settings, hw, fun draws => ?_, ?_⟩
    · obtain ⟨h1, h2, h3⟩ := hcnt draws
      refine ⟨h1.trans_eq ?_, h2.trans_eq ?_, h3.trans_eq ?_⟩ <;>
        simp only [v, claimPref, claimSuff, claimProbes, one_div_one_div, sub_sub_cancel]
    · have e : claimHeldOut v
          = ⌈32 * log (16 * ((Fintype.card σ : ℝ) + 1) * ((Fintype.card σ : ℝ) + 2) / δ)
            / ξ ^ 2⌉₊ := by
        simp only [v, claimHeldOut, one_div_one_div]
      rw [e]
      exact hprob

end OrthoDFA
