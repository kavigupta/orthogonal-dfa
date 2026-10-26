import OrthoDFA.Claims.Loop
import OrthoDFA.Proofs.LoopDFA
import OrthoDFA.Proofs.BigO

/-!
# The learner's costs, asymptotically: the proof

`learner_loop_dfa_counts` bounds, for every draw, the loop's reads by `(|Q| + 1)` rounds of
`6 (|Q| + 2) loopCount` prefixes and `loopPool + 1` suffixes, and its probes by
`(|Q| + 1) loopBudget`, all at `ℓ = L + |Q| + 4`. In the inverted coordinates `loopΘ` is a
polynomial, so `loopCount` is `ℓ² Θ²/s⁴` times the log of a polynomial and `loopPool` a sum of
logs of polynomials; `L₀` is the log of a polynomial over `log |Σ|`. `LoopClaimSimple` follows
from `LoopClaim`, the prefix and suffix totals' tight shapes bounded by their products.
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory Asymptotics Filter Real


/-- `(|Q|, |Σ|, 1/δ, 1/c, 1/ξ, 1/pAP, 1/s₀)`. -/
private abbrev LoopVec := ℝ × ℝ × ℝ × ℝ × ℝ × ℝ × ℝ

private def vecRegion : Set LoopVec := Set.Ici (1, 2, 2, 1, 1, 1, 2)

private lemma loopΘ_inv (Q A ic iξ is : ℝ) :
    loopΘ Q A (1 / ic) (1 / iξ) (1 / 2 - 1 / is)
      = (4 * (A + 1) * ic + 8 * iξ) * is ^ 2 / 2 + 8 * (4 * (A + 1) * ic + 8 * iξ)
        + 4608 * Q * (Q + 1) * (4 * (A + 1) * ic) := by
  unfold loopΘ
  rw [sub_sub_cancel]
  field_simp

private lemma loopW₀_inv (Q A iδ : ℝ) :
    loopW₀ Q A (1 / iδ) = Real.exp 1 * Real.sqrt (1 / iδ / (8 * (Q + 1)))
      * (16 / 5 * (Q * ((Q + 2) * A + 1) + 2 * A + 1)
        * Real.log (8 * (Q * ((Q + 2) * A + 1) + 2 * A + 1)
          * (Q * ((Q + 2) * A + 1) + 2 * A + 1 + 1) * (8 * (Q + 1) * iδ))
        + 3 + (Q + 2) * A * (Q + 1)) / 2 + 1 := by
  unfold loopW₀
  dsimp only
  congr 5
  field_simp


private lemma loopPool_inv (ℓ Q A iδ ic iξ ip is : ℝ) :
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

private lemma loopCount_inv (ℓ Q A iδ ic iξ ip is : ℝ) :
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

private lemma loopBudget_inv (ℓ Q A iδ ic : ℝ) :
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


private lemma loopΘ_le {Q A ic iξ is : ℝ} (hQ : 1 ≤ Q) (hA : 1 ≤ A) (hic : 1 ≤ ic) (hiξ : 1 ≤ iξ)
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

private lemma loopΘ_le_pow {Q A ic iξ is X : ℝ} (hQ : 1 ≤ Q) (hA : 1 ≤ A) (hic : 1 ≤ ic)
    (hiξ : 1 ≤ iξ) (his : 1 ≤ is) (hQX : Q ≤ X) (hAX : A ≤ X) (hcX : ic ≤ X) (hξX : iξ ≤ X)
    (hsX : is ≤ X) :
    loopΘ Q A (1 / ic) (1 / iξ) (1 / 2 - 1 / is) ≤ 2 ^ 19 * X ^ 4 := by
  have hX : 1 ≤ X := hQ.trans hQX
  refine (loopΘ_le hQ hA hic hiξ his).2.trans ?_
  have h : (A * ic + iξ) * is ^ 2 + Q ^ 2 * A * ic ≤ (X * X + X) * X ^ 2 + X ^ 2 * X * X := by
    gcongr
  have h3 : X ^ 3 ≤ X ^ 4 := pow_le_pow_right₀ hX (by norm_num)
  nlinarith

private lemma loopW₀_le {Q A iδ X : ℝ} (hQ : 1 ≤ Q) (hA : 1 ≤ A) (hiδ : 1 ≤ iδ) (hQX : Q ≤ X)
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


private lemma loopPool_ge {ℓ Q A iδ ic iξ ip is : ℝ} (hℓ : 1 ≤ ℓ) (hQ : 1 ≤ Q) (hA : 1 ≤ A)
    (hiδ : 1 ≤ iδ) (hic : 1 ≤ ic) (hiξ : 1 ≤ iξ) (hip : 1 ≤ ip) (his : 1 ≤ is) :
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

private lemma loopPool_le_pow {ℓ Q A iδ ic iξ ip is X : ℝ} (hℓ : 1 ≤ ℓ) (hQ : 1 ≤ Q) (hA : 1 ≤ A)
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


private lemma loopPool_le_log {ℓ Q A iδ ic iξ ip is X : ℝ} (hℓ : 1 ≤ ℓ) (hQ : 1 ≤ Q) (hA : 1 ≤ A)
    (hiδ : 2 ≤ iδ) (hic : 1 ≤ ic) (hiξ : 1 ≤ iξ) (hip : 1 ≤ ip) (his : 1 ≤ is) (hX : 2 ≤ X)
    (hℓX : ℓ ≤ 4 * X) (hQX : Q ≤ X) (hAX : A ≤ X) (hδX : iδ ≤ X) (hcX : ic ≤ X) (hξX : iξ ≤ X)
    (hsX : is ≤ X) :
    loopPool ℓ Q A (1 / iδ) (1 / ic) (1 / iξ) (1 / ip) (1 / 2 - 1 / is)
      ≤ 2400 * log X * ip * is ^ 2 + 640 * log (Q * iδ) * ip ^ 2 := by
  rw [loopPool_inv]
  set Θ := loopΘ Q A (1 / ic) (1 / iξ) (1 / 2 - 1 / is)
  set W₀ := loopW₀ Q A (1 / iδ)
  have hQ1 : 1 ≤ Q := hQ
  have hδ1 : 1 ≤ iδ := by linarith
  have hΘ1 := (loopΘ_le hQ1 hA hic hiξ his).1
  have hΘX := loopΘ_le_pow hQ1 hA hic hiξ his hQX hAX hcX hξX hsX
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
  have hc : log (16 * (Q + 4) * (Q + 1) * iδ) ≤ 10 * log (Q * iδ) := by
    have hb : 16 * (Q + 4) * (Q + 1) * iδ ≤ 2 ^ 8 * (Q * iδ) ^ 2 := by
      have e1 : 16 * (Q + 4) * (Q + 1) ≤ 160 * Q ^ 2 := by nlinarith
      have e2 : 16 * (Q + 4) * (Q + 1) * iδ ≤ 160 * Q ^ 2 * iδ :=
        mul_le_mul_of_nonneg_right e1 (by linarith)
      have e3 : 160 * Q ^ 2 * iδ ≤ 160 * Q ^ 2 * iδ ^ 2 :=
        mul_le_mul_of_nonneg_left (by nlinarith) (by positivity)
      nlinarith
    have := log_le_mul_log_of_le_pow (8 + 2) (by positivity) (hb.trans (by rw [pow_add]; gcongr))
    push_cast at this
    linarith
  have hip0 : 0 ≤ ip := by linarith
  have := mul_le_mul_of_nonneg_right hb hip0
  have := mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left hc (by norm_num : (0 : ℝ) ≤ 64))
    (sq_nonneg ip)
  nlinarith


set_option maxHeartbeats 1000000 in -- the bounds chain through many `gcongr` steps
private lemma claim_pref_le {L Q A iδ ic iξ ip is : ℝ} (hL : 0 ≤ L) (hQ : 1 ≤ Q) (hA : 2 ≤ A)
    (hiδ : 2 ≤ iδ) (hic : 1 ≤ ic) (hiξ : 1 ≤ iξ) (hip : 1 ≤ ip) (his : 2 ≤ is) :
    0 ≤ (Q + 1) * (6 * (Q + 2)
        * loopCount (L + Q + 4) Q A (1 / iδ) (1 / ic) (1 / iξ) (1 / ip) (1 / 2 - 1 / is))
      ∧ (Q + 1) * (6 * (Q + 2)
        * loopCount (L + Q + 4) Q A (1 / iδ) (1 / ic) (1 / iξ) (1 / ip) (1 / 2 - 1 / is))
        ≤ 2 ^ 74 * (Q ^ 4 * (L + Q) ^ 2 * log ((L + Q) * A * iδ * ic * iξ * ip * is)
          * ((A ^ 2 * ic ^ 2 + iξ ^ 2) * is ^ 8 + Q ^ 4 * A ^ 2 * ic ^ 2 * is ^ 4)) := by
  have hA1 : 1 ≤ A := by linarith
  have hδ1 : 1 ≤ iδ := by linarith
  have hc1 : 1 ≤ ic := by linarith
  have hξ1 : 1 ≤ iξ := by linarith
  have hp1 : 1 ≤ ip := by linarith
  have hs1 : 1 ≤ is := by linarith
  have hQ1 : 1 ≤ Q := hQ
  have hLQ : 1 ≤ L + Q := by linarith
  have hl : ∀ x ∈ [L + Q, A, iδ, ic, iξ, ip, is], 1 ≤ x := by
    simp [hA1, hδ1, hc1, hξ1, hp1, hs1, hLQ]
  have hsub (l : List ℝ) (hs : l.Sublist [L + Q, A, iδ, ic, iξ, ip, is]) :
      l.prod ≤ (L + Q) * A * iδ * ic * iξ * ip * is := by
    simpa [mul_assoc] using prod_le_prod_of_sublist hs hl
  have hLQA := hsub [L + Q, A] (by simp)
  have h8 := hsub [A, iδ, is] (by simp)
  set X := (L + Q) * A * iδ * ic * iξ * ip * is with hXd
  have hLQX : 2 * (L + Q) ≤ X := by simp only [List.prod_cons, List.prod_nil] at hLQA; nlinarith
  have hX8 : 8 ≤ X := by
    simp only [List.prod_cons, List.prod_nil, mul_one] at h8
    have : 4 ≤ A * iδ := by nlinarith
    nlinarith
  have hX : 2 ≤ X := by linarith
  have hQX : Q ≤ X := by linarith
  have hAX : A ≤ X := by simpa using hsub [A] (by simp)
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
  set Θ := loopΘ Q A (1 / ic) (1 / iξ) (1 / 2 - 1 / is)
  set PB := loopPool ℓ Q A (1 / iδ) (1 / ic) (1 / iξ) (1 / ip) (1 / 2 - 1 / is)
  obtain ⟨hΘ1, hΘu⟩ := loopΘ_le hQ1 hA1 hc1 hξ1 hs1
  have hΘX := loopΘ_le_pow hQ1 hA1 hc1 hξ1 hs1 hQX hAX hcX hξX hsX
  have hPB1 : 1 ≤ PB := loopPool_ge hℓ1 hQ1 hA1 hδ1 hc1 hξ1 hp1 hs1
  have hPBX := loopPool_le_pow hℓ1 hQ1 hA1 hδ1 hc1 hξ1 hp1 hs1 hℓX hQX hAX hδX hcX hξX hpX hsX
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
    have a2 : Q + 2 ≤ 3 * Q := by linarith only [hQ]
    have h1 : (Q + 1) * (Q + 2) ^ 3 * ℓ ^ 2 * log Y
        ≤ (2 * Q) * (3 * Q) ^ 3 * (5 * (L + Q)) ^ 2 * (95 * log X) := by
      gcongr
    have hG0 : 0 ≤ (A ^ 2 * ic ^ 2 + iξ ^ 2) * is ^ 8 + Q ^ 4 * A ^ 2 * ic ^ 2 * is ^ 4 := by
      positivity
    have hq0 : 0 ≤ (Q + 1) * (Q + 2) ^ 3 * ℓ ^ 2 * log Y := by positivity
    calc 966000 * ((Q + 1) * (Q + 2) ^ 3 * ℓ ^ 2 * log Y) * (Θ ^ 2 * is ^ 4)
        ≤ 966000 * ((2 * Q) * (3 * Q) ^ 3 * (5 * (L + Q)) ^ 2 * (95 * log X))
          * (2 ^ 36 * ((A ^ 2 * ic ^ 2 + iξ ^ 2) * is ^ 8 + Q ^ 4 * A ^ 2 * ic ^ 2 * is ^ 4)) := by
          gcongr
      _ = (966000 * 2 * 27 * 25 * 95 * 2 ^ 36) * (Q ^ 4 * (L + Q) ^ 2 * log X
          * ((A ^ 2 * ic ^ 2 + iξ ^ 2) * is ^ 8 + Q ^ 4 * A ^ 2 * ic ^ 2 * is ^ 4)) := by ring
      _ ≤ _ := by gcongr; norm_num


private lemma claim_suff_le {L Q A iδ ic iξ ip is : ℝ} (hL : 0 ≤ L) (hQ : 1 ≤ Q) (hA : 2 ≤ A)
    (hiδ : 2 ≤ iδ) (hic : 1 ≤ ic) (hiξ : 1 ≤ iξ) (hip : 1 ≤ ip) (his : 2 ≤ is) :
    0 ≤ (Q + 1)
        * (loopPool (L + Q + 4) Q A (1 / iδ) (1 / ic) (1 / iξ) (1 / ip) (1 / 2 - 1 / is) + 1)
      ∧ (Q + 1)
        * (loopPool (L + Q + 4) Q A (1 / iδ) (1 / ic) (1 / iξ) (1 / ip) (1 / 2 - 1 / is) + 1)
        ≤ 2 ^ 13 * (Q * (log ((L + Q) * A * iδ * ic * iξ * is) * ip * is ^ 2
          + log (Q * iδ) * ip ^ 2)) := by
  have hA1 : 1 ≤ A := by linarith
  have hδ1 : 1 ≤ iδ := by linarith
  have hc1 : 1 ≤ ic := by linarith
  have hξ1 : 1 ≤ iξ := by linarith
  have hp1 : 1 ≤ ip := by linarith
  have hs1 : 1 ≤ is := by linarith
  have hQ1 : 1 ≤ Q := hQ
  have hLQ : 1 ≤ L + Q := by linarith
  have hl : ∀ x ∈ [L + Q, A, iδ, ic, iξ, is], 1 ≤ x := by
    simp [hA1, hδ1, hc1, hξ1, hs1, hLQ]
  have hsub (l : List ℝ) (hs : l.Sublist [L + Q, A, iδ, ic, iξ, is]) :
      l.prod ≤ (L + Q) * A * iδ * ic * iξ * is := by
    simpa [mul_assoc] using prod_le_prod_of_sublist hs hl
  have hLQA := hsub [L + Q, A] (by simp)
  have h8 := hsub [A, iδ, is] (by simp)
  set X := (L + Q) * A * iδ * ic * iξ * is with hXd
  have hLQX : 2 * (L + Q) ≤ X := by simp only [List.prod_cons, List.prod_nil] at hLQA; nlinarith
  have hX8 : 8 ≤ X := by
    simp only [List.prod_cons, List.prod_nil, mul_one] at h8
    have : 4 ≤ A * iδ := by nlinarith
    nlinarith
  have hX : 2 ≤ X := by linarith
  have hQX : Q ≤ X := by linarith
  have hAX : A ≤ X := by simpa using hsub [A] (by simp)
  have hδX : iδ ≤ X := by simpa using hsub [iδ] (by simp)
  have hcX : ic ≤ X := by simpa using hsub [ic] (by simp)
  have hξX : iξ ≤ X := by simpa using hsub [iξ] (by simp)
  have hsX : is ≤ X := by simpa using hsub [is] (by simp)
  have hℓ1 : 1 ≤ L + Q + 4 := by linarith
  have hℓX : L + Q + 4 ≤ 4 * X := by linarith
  have hP := loopPool_le_log hℓ1 hQ hA1 hiδ hc1 hξ1 hp1 hs1 hX hℓX hQX hAX hδX hcX hξX hsX
  have hP1 := loopPool_ge hℓ1 hQ1 hA1 hδ1 hc1 hξ1 hp1 hs1
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
  have hc0 : 0 ≤ log (Q * iδ) * ip ^ 2 := by positivity
  constructor
  · positivity
  · have h2 : loopPool (L + Q + 4) Q A (1 / iδ) (1 / ic) (1 / iξ) (1 / ip) (1 / 2 - 1 / is) + 1
        ≤ 2402 * (log X * ip * is ^ 2) + 640 * (log (Q * iδ) * ip ^ 2) := by linarith
    have h3 : (Q + 1) ≤ 2 * Q := by linarith
    calc (Q + 1) * (loopPool (L + Q + 4) Q A (1 / iδ) (1 / ic) (1 / iξ) (1 / ip) (1 / 2 - 1 / is)
          + 1)
        ≤ (2 * Q) * (2402 * (log X * ip * is ^ 2) + 640 * (log (Q * iδ) * ip ^ 2)) := by
          gcongr
      _ ≤ _ := by nlinarith

private lemma claim_probes_le {L Q A iδ ic : ℝ} (hL : 0 ≤ L) (hQ : 1 ≤ Q) (hA : 2 ≤ A)
    (hiδ : 2 ≤ iδ) (hic : 1 ≤ ic) :
    0 ≤ (Q + 1) * loopBudget (L + Q + 4) Q A (1 / iδ) (1 / ic)
      ∧ (Q + 1) * loopBudget (L + Q + 4) Q A (1 / iδ) (1 / ic)
        ≤ 2 ^ 19 * (Q ^ 3 * A ^ 2 * (L + Q) * ic * log (Q * A * iδ)) := by
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
  · have hℓ : L + Q + 4 ≤ 5 * (L + Q) := by linarith
    calc (Q + 1) * ((36 * S * (4 * (A + 1) * ic) * log z + S + 1) * (L + Q + 4))
        ≤ (2 * Q) * ((40400 * (Q ^ 2 * A ^ 2 * ic * log Y)) * (5 * (L + Q))) := by
          gcongr <;> linarith
      _ = 404000 * (Q ^ 3 * A ^ 2 * (L + Q) * ic * log Y) := by ring
      _ ≤ _ := by
          have : 0 ≤ Q ^ 3 * A ^ 2 * (L + Q) * ic * log Y := by
            have : 0 ≤ log Y := by linarith
            have : 0 ≤ L + Q := by linarith
            positivity
          nlinarith

private lemma claim_heldOut_le {Q iδ iξ : ℝ} (hQ : 1 ≤ Q) (hiδ : 2 ≤ iδ) (hiξ : 1 ≤ iξ) :
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


/-- `LearnerLoopDFA`'s `L₀`, with `nQ` and `nA` standing for `|Q|` and `|Σ|` as exponents. -/
private def loopL0 (Q A : ℝ) (nQ nA : ℕ) (δ c ξ pAP η₀ : ℝ) : ℝ :=
  let δ' : ℝ := δ / (8 * (Q + 1))
  let s : ℝ := 1 / 2 - η₀
  let S : ℝ := Q * ((Q + 2) * A + 1) + 2 * A + 1
  let n : ℕ := ⌈32 * Real.log (16 * (Q + 1) * (Q + 2) / δ) / ξ ^ 2⌉₊
  let aε : ℝ := 4 * (A + 1) / c
  let Θ : ℝ := loopΘ Q A c ξ η₀
  let K₁ : ℝ := 36 * S * aε * Real.log (16 * S * (S + 1) / δ') + S + 1
  let W₀ : ℝ := loopW₀ Q A δ
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
private lemma claim_L0_le {Q A iδ ic iξ ip is : ℝ} (hQ : 1 ≤ Q) (hA : 2 ≤ A) (hiδ : 2 ≤ iδ)
    (hic : 1 ≤ ic) (hiξ : 1 ≤ iξ) (hip : 1 ≤ ip) (his : 2 ≤ is) :
    |loopL0 Q A ⌊Q⌋₊ ⌊A⌋₊ (1 / iδ) (1 / ic) (1 / iξ) (1 / ip) (1 / 2 - 1 / is)|
      ≤ 1000 * |log (Q * A * iδ * ic * iξ * ip * is) / log A| := by
  unfold loopL0
  extract_lets δ' s S n aε Θ K₁ W₀ P₀ R₀ E₁ Kc E E₂ E₃ T
  have hΘe : Θ = loopΘ Q A (1 / ic) (1 / iξ) (1 / 2 - 1 / is) := rfl
  have hW₀e : W₀ = loopW₀ Q A (1 / iδ) := rfl
  have hP₀e : P₀ = loopPool (exp 1) Q A (1 / iδ) (1 / ic) (1 / iξ) (1 / ip) (1 / 2 - 1 / is) := by
    unfold loopPool
    rw [Real.log_exp, mul_one]
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
  clear_value δ' s S n aε Θ K₁ W₀ P₀ R₀ E₁ Kc E E₂ E₃ T
  have hQ1 : 1 ≤ Q := by linarith
  have hA1 : 1 ≤ A := by linarith
  have hδ1 : 1 ≤ iδ := by linarith
  have hc1 : 1 ≤ ic := by linarith
  have hξ1 : 1 ≤ iξ := by linarith
  have hp1 : 1 ≤ ip := by linarith
  have hs1 : 1 ≤ is := by linarith
  have hl : ∀ x ∈ [Q, A, iδ, ic, iξ, ip, is], 1 ≤ x := by
    simp [hQ1, hA1, hδ1, hc1, hξ1, hp1, hs1]
  have hsub (l : List ℝ) (hs : l.Sublist [Q, A, iδ, ic, iξ, ip, is]) :
      l.prod ≤ Q * A * iδ * ic * iξ * ip * is := by
    simpa [mul_assoc] using prod_le_prod_of_sublist hs hl
  have h8 := hsub [A, iδ, is] (by simp)
  set X := Q * A * iδ * ic * iξ * ip * is with hXd
  have hQX : Q ≤ X := by simpa using hsub [Q] (by simp)
  have hAX : A ≤ X := by simpa using hsub [A] (by simp)
  have hδX : iδ ≤ X := by simpa using hsub [iδ] (by simp)
  have hcX : ic ≤ X := by simpa using hsub [ic] (by simp)
  have hξX : iξ ≤ X := by simpa using hsub [iξ] (by simp)
  have hpX : ip ≤ X := by simpa using hsub [ip] (by simp)
  have hsX : is ≤ X := by simpa using hsub [is] (by simp)
  clear_value X
  have hX : 8 ≤ X := by
    simp only [List.prod_cons, List.prod_nil, mul_one] at h8
    have : 4 ≤ A * iδ := by nlinarith
    nlinarith
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
    have h1 : 1 ≤ 16 * (Q + 4) * P₀ * Θ :=
      one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le (by linarith) hP₀1) hΘ1
    exact one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le h1 (one_le_pow₀ hD1)) hp1
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
    have h0 : 1 ≤ 2 ^ 20 * Θ := one_le_mul_of_one_le_of_one_le (by norm_num) hΘ1
    have h1 : 1 ≤ 2 ^ 20 * Θ * (R₀ + 1) * is :=
      one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le h0 (by linarith)) hs1
    have h2 : 0 ≤ 2 ^ 36 * Θ ^ 2 * (13 + 2 * Real.logb 2 (R₀ + 1)) * D := by positivity
    have h3 : 0 ≤ 2 ^ 16 * (P₀ + 2) * ip * D := by positivity
    exact one_le_mul_of_one_le_of_one_le (one_le_pow₀ (by linarith)) (by linarith)
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
    exact one_le_mul_of_one_le_of_one_le (one_le_pow₀ (le_add_of_nonneg_right (by positivity)))
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
    exact Real.log_nonneg 
      (one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le (by linarith) hKc1) hD1)
  have hE1 : 1 ≤ E := by
    rw [hE]
    have : 1 ≤ 1536 * aε * E₁ :=
      one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le (by norm_num) haε1) hE₁1
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
      Real.log_nonneg (one_le_mul_of_one_le_of_one_le (by nlinarith only [hQ1]) hδ1)
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
    exact Real.log_nonneg (one_le_mul_of_one_le_of_one_le (by nlinarith only [hS1]) hD1)
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
  have hX2 : 2 ≤ X := by linarith
  have hlEs : log Es ≤ 282 * log X := by
    have := log_le_mul_log_of_le_pow (228 + 54) (by linarith)
      (hEsX.trans (by rw [pow_add]; gcongr))
    push_cast at this
    linarith
  have hlEs0 : 0 ≤ log Es := Real.log_nonneg hEs1
  have hlA : log 2 ≤ log A := log_two_le_log hA
  have hl2 := Real.log_two_gt_d9
  have hlA0 : 0 < log A := by linarith
  have hlX0 : 0 < log X := Real.log_pos (by linarith)
  have hlXX : log X ≤ X := Real.log_le_self (by linarith)
  have hT5 : 5 ≤ T := by
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
    have := log_le_mul_log_of_le_pow (10 + 1) (by linarith)
      (hTX.trans (by rw [pow_add, pow_one]; gcongr))
    push_cast at this
    linarith
  have hlT0 : 0 ≤ log T := Real.log_nonneg (by linarith)
  have hTA : 12 ≤ T * log A - 12 := by
    have e : T * log A = 2 * (log Es + 12) + (Q + 4) * log A := by
      rw [hT]; field_simp; ring
    have : 0 ≤ (Q + 4) * log A := by positivity
    linarith
  clear_value Es
  -- the threshold
  have hden : 0 < log A * (T * log A - 12) := by positivity
  have hlX2 : 2 ≤ log X := by
    rw [Real.le_log_iff_exp_le (by linarith)]
    have e2 : Real.exp 2 = Real.exp 1 * Real.exp 1 := by rw [← Real.exp_add]; norm_num
    have := Real.exp_one_lt_d9
    nlinarith only [e2, this, hex, hX]
  have h1 : 2 * (log Es + 6 * log T) / log A ≤ 696 * (log X / log A) := by
    rw [mul_div_assoc']
    exact div_le_div_of_nonneg_right (by linarith) hlA0.le
  have h2 : |144 * (log T - 2) / (log A * (T * log A - 12))| ≤ 144 * (log X / log A) := by
    rw [abs_div, abs_of_pos hden, div_le_iff₀ hden]
    have e : 144 * (log X / log A) * (log A * (T * log A - 12))
        = 144 * log X * (T * log A - 12) := by
      field_simp
    rw [e]
    have ha : |144 * (log T - 2)| ≤ 144 * (log T + 2) := by
      rw [abs_le]; constructor <;> linarith
    linarith only [ha, hlT, hlX2, mul_le_mul_of_nonneg_left hTA hlX0.le]
  have h0a : 0 ≤ 2 * (log Es + 6 * log T) / log A := by positivity
  have hq : 0 ≤ log X / log A := by positivity
  rw [abs_of_nonneg hq]
  calc |2 * (log Es + 6 * log T) / log A + 144 * (log T - 2) / (log A * (T * log A - 12))|
      ≤ |2 * (log Es + 6 * log T) / log A| + |144 * (log T - 2) / (log A * (T * log A - 12))| :=
        abs_add_le _ _
    _ ≤ 1000 * (log X / log A) := by rw [abs_of_nonneg h0a]; linarith

private def claimL0 : LoopVec → ℝ
  | (Q, A, iδ, ic, iξ, ip, is) =>
    loopL0 Q A ⌊Q⌋₊ ⌊A⌋₊ (1 / iδ) (1 / ic) (1 / iξ) (1 / ip) (1 / 2 - 1 / is)

private def claimPref : ℝ × LoopVec → ℝ
  | (L, Q, A, iδ, ic, iξ, ip, is) =>
    (Q + 1) * (6 * (Q + 2)
      * loopCount (L + Q + 4) Q A (1 / iδ) (1 / ic) (1 / iξ) (1 / ip) (1 / 2 - 1 / is))

private def claimSuff : ℝ × LoopVec → ℝ
  | (L, Q, A, iδ, ic, iξ, ip, is) =>
    (Q + 1) * (loopPool (L + Q + 4) Q A (1 / iδ) (1 / ic) (1 / iξ) (1 / ip) (1 / 2 - 1 / is) + 1)

private def claimProbes : ℝ × LoopVec → ℝ
  | (L, Q, A, iδ, ic, _, _, _) => (Q + 1) * loopBudget (L + Q + 4) Q A (1 / iδ) (1 / ic)

private def claimHeldOut : LoopVec → ℕ
  | (Q, _, iδ, _, iξ, _, _) => ⌈32 * log (16 * (Q + 1) * (Q + 2) / (1 / iδ)) / (1 / iξ) ^ 2⌉₊

private lemma loop_region {Q A iδ ic iξ ip is : ℝ} (hb : (Q, A, iδ, ic, iξ, ip, is) ∈ vecRegion) :
    1 ≤ Q ∧ 2 ≤ A ∧ 2 ≤ iδ ∧ 1 ≤ ic ∧ 1 ≤ iξ ∧ 1 ≤ ip ∧ 2 ≤ is := by
  simpa only [vecRegion, Set.mem_Ici, Prod.mk_le_mk] using hb

private lemma loop_region' {L Q A iδ ic iξ ip is : ℝ}
    (hb : (L, Q, A, iδ, ic, iξ, ip, is) ∈ Set.Ici 0 ×ˢ vecRegion) :
    0 ≤ L ∧ 1 ≤ Q ∧ 2 ≤ A ∧ 2 ≤ iδ ∧ 1 ≤ ic ∧ 1 ≤ iξ ∧ 1 ≤ ip ∧ 2 ≤ is := by
  rw [Set.mem_prod] at hb
  exact ⟨hb.1, loop_region hb.2⟩

private lemma claimL0_isBigO : claimL0 =O[𝓟 vecRegion]
    (fun (Q, A, iδ, ic, iξ, ip, is) => 1 + log (Q * iδ * ic * iξ * ip * is) / log A
      : LoopVec → ℝ) := by
  refine isBigO_principal.2 ⟨1000, fun b hb => ?_⟩
  obtain ⟨Q, A, iδ, ic, iξ, ip, is⟩ := b
  obtain ⟨hQ, hA, hiδ, hic, hiξ, hip, his⟩ := loop_region hb
  have hlA : 0 < log A := Real.log_pos (by linarith)
  have hX : 0 < Q * iδ * ic * iξ * ip * is := by
    have : 0 < Q := by linarith
    have : 0 < iδ := by linarith
    have : 0 < ic := by linarith
    have : 0 < iξ := by linarith
    have : 0 < ip := by linarith
    have : 0 < is := by linarith
    positivity
  have e : log (Q * A * iδ * ic * iξ * ip * is) / log A
      = 1 + log (Q * iδ * ic * iξ * ip * is) / log A := by
    rw [show Q * A * iδ * ic * iξ * ip * is = A * (Q * iδ * ic * iξ * ip * is) by ring,
      Real.log_mul (by linarith : (0 : ℝ) < A).ne' hX.ne']
    field_simp [hlA.ne']
  have h := claim_L0_le hQ hA hiδ hic hiξ hip his
  rw [e] at h
  simpa only [Real.norm_eq_abs, claimL0] using h

private lemma claimPref_isBigO : claimPref =O[𝓟 (Set.Ici 0 ×ˢ vecRegion)]
    (fun (L, Q, A, iδ, ic, iξ, ip, is) =>
      Q ^ 4 * (L + Q) ^ 2 * log ((L + Q) * A * iδ * ic * iξ * ip * is)
        * ((A ^ 2 * ic ^ 2 + iξ ^ 2) * is ^ 8 + Q ^ 4 * A ^ 2 * ic ^ 2 * is ^ 4)
      : ℝ × LoopVec → ℝ) := by
  refine isBigO_on (2 ^ 74) fun b hb => ?_
  obtain ⟨L, Q, A, iδ, ic, iξ, ip, is⟩ := b
  obtain ⟨hL, hQ, hA, hiδ, hic, hiξ, hip, his⟩ := loop_region' hb
  obtain ⟨h0, h⟩ := claim_pref_le hL hQ hA hiδ hic hiξ hip his
  exact ⟨h0, nonneg_of_mul_nonneg_right (h0.trans h) (by positivity), h⟩

private lemma claimSuff_isBigO : claimSuff =O[𝓟 (Set.Ici 0 ×ˢ vecRegion)]
    (fun (L, Q, A, iδ, ic, iξ, ip, is) =>
      Q * (log ((L + Q) * A * iδ * ic * iξ * is) * ip * is ^ 2 + log (Q * iδ) * ip ^ 2)
      : ℝ × LoopVec → ℝ) := by
  refine isBigO_on (2 ^ 13) fun b hb => ?_
  obtain ⟨L, Q, A, iδ, ic, iξ, ip, is⟩ := b
  obtain ⟨hL, hQ, hA, hiδ, hic, hiξ, hip, his⟩ := loop_region' hb
  obtain ⟨h0, h⟩ := claim_suff_le hL hQ hA hiδ hic hiξ hip his
  exact ⟨h0, nonneg_of_mul_nonneg_right (h0.trans h) (by positivity), h⟩

private lemma claimProbes_isBigO : claimProbes =O[𝓟 (Set.Ici 0 ×ˢ vecRegion)]
    (fun (L, Q, A, iδ, ic, _, _, _) => Q ^ 3 * A ^ 2 * (L + Q) * ic * log (Q * A * iδ)
      : ℝ × LoopVec → ℝ) := by
  refine isBigO_on (2 ^ 19) fun b hb => ?_
  obtain ⟨L, Q, A, iδ, ic, iξ, ip, is⟩ := b
  obtain ⟨hL, hQ, hA, hiδ, hic, -, -, -⟩ := loop_region' hb
  obtain ⟨h0, h⟩ := claim_probes_le hL hQ hA hiδ hic
  exact ⟨h0, nonneg_of_mul_nonneg_right (h0.trans h) (by positivity), h⟩

private lemma claimHeldOut_isBigO : (fun v => (claimHeldOut v : ℝ)) =O[𝓟 vecRegion]
    (fun (Q, _, iδ, _, iξ, _, _) => log (Q * iδ) * iξ ^ 2 : LoopVec → ℝ) := by
  refine isBigO_on (2 ^ 9) fun b hb => ?_
  obtain ⟨Q, A, iδ, ic, iξ, ip, is⟩ := b
  obtain ⟨hQ, -, hiδ, -, hiξ, -, -⟩ := loop_region hb
  have h := claim_heldOut_le hQ hiδ hiξ
  exact ⟨Nat.cast_nonneg _,
    nonneg_of_mul_nonneg_right ((Nat.cast_nonneg _).trans h) (by positivity), h⟩

private lemma pref_dom : (fun (L, Q, A, iδ, ic, iξ, ip, is) =>
    Q ^ 4 * (L + Q) ^ 2 * log ((L + Q) * A * iδ * ic * iξ * ip * is)
      * ((A ^ 2 * ic ^ 2 + iξ ^ 2) * is ^ 8 + Q ^ 4 * A ^ 2 * ic ^ 2 * is ^ 4)
    : ℝ × LoopVec → ℝ)
    =O[𝓟 (Set.Ici (0 : ℝ) ×ˢ vecRegion)] (fun (L, Q, A, iδ, ic, iξ, ip, is) =>
      Q ^ 8 * (L + Q) ^ 2 * A ^ 2 * log ((L + Q) * A * iδ * ic * iξ * ip * is)
        * ic ^ 2 * iξ ^ 2 * is ^ 8 : ℝ × LoopVec → ℝ) := by
  refine isBigO_on 3 fun b hb => ?_
  obtain ⟨L, Q, A, iδ, ic, iξ, ip, is⟩ := b
  obtain ⟨hL, hQ, hA, hiδ, hic, hiξ, hip, his⟩ := loop_region' hb
  have hY : 1 ≤ (L + Q) * A * iδ * ic * iξ * ip * is := by
    refine one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le
      (one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le
        (one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le ?_ ?_) ?_) ?_) ?_) ?_) ?_
      <;> linarith
  have hl : 0 ≤ log ((L + Q) * A * iδ * ic * iξ * ip * is) := Real.log_nonneg hY
  have hQ4 : 1 ≤ Q ^ 4 := one_le_pow₀ hQ
  have hA2 : 1 ≤ A ^ 2 := one_le_pow₀ (by linarith)
  have hc2 : 1 ≤ ic ^ 2 := one_le_pow₀ hic
  have hξ2 : 1 ≤ iξ ^ 2 := one_le_pow₀ hiξ
  have hs4 : 1 ≤ is ^ 4 := one_le_pow₀ (by linarith)
  have hs8 : 0 ≤ is ^ 8 := by positivity
  set M := Q ^ 4 * A ^ 2 * ic ^ 2 * iξ ^ 2 * is ^ 8
  have m1 : A ^ 2 * ic ^ 2 * is ^ 8 ≤ M := (le_mul_of_one_le_left (by positivity)
    (one_le_mul_of_one_le_of_one_le hQ4 hξ2)).trans_eq (by ring)
  have m2 : iξ ^ 2 * is ^ 8 ≤ M := (le_mul_of_one_le_left (by positivity)
    (one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le hQ4 hA2) hc2)).trans_eq
      (by ring)
  have m3 : Q ^ 4 * A ^ 2 * ic ^ 2 * is ^ 4 ≤ M := (le_mul_of_one_le_left (by positivity)
    (one_le_mul_of_one_le_of_one_le hξ2 hs4)).trans_eq (by ring)
  have hK : 0 ≤ Q ^ 4 * (L + Q) ^ 2 * log ((L + Q) * A * iδ * ic * iξ * ip * is) := by
    positivity
  have h := mul_le_mul_of_nonneg_left (show (A ^ 2 * ic ^ 2 + iξ ^ 2) * is ^ 8
    + Q ^ 4 * A ^ 2 * ic ^ 2 * is ^ 4 ≤ 3 * M by linarith) hK
  dsimp only
  refine ⟨mul_nonneg hK (by positivity), by positivity, h.trans_eq (by ring)⟩

private lemma suff_dom : (fun (L, Q, A, iδ, ic, iξ, ip, is) =>
    Q * (log ((L + Q) * A * iδ * ic * iξ * is) * ip * is ^ 2 + log (Q * iδ) * ip ^ 2)
    : ℝ × LoopVec → ℝ)
    =O[𝓟 (Set.Ici (0 : ℝ) ×ˢ vecRegion)] (fun (L, Q, A, iδ, ic, iξ, ip, is) =>
      Q * log ((L + Q) * A * iδ * ic * iξ * ip * is) * ip ^ 2 * is ^ 2
      : ℝ × LoopVec → ℝ) := by
  refine isBigO_on 2 fun b hb => ?_
  obtain ⟨L, Q, A, iδ, ic, iξ, ip, is⟩ := b
  obtain ⟨hL, hQ, hA, hiδ, hic, hiξ, hip, his⟩ := loop_region' hb
  have hY' : 1 ≤ (L + Q) * A * iδ * ic * iξ * is := by
    refine one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le
      (one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le
        (one_le_mul_of_one_le_of_one_le ?_ ?_) ?_) ?_) ?_) ?_ <;> linarith
  have hYY : (L + Q) * A * iδ * ic * iξ * is ≤ (L + Q) * A * iδ * ic * iξ * ip * is :=
    (le_mul_of_one_le_right (by linarith) hip).trans_eq (by ring)
  have hQδ : 1 ≤ Q * iδ := one_le_mul_of_one_le_of_one_le hQ (by linarith)
  have hQY : Q * iδ ≤ (L + Q) * A * iδ * ic * iξ * ip * is := by
    have h1 : Q * iδ ≤ (L + Q) * iδ := by gcongr; linarith
    have h2 : 1 ≤ A * ic * iξ * ip * is := by
      refine one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le
        (one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le ?_ ?_) ?_) ?_) ?_
        <;> linarith
    exact h1.trans ((le_mul_of_one_le_right (by nlinarith) h2).trans_eq (by ring))
  have l1 := Real.log_le_log (by linarith) hYY
  have l2 := Real.log_le_log (by linarith) hQY
  have l0 : 0 ≤ log ((L + Q) * A * iδ * ic * iξ * is) := Real.log_nonneg hY'
  have l0' : 0 ≤ log (Q * iδ) := Real.log_nonneg hQδ
  set ℓ := log ((L + Q) * A * iδ * ic * iξ * ip * is)
  have hℓ : 0 ≤ ℓ := l0.trans l1
  have hp2 : ip ≤ ip ^ 2 := by nlinarith
  have hs2 : 1 ≤ is ^ 2 := one_le_pow₀ (by linarith)
  have e1 : log ((L + Q) * A * iδ * ic * iξ * is) * ip * is ^ 2 ≤ ℓ * ip ^ 2 * is ^ 2 := by
    gcongr
  have e2 : log (Q * iδ) * ip ^ 2 ≤ ℓ * ip ^ 2 * is ^ 2 :=
    calc log (Q * iδ) * ip ^ 2 ≤ ℓ * ip ^ 2 := by gcongr
      _ ≤ ℓ * ip ^ 2 * is ^ 2 := le_mul_of_one_le_right (mul_nonneg hℓ (by positivity)) hs2
  have h := mul_le_mul_of_nonneg_left (add_le_add e1 e2) (by linarith : (0 : ℝ) ≤ Q)
  dsimp only
  refine ⟨mul_nonneg (by linarith) (by positivity),
    mul_nonneg (mul_nonneg (mul_nonneg (by linarith) hℓ) (by positivity)) (by positivity),
    h.trans_eq (by ring)⟩

private def toVec (x : LoopInputs) : LoopVec :=
  (x.Q, x.A, 1 / x.δ, 1 / x.c, 1 / x.ξ, 1 / x.pAP, 1 / x.s₀)

private def toRunVec (r : LoopRun) : ℝ × LoopVec := (r.L, toVec r.toLoopInputs)

private lemma toVec_mem {x : LoopInputs} (hx : x ∈ loopRegion) : toVec x ∈ vecRegion := by
  obtain ⟨hQ, hA, hδ, hδ2, hc, hc1, hξ, hξ1, hp, hp1, hs, hs1⟩ := hx
  simp only [toVec, vecRegion, Set.mem_Ici, Prod.mk_le_mk]
  exact ⟨by exact_mod_cast hQ, by exact_mod_cast hA, (le_div_iff₀ hδ).2 (by linarith),
    (le_div_iff₀ hc).2 (by linarith), (le_div_iff₀ hξ).2 (by linarith),
    (le_div_iff₀ hp).2 (by linarith), (le_div_iff₀ hs).2 (by linarith)⟩

private lemma transfer {f g : LoopVec → ℝ} {g' : LoopInputs → ℝ} (h : f =O[𝓟 vecRegion] g)
    (hg : ∀ x, g (toVec x) = g' x) : (fun x => f (toVec x)) =O[𝓟 loopRegion] g' :=
  (isBigO_comp_on h fun _ hx => toVec_mem hx).congr_right hg

private lemma transferRun {f g : ℝ × LoopVec → ℝ} {g' : LoopRun → ℝ}
    (h : f =O[𝓟 (Set.Ici 0 ×ˢ vecRegion)] g) (hg : ∀ r, g (toRunVec r) = g' r) :
    (fun r => f (toRunVec r)) =O[𝓟 loopRunRegion] g' :=
  (isBigO_comp_on (φ := toRunVec) h fun r hr => Set.mk_mem_prod
    (Set.mem_Ici.2 (Nat.cast_nonneg r.L)) (toVec_mem hr)).congr_right hg

private lemma dominateRun {f g : ℝ × LoopVec → ℝ} {f' g' : LoopRun → ℝ}
    (h : f =O[𝓟 (Set.Ici 0 ×ˢ vecRegion)] g) (hf : ∀ r, f (toRunVec r) = f' r)
    (hg : ∀ r, g (toRunVec r) = g' r) : f' =O[𝓟 loopRunRegion] g' :=
  (transferRun h hg).congr_left hf

theorem loopClaim_holds : LoopClaim := by
  refine ⟨fun x => claimL0 (toVec x), fun r => claimPref (toRunVec r),
    fun r => claimSuff (toRunVec r), fun r => claimProbes (toRunVec r),
    fun x => claimHeldOut (toVec x),
    transfer claimL0_isBigO fun x => by simp only [toVec]; ring_nf,
    transferRun claimPref_isBigO fun r => by simp only [toRunVec, toVec]; ring_nf,
    transferRun claimSuff_isBigO fun r => by simp only [toRunVec, toVec]; ring_nf,
    transferRun claimProbes_isBigO fun r => by simp only [toRunVec, toVec]; ring_nf,
    transfer claimHeldOut_isBigO fun x => by simp only [toVec]; ring_nf, ?_⟩
  intro α _ _ _ σ _ Ω _ μ _ x M O L γ k L₁ hx hQ hA hL hηle hpAPu hceil hL₀
  obtain ⟨Q, A, δ, c, ξ, pAP, s₀⟩ := x
  simp only [loopRegion, Set.mem_ofPred_eq] at hx hQ hA hηle hpAPu hceil
  obtain ⟨-, hA2, hδ, hδ2, hc, hc1, hξ, hξ1, hpAP, hpAP1, hs, hs1⟩ := hx
  obtain ⟨η₀, rfl⟩ : ∃ η₀, s₀ = 1 / 2 - η₀ := ⟨1 / 2 - s₀, by ring⟩
  subst hQ hA
  rw [sub_sub_cancel] at hηle
  have hη₀ : η₀ < 1 / 2 := by linarith
  have hL₀' : loopL0 (Fintype.card σ) (Fintype.card α) (Fintype.card σ) (Fintype.card α) δ c ξ
      pAP η₀ ≤ L := by
    have e : claimL0 (toVec ⟨Fintype.card σ, Fintype.card α, δ, c, ξ, pAP, 1 / 2 - η₀⟩)
        = loopL0 (Fintype.card σ) (Fintype.card α) (Fintype.card σ) (Fintype.card α) δ c ξ
          pAP η₀ := by
      simp only [toVec, claimL0, Nat.floor_natCast, one_div_one_div, sub_sub_cancel]
    rw [← e]
    exact hL₀
  obtain ⟨settings, hw, -, -, hcnt, hprob⟩ := learner_loop_dfa_counts M O L η₀ c γ δ pAP ξ hL
    hηle hη₀ hc hδ (by linarith) hpAP hξ (by exact_mod_cast hA2) hpAPu hceil hL₀'
  refine ⟨settings, hw, fun draws => ?_, ?_⟩
  · obtain ⟨h1, h2, h3⟩ := hcnt draws
    refine ⟨h1.trans_eq ?_, h2.trans_eq ?_, h3.trans_eq ?_⟩ <;>
      simp only [toRunVec, toVec, claimPref, claimSuff, claimProbes, one_div_one_div,
        sub_sub_cancel]
  · have e : claimHeldOut (toVec ⟨Fintype.card σ, Fintype.card α, δ, c, ξ, pAP, 1 / 2 - η₀⟩)
        = ⌈32 * log (16 * ((Fintype.card σ : ℝ) + 1) * ((Fintype.card σ : ℝ) + 2) / δ)
          / ξ ^ 2⌉₊ := by
      simp only [toVec, claimHeldOut, one_div_one_div]
    beta_reduce
    rw [e]
    exact hprob

theorem loopClaimSimple_holds : LoopClaimSimple := by
  obtain ⟨L₀, pref, suff, probes, heldOut, hL₀, hp, hs, hpr, hh, hw⟩ := loopClaim_holds
  exact ⟨L₀, pref, suff, probes, heldOut, hL₀,
    hp.trans (dominateRun pref_dom (fun r => by simp only [toRunVec, toVec]; ring_nf)
      fun r => by simp only [toRunVec, toVec]; ring_nf),
    hs.trans (dominateRun suff_dom (fun r => by simp only [toRunVec, toVec]; ring_nf)
      fun r => by simp only [toRunVec, toVec]; ring_nf), hpr, hh, hw⟩

end OrthoDFA
