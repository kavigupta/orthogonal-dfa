import RLStar.Proofs.BigO

/-!
# The trie tolerance's arithmetic

`trie_loop_tol`'s steps over abstract reals: the tolerance

    T = ℓ⁴ Λ E

against the rates and caps `trie_loop_core` needs, once `2⁴ T⁵ ≤ |Σ|^k`.
-/

noncomputable section

namespace OrthoDFA

open Real

lemma tol_T_le {ℓ Λ E E₁ aε lg T : ℝ} (hℓ : 1 ≤ ℓ) (hΛ : 1 ≤ Λ) (haε : 1 ≤ aε)
    (hE₁ : 1 ≤ E₁) (hlg : 0 ≤ lg) (hE : E = 1536 * aε * E₁ + 1536 * lg * aε)
    (hT : T = ℓ ^ 4 * Λ * E) :
    1 ≤ T ∧ ℓ ^ 2 * Λ * E₁ ≤ T ∧ E₁ ≤ T ∧ ℓ ≤ ℓ ^ 2 * Λ ∧ ℓ ≤ T ∧ aε * ℓ ≤ T ∧ aε ≤ T := by
  have hΛ0 : 0 ≤ Λ := by linarith only [hΛ]
  have hℓ0 : 0 ≤ ℓ := by linarith only [hℓ]
  have ha0 : 0 ≤ aε := by linarith only [haε]
  have hE₁0 : 0 ≤ E₁ := by linarith only [hE₁]
  have hu0 : 0 ≤ ℓ ^ 4 * Λ := mul_nonneg (pow_nonneg hℓ0 4) hΛ0
  have hl2 : 1 ≤ ℓ ^ 2 * Λ := one_le_mul_of_one_le_of_one_le (one_le_pow₀ hℓ) hΛ
  have hl24 : ℓ ^ 2 * Λ ≤ ℓ ^ 4 * Λ :=
    mul_le_mul_of_nonneg_right (pow_le_pow_right₀ hℓ (by norm_num)) hΛ0
  have hu1 : 1 ≤ ℓ ^ 4 * Λ := hl2.trans hl24
  have hae1 : 1 ≤ aε * E₁ := one_le_mul_of_one_le_of_one_le haε hE₁
  have hTE : ℓ ^ 4 * Λ * (aε * E₁) ≤ T := by
    have hEaE : aε * E₁ ≤ E := by
      have h0 : 0 ≤ lg * aε := mul_nonneg hlg ha0
      rw [hE]; linarith only [h0, hae1]
    rw [hT]; exact mul_le_mul_of_nonneg_left hEaE hu0
  have hlℓ : ℓ ≤ ℓ ^ 2 * Λ := by
    have : ℓ ≤ ℓ ^ 2 := by nlinarith only [hℓ]
    exact this.trans (le_mul_of_one_le_right (sq_nonneg ℓ) hΛ)
  have hTlE : ℓ ^ 2 * Λ * E₁ ≤ T := by
    have h1 := mul_le_mul_of_nonneg_right hl24 hE₁0
    have h2 := mul_le_mul_of_nonneg_left (le_mul_of_one_le_left hE₁0 haε) hu0
    linarith only [h1, h2, hTE]
  have hTE₁ : E₁ ≤ T := (le_mul_of_one_le_left hE₁0 haε).trans
    ((le_mul_of_one_le_left (by linarith only [hae1]) hu1).trans hTE)
  have hTℓ : ℓ ≤ T := (hlℓ.trans hl24).trans
    ((le_mul_of_one_le_right (by linarith only [hu1]) hae1).trans hTE)
  have hTaℓ : aε * ℓ ≤ T := by
    have h1 : aε * ℓ ≤ aε * (ℓ ^ 4 * Λ) := mul_le_mul_of_nonneg_left (hlℓ.trans hl24) ha0
    have h2 := mul_le_mul_of_nonneg_left hE₁ (mul_nonneg hu0 ha0)
    linarith only [h1, h2, hTE]
  exact ⟨(one_le_mul_of_one_le_of_one_le hu1 hae1).trans hTE, hTlE, hTE₁, hlℓ, hTℓ, hTaℓ,
    (le_mul_of_one_le_right ha0 hℓ).trans hTaℓ⟩

/-- `K₁ ≤ aε R₀`: `R₀` carries `Θι² ≥ S²` and the log of `1/δ'`. -/
lemma tol_K₁ {Q S aε Θι δ' P₀ s R₀ K₁ : ℝ} (hQ : 0 ≤ Q) (hS : 1 ≤ S) (haε : 1 ≤ aε)
    (hSΘ : S ≤ Θι) (hδ'0 : 0 < δ') (hδ'1 : δ' ≤ 1) (hP₀ : 1 ≤ P₀) (hs0 : 0 < s)
    (hs1 : s ≤ 1 / 2) (hK₁ : K₁ = 36 * S * aε * Real.log (16 * S * (S + 1) / δ') + S + 1)
    (hR₀ : 2 ^ 20 * (Q + 2) * Θι ^ 2 * (Real.log (256 * P₀ / δ') + 1) / s ^ 4 ≤ R₀) :
    K₁ ≤ aε * R₀ := by
  set Lg := Real.log (256 * P₀ / δ') with hLgd
  have hS0 : 0 < S := by linarith only [hS]
  have hP0 : 0 < P₀ := by linarith only [hP₀]
  have hLg0 : 0 ≤ Lg := Real.log_nonneg (by rw [le_div_iff₀ hδ'0]; linarith only [hP₀, hδ'1])
  have hlog : Real.log (16 * S * (S + 1) / δ') ≤ 5 * S + Lg := by
    have e1 : Real.log (16 * S * (S + 1) / δ')
        = Real.log 16 + Real.log S + Real.log (S + 1) - Real.log δ' := by
      rw [Real.log_div (by positivity) hδ'0.ne', Real.log_mul (by positivity) (by positivity),
        Real.log_mul (by norm_num) hS0.ne']
    have e2 : Lg = Real.log 256 + Real.log P₀ - Real.log δ' := by
      rw [hLgd, Real.log_div (by positivity) hδ'0.ne', Real.log_mul (by norm_num) hP0.ne']
    have l16 : Real.log 16 < 3 := by
      have := Real.log_two_lt_d9
      have e : Real.log 16 = 4 * Real.log 2 := by
        rw [show (16 : ℝ) = 2 ^ 4 by norm_num, Real.log_pow]; norm_num
      linarith only [this, e]
    have l256 : 0 ≤ Real.log 256 := Real.log_nonneg (by norm_num)
    have lP : 0 ≤ Real.log P₀ := Real.log_nonneg hP₀
    have lS : Real.log S ≤ S - 1 := Real.log_le_sub_one_of_pos hS0
    have lS1 : Real.log (S + 1) ≤ S := by
      have := Real.log_le_sub_one_of_pos (by linarith only [hS] : (0 : ℝ) < S + 1)
      linarith only [this]
    linarith only [e1, e2, l16, l256, lP, lS, lS1, hS]
  have hX0 : 0 ≤ Θι ^ 2 * (Lg + 1) := by positivity
  have h1 : K₁ ≤ aε * (218 * (Θι ^ 2 * (Lg + 1))) := by
    have a : K₁ ≤ aε * (180 * S ^ 2 + 36 * S * Lg) + S + 1 := by
      rw [hK₁]
      have := mul_le_mul_of_nonneg_left hlog (by positivity : (0 : ℝ) ≤ 36 * S * aε)
      nlinarith only [this]
    have b : 180 * S ^ 2 + 36 * S * Lg + S + 1 ≤ 218 * (S ^ 2 * (Lg + 1)) := by
      nlinarith only [hS, hLg0, mul_nonneg (by linarith only [hS] : (0 : ℝ) ≤ S - 1) hLg0,
        mul_nonneg (mul_nonneg hS0.le (by linarith only [hS] : (0 : ℝ) ≤ S - 1)) hLg0]
    have c : S ^ 2 * (Lg + 1) ≤ Θι ^ 2 * (Lg + 1) :=
      mul_le_mul_of_nonneg_right (pow_le_pow_left₀ hS0.le hSΘ 2) (by linarith only [hLg0])
    have d : S + 1 ≤ aε * (S + 1) := le_mul_of_one_le_left (by linarith only [hS]) haε
    have bc : 180 * S ^ 2 + 36 * S * Lg + S + 1 ≤ 218 * (Θι ^ 2 * (Lg + 1)) := by
      linarith only [b, c]
    have := mul_le_mul_of_nonneg_left bc (by linarith only [haε] : (0 : ℝ) ≤ aε)
    linarith only [a, d, this]
  have h2 : 218 * (Θι ^ 2 * (Lg + 1)) ≤ R₀ := by
    refine le_trans ?_ hR₀
    have hs4 : s ^ 4 ≤ 1 := pow_le_one₀ hs0.le (by linarith only [hs1])
    rw [le_div_iff₀ (by positivity)]
    have := mul_le_mul_of_nonneg_left hs4 hX0
    nlinarith only [this, hX0, hQ, mul_nonneg hQ hX0]
  exact h1.trans (mul_le_mul_of_nonneg_left h2 (by linarith only [haε]))

/-- The walk starts' `B₃ ≤ K₁ ℓ ≤ T`, so `T⁵` pays their mass. -/
lemma tol_B₃ {ℓ Λ E E₁ aε lg K₁ B₃ R₀ T Ak : ℝ} (hℓ : 1 ≤ ℓ) (hΛ : 1 ≤ Λ) (haε : 1 ≤ aε)
    (hlg : 0 ≤ lg) (hE : E = 1536 * aε * E₁ + 1536 * lg * aε) (hR₀ : 0 ≤ R₀) (hRE : R₀ ≤ E₁)
    (hK : K₁ ≤ aε * R₀) (hK₁ : 0 ≤ K₁) (hB₃0 : 0 ≤ B₃) (hB₃ : B₃ ≤ K₁ * ℓ)
    (hT : T = ℓ ^ 4 * Λ * E) (hT1 : 1 ≤ T) (hAk : 2 ^ 4 * T ^ 5 ≤ Ak) :
    16 * B₃ ^ 2 * Ak⁻¹ ≤ 1 := by
  have ha0 : 0 ≤ aε := by linarith only [haε]
  have hE₁0 : 0 ≤ E₁ := hR₀.trans hRE
  have hKE : K₁ ≤ E := by
    have a := mul_le_mul_of_nonneg_left hRE ha0
    have b : 0 ≤ 1535 * aε * E₁ + 1536 * lg * aε := by positivity
    rw [hE]; linarith only [hK, a, b]
  have hE0 : 0 ≤ E := hK₁.trans hKE
  have hΛ0 : 0 ≤ Λ := by linarith only [hΛ]
  have hl : ℓ ≤ ℓ ^ 4 * Λ := by
    have : ℓ ≤ ℓ ^ 4 := le_self_pow₀ hℓ (by norm_num)
    exact this.trans (le_mul_of_one_le_right (by positivity) hΛ)
  have hBT : B₃ ≤ T := by
    have a := mul_le_mul hKE hl (by linarith only [hℓ]) hE0
    rw [hT]; linarith only [hB₃, a]
  have h2 : B₃ ^ 2 ≤ T ^ 2 := pow_le_pow_left₀ hB₃0 hBT 2
  have h5 : T ^ 2 ≤ T ^ 5 := pow_le_pow_right₀ hT1 (by norm_num)
  have hAk0 : 0 < Ak := lt_of_lt_of_le (by positivity) hAk
  rw [← div_eq_mul_inv, div_le_one hAk0]
  linarith only [h2, h5, hAk]

lemma tol_E₁_ge {Q R₀ Θι Θa P₀ pAP s l16 l128 E₁ : ℝ} (hR₀ : 0 ≤ R₀) (hΘι : 1 ≤ Θι)
    (hΘa : 1 ≤ Θa) (hP₀ : 1 ≤ P₀) (hp : 0 < pAP) (hp1 : pAP ≤ 1) (hs : 0 < s)
    (hl16 : 0 ≤ l16) (hl : l16 ≤ l128)
    (hE₁ : E₁ = 2 ^ 20 * Θa * (R₀ + 1) / s + (Q + 3) ^ 2 * (2 ^ 24 * Θι * (R₀ + 1)
      + 2 ^ 28 * Θι ^ 2 * l128 + 2 ^ 18 * (P₀ + 2) / pAP)) :
    (Q + 3) ^ 2 * (2 ^ 24 * (R₀ + 1) + 2 ^ 28 * l16 + (P₀ + 2)) ≤ E₁ := by
  have hP2 : 0 ≤ P₀ + 2 := by linarith only [hP₀]
  have hR1 : 0 ≤ R₀ + 1 := by linarith only [hR₀]
  have a1 := mul_le_mul_of_nonneg_right hΘι hR1
  have a2 := mul_le_mul (one_le_pow₀ hΘι : (1 : ℝ) ≤ Θι ^ 2) hl hl16 (sq_nonneg Θι)
  have a3 : P₀ + 2 ≤ 2 ^ 18 * (P₀ + 2) / pAP := by
    rw [le_div_iff₀ (by positivity)]
    have h5 : pAP ≤ 2 ^ 18 := by linarith only [hp1]
    nlinarith only [mul_le_mul_of_nonneg_left h5 hP2]
  have a0 : 0 ≤ 2 ^ 20 * Θa * (R₀ + 1) / s :=
    div_nonneg (mul_nonneg (mul_nonneg (by norm_num) (by linarith only [hΘa])) hR1) hs.le
  have hsum : 2 ^ 24 * (R₀ + 1) + 2 ^ 28 * l16 + (P₀ + 2)
      ≤ 2 ^ 24 * Θι * (R₀ + 1) + 2 ^ 28 * Θι ^ 2 * l128 + 2 ^ 18 * (P₀ + 2) / pAP := by
    linarith only [a1, a2, a3]
  have := mul_le_mul_of_nonneg_left hsum (sq_nonneg (Q + 3))
  rw [hE₁]; linarith only [this, a0]

lemma tol_FV {X pAP lp PB P₀ ℓ Lm E₁ T : ℝ} (hX : 0 ≤ X) (hp : 0 < pAP) (hp1 : pAP ≤ 1)
    (hlp : 0 ≤ lp) (hPBd : PB = X / pAP + lp) (hPB : PB ≤ P₀ * ℓ) (hP₀E : P₀ ≤ E₁) (hℓ : 0 ≤ ℓ)
    (hE₁ : 0 ≤ E₁) (hℓLm : ℓ ≤ Lm) (hT : Lm * E₁ ≤ T) : X / 4 ≤ T := by
  have h2 : X ≤ X / pAP := le_div_self hX hp hp1
  have k1 := mul_le_mul_of_nonneg_right hP₀E hℓ
  have k2 := mul_le_mul_of_nonneg_left hℓLm hE₁
  rw [hPBd] at hPB
  linarith only [h2, hlp, hPB, k1, k2, hT, hX]

/-- The family's misread tail, `F ≤ T` letters past a length-`L` cut. -/
lemma tol_k6 {A T c δ X : ℝ} {F Ls : ℕ} (hA : 2 ≤ A) (hT1 : 1 ≤ T) (hδ : 0 < δ)
    (hFT : (F : ℝ) ≤ T) (hAT : A ≤ T) (hLsT : (Ls : ℝ) + 1 ≤ 2 * T) (hcT : 1 ≤ c * T)
    (hδT : Real.log (16 / δ) ≤ 16 * T) (hAL : 2 ^ 4 * T ^ 5 ≤ X) :
    2 * ((F : ℝ) + 1) ^ 3 * (A ^ Ls + 1) ^ F * Real.exp (-2 * X * c ^ 2) ≤ δ / 8 := by
  have hT0 : 0 ≤ T := by linarith only [hT1]
  have hA0 : 0 < A := by linarith only [hA]
  have hA1 : 1 ≤ A := by linarith only [hA]
  have hF0 : (0 : ℝ) ≤ F := Nat.cast_nonneg _
  have e1 : ((F : ℝ) + 1) ^ 3 ≤ Real.exp (3 * F) := by
    calc ((F : ℝ) + 1) ^ 3 ≤ Real.exp F ^ 3 :=
          pow_le_pow_left₀ (by linarith only [hF0])
            (by linarith only [Real.add_one_le_exp (F : ℝ)]) 3
      _ = Real.exp (3 * F) := by rw [← Real.exp_nat_mul]; norm_num
  have e2 : (A ^ Ls + 1) ^ F ≤ Real.exp (A * ((Ls : ℝ) + 1) * F) := by
    have b1 : A ^ Ls + 1 ≤ A ^ (Ls + 1) := by
      have h1 : 1 ≤ A ^ Ls := one_le_pow₀ hA1
      rw [pow_succ]
      nlinarith only [mul_le_mul_of_nonneg_left hA (by linarith only [h1] : (0 : ℝ) ≤ A ^ Ls), h1]
    have b2 : A ^ (Ls + 1) ≤ Real.exp (A * ((Ls : ℝ) + 1)) := by
      calc A ^ (Ls + 1) ≤ Real.exp A ^ (Ls + 1) :=
            pow_le_pow_left₀ hA0.le (by linarith only [Real.add_one_le_exp A]) _
        _ = Real.exp (A * ((Ls : ℝ) + 1)) := by
            rw [← Real.exp_nat_mul]; push_cast; ring_nf
    calc (A ^ Ls + 1) ^ F ≤ Real.exp (A * ((Ls : ℝ) + 1)) ^ F :=
          pow_le_pow_left₀ (by positivity) (b1.trans b2) F
      _ = Real.exp (A * ((Ls : ℝ) + 1) * F) := by rw [← Real.exp_nat_mul]; ring_nf
  have hexp : 3 * (F : ℝ) + A * ((Ls : ℝ) + 1) * F + Real.log (16 / δ) ≤ 2 * X * c ^ 2 := by
    have l1 : Real.log (16 / δ) ≤ 16 * T := hδT
    have l2 : A * ((Ls : ℝ) + 1) * F ≤ T * (2 * T) * T :=
      mul_le_mul (mul_le_mul hAT hLsT (by positivity) hT0) hFT hF0
        (mul_nonneg hT0 (by linarith only [hT1]))
    have hT3 : T ≤ T ^ 3 := le_self_pow₀ hT1 (by norm_num)
    have l3 : 3 * (F : ℝ) + A * ((Ls : ℝ) + 1) * F + Real.log (16 / δ) ≤ 21 * T ^ 3 := by
      linarith only [l1, l2, hT3, hFT]
    have l4 : 2 ^ 5 * T ^ 3 ≤ 2 * X * c ^ 2 := by
      have h3 : T ^ 3 ≤ T ^ 5 * c ^ 2 := by
        have h' : 1 ≤ (c * T) ^ 2 := one_le_pow₀ hcT
        have e : T ^ 5 * c ^ 2 = T ^ 3 * (c * T) ^ 2 := by ring
        rw [e]; exact le_mul_of_one_le_right (pow_nonneg hT0 3) h'
      have := mul_le_mul_of_nonneg_right hAL (sq_nonneg c)
      linarith only [h3, this]
    linarith only [l3, l4, pow_nonneg hT0 3]
  have hlog : Real.exp (3 * F) * Real.exp (A * ((Ls : ℝ) + 1) * F)
      * Real.exp (-2 * X * c ^ 2) ≤ δ / 16 := by
    rw [← Real.exp_add, ← Real.exp_add]
    calc Real.exp (3 * F + A * ((Ls : ℝ) + 1) * F + -2 * X * c ^ 2)
        ≤ Real.exp (-Real.log (16 / δ)) := Real.exp_le_exp.2 (by linarith only [hexp])
      _ = δ / 16 := by
          rw [Real.exp_neg, Real.exp_log (by positivity), inv_div]
  have hE0 : 0 ≤ Real.exp (-2 * X * c ^ 2) := (Real.exp_pos _).le
  have hP : 0 ≤ (A ^ Ls + 1) ^ F := by positivity
  calc 2 * ((F : ℝ) + 1) ^ 3 * (A ^ Ls + 1) ^ F * Real.exp (-2 * X * c ^ 2)
      ≤ 2 * Real.exp (3 * F) * Real.exp (A * ((Ls : ℝ) + 1) * F)
        * Real.exp (-2 * X * c ^ 2) := by gcongr
    _ = 2 * (Real.exp (3 * F) * Real.exp (A * ((Ls : ℝ) + 1) * F)
        * Real.exp (-2 * X * c ^ 2)) := by ring
    _ ≤ 2 * (δ / 16) := by gcongr
    _ = δ / 8 := by ring

end OrthoDFA

end
