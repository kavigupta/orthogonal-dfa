import RLStar.Proofs.BigO

/-!
# The trie tolerance's arithmetic

`trie_loop_tol`'s steps over abstract reals: the tolerance

    T = ℓ⁴ Λ E + ℓ² E₃

against the rates and caps `trie_loop_core` needs, once `2³⁰ T⁹ ≤ |Σ|^k`.
-/

noncomputable section

namespace OrthoDFA

open Real

lemma tol_T_le {ℓ Λ E E₁ E₂ E₃ aε lg T : ℝ} (hℓ : 1 ≤ ℓ) (hΛ : 1 ≤ Λ) (haε : 1 ≤ aε)
    (hE₁ : 1 ≤ E₁) (hE₂ : 0 ≤ E₂) (hE₃ : 0 ≤ E₃) (hlg : 0 ≤ lg)
    (hE : E = 1536 * aε * E₁ + 1536 * lg * aε)
    (hT : T = ℓ ^ 4 * Λ * E + ℓ ^ 2 * Λ * E₂ + ℓ ^ 2 * E₃) :
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
    have h1 := mul_le_mul_of_nonneg_left hEaE hu0
    have h2 : 0 ≤ ℓ ^ 2 * Λ * E₂ := mul_nonneg (mul_nonneg (sq_nonneg ℓ) hΛ0) hE₂
    have h3 : 0 ≤ ℓ ^ 2 * E₃ := mul_nonneg (sq_nonneg ℓ) hE₃
    rw [hT]; linarith only [h1, h2, h3]
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

/-- `E₃ = 16 K₁²/δ'` with `K₁ ≥ |Q| + 1` holds `(|Q| + 1)²/δ` and `1/δ`. -/
lemma tol_Qδ {Q K₁ δ δ' T E₃ : ℝ} (hQ : 0 ≤ Q) (hK : Q + 1 ≤ K₁) (hδ : 0 < δ)
    (hδ' : δ' = δ / (8 * (Q + 1))) (hE₃ : E₃ = 16 * K₁ ^ 2 / δ') (hET : E₃ ≤ T) :
    (Q + 1) ^ 2 ≤ T * δ ∧ 1 / δ ≤ T := by
  have e : E₃ * δ = 128 * K₁ ^ 2 * (Q + 1) := by
    rw [hE₃, hδ']; field_simp; ring
  have h1 := mul_le_mul_of_nonneg_right hET hδ.le
  have k1 : (Q + 1) ^ 2 ≤ K₁ ^ 2 := pow_le_pow_left₀ (by linarith only [hQ]) hK 2
  have k2 : K₁ ^ 2 ≤ 128 * K₁ ^ 2 * (Q + 1) := by nlinarith only [hQ, sq_nonneg K₁]
  have hQδ : (Q + 1) ^ 2 ≤ T * δ := by linarith only [k1, k2, h1, e]
  refine ⟨hQδ, ?_⟩
  rw [div_le_iff₀ hδ]
  have : 1 ≤ (Q + 1) ^ 2 := one_le_pow₀ (by linarith only [hQ])
  linarith only [this, hQδ]

/-- The scan cap `SC`, the harvest cap `HN` and the top-up's `Tb` within powers of `T`. -/
lemma tol_k5 {Q RB R₀ Lm l16 Z E₁ T δ δ' ε p₀ SC HN Tb : ℝ} (hQ : 0 ≤ Q) (hRB0 : 0 ≤ RB)
    (hRB : RB ≤ R₀ * Lm) (hR₀ : 0 ≤ R₀) (hLm : 1 ≤ Lm) (hl16 : 0 ≤ l16) (hZ : 0 ≤ Z)
    (hE : (Q + 3) ^ 2 * (2 ^ 24 * (R₀ + 1) + 2 ^ 28 * l16 + Z) ≤ E₁) (hTlE : Lm * E₁ ≤ T)
    (hT1 : 1 ≤ T) (hQδ : (Q + 1) ^ 2 ≤ T * δ) (hδ : 0 < δ) (hδ' : δ' = δ / (8 * (Q + 1)))
    (hδ'1 : δ' ≤ 1) (hε : 0 < ε) (hinvε : 1 / ε ≤ T) (hp₀ : 0 < p₀)
    (hT9p : 2 ^ 30 * T ^ 9 * p₀ ≤ 1)
    (hSC : SC = 8 * (Q + 4) * RB + 16 * l16 + 2)
    (hHN : HN = 192 / ε * Real.log (4 * (Q + 1) * SC / δ') + 1)
    (hTb : Tb = (Q + 1) * (6 * (Q + 2) * RB)) :
    (Q + 1) ^ 2 * (SC * HN * Tb + (SC * HN) ^ 2) * p₀ ≤ δ / 8 := by
  have hLm0 : 0 ≤ Lm := by linarith only [hLm]
  have hT0 : 0 ≤ T := by linarith only [hT1]
  have hd : 0 < δ' := by rw [hδ']; positivity
  have hSCT : SC ≤ T := by
    have c1 : 8 * (Q + 4) ≤ 2 ^ 24 * (Q + 3) ^ 2 := by nlinarith only [hQ]
    have c2 := mul_le_mul_of_nonneg_right c1 hR₀
    have c3 : 16 * l16 ≤ 2 ^ 28 * (Q + 3) ^ 2 * l16 :=
      mul_le_mul_of_nonneg_right (by nlinarith only [hQ]) hl16
    have c4 : 2 ≤ 2 ^ 24 * (Q + 3) ^ 2 := by nlinarith only [hQ]
    have c5 : 0 ≤ (Q + 3) ^ 2 * Z := mul_nonneg (sq_nonneg _) hZ
    have q1 : 8 * (Q + 4) * R₀ + 16 * l16 + 2 ≤ E₁ := by linarith only [c2, c3, c4, c5, hE]
    have r1 := mul_le_mul_of_nonneg_left hRB (by linarith only [hQ] : (0 : ℝ) ≤ 8 * (Q + 4))
    have r2 : 16 * l16 + 2 ≤ Lm * (16 * l16 + 2) :=
      le_mul_of_one_le_left (by linarith only [hl16]) hLm
    have q3 := mul_le_mul_of_nonneg_left q1 hLm0
    rw [hSC]; linarith only [r1, r2, q3, hTlE]
  have hSC2 : 2 ≤ SC := by
    have : 0 ≤ 8 * (Q + 4) * RB := mul_nonneg (by linarith only [hQ]) hRB0
    rw [hSC]; linarith only [this, hl16]
  have hTbT : Tb ≤ T := by
    have c1 : 6 * (Q + 1) * (Q + 2) ≤ 2 ^ 24 * (Q + 3) ^ 2 := by nlinarith only [hQ]
    have r1 : Tb ≤ 6 * (Q + 1) * (Q + 2) * (R₀ * Lm) := by
      have := mul_le_mul_of_nonneg_left hRB
        (by nlinarith only [hQ] : (0 : ℝ) ≤ 6 * (Q + 1) * (Q + 2))
      rw [hTb]; linarith only [this]
    have r2 : 6 * (Q + 1) * (Q + 2) * R₀ ≤ E₁ := by
      have k1 := mul_le_mul_of_nonneg_right c1 hR₀
      have k2 : 0 ≤ (Q + 3) ^ 2 * (2 ^ 24 + 2 ^ 28 * l16 + Z) :=
        mul_nonneg (sq_nonneg _) (by linarith only [hl16, hZ])
      linarith only [k1, k2, hE]
    have r3 := mul_le_mul_of_nonneg_right r2 hLm0
    linarith only [r1, r3, hTlE]
  have hy0 : 1 ≤ 4 * (Q + 1) * SC / δ' := by
    rw [le_div_iff₀ hd]
    nlinarith only [mul_le_mul (by linarith only [hQ] : (1 : ℝ) ≤ Q + 1) hSC2 (by norm_num)
      (by linarith only [hQ]), hδ'1]
  have hy : 4 * (Q + 1) * SC / δ' ≤ 32 * T ^ 2 := by
    have h1 : (Q + 1) ^ 2 * SC ≤ T * δ * T :=
      mul_le_mul hQδ hSCT (by linarith only [hSC2]) (mul_nonneg hT0 hδ.le)
    have e : 4 * (Q + 1) * SC / δ' = 32 * ((Q + 1) ^ 2 * SC) / δ := by
      rw [hδ', div_div_eq_mul_div]; ring
    rw [e, div_le_iff₀ hδ]
    linarith only [h1]
  have hl0 : 0 ≤ Real.log (4 * (Q + 1) * SC / δ') := Real.log_nonneg hy0
  have hHN1 : 1 ≤ HN := by
    have : 0 ≤ 192 / ε * Real.log (4 * (Q + 1) * SC / δ') :=
      mul_nonneg (div_nonneg (by norm_num) hε.le) hl0
    rw [hHN]; linarith only [this]
  have hHNT : HN ≤ 2 ^ 13 * T ^ 3 := by
    have hl : Real.log (4 * (Q + 1) * SC / δ') ≤ 32 * T ^ 2 :=
      (Real.log_le_sub_one_of_pos (by linarith only [hy0])).trans (by linarith only [hy])
    have hε' : 192 / ε ≤ 192 * T := by
      rw [div_eq_mul_one_div]; linarith only [hinvε]
    have h1 : 192 / ε * Real.log (4 * (Q + 1) * SC / δ') ≤ 192 * T * (32 * T ^ 2) :=
      mul_le_mul hε' hl hl0 (by linarith only [hT1])
    have : 1 ≤ T ^ 3 := one_le_pow₀ hT1
    rw [hHN]; linarith only [h1, this]
  have hSH : SC * HN ≤ 2 ^ 13 * T ^ 4 := by
    have := mul_le_mul hSCT hHNT (by linarith only [hHN1]) hT0
    linarith only [this]
  have hSH0 : 0 ≤ SC * HN := mul_nonneg (by linarith only [hSC2]) (by linarith only [hHN1])
  have hTb0 : 0 ≤ Tb := by
    rw [hTb]
    exact mul_nonneg (by linarith only [hQ])
      (mul_nonneg (by linarith only [hQ]) hRB0)
  have hs1 : SC * HN * Tb ≤ 2 ^ 13 * T ^ 4 * T :=
    mul_le_mul hSH hTbT hTb0 (mul_nonneg (by norm_num) (pow_nonneg hT0 4))
  have hs2 : (SC * HN) ^ 2 ≤ (2 ^ 13 * T ^ 4) ^ 2 := pow_le_pow_left₀ hSH0 hSH 2
  have hT58 : T ^ 5 ≤ T ^ 8 := pow_le_pow_right₀ hT1 (by norm_num)
  have hsum : SC * HN * Tb + (SC * HN) ^ 2 ≤ 2 ^ 27 * T ^ 8 := by
    linarith only [hs1, hs2, hT58, pow_nonneg hT0 8]
  have h1 : (Q + 1) ^ 2 * (SC * HN * Tb + (SC * HN) ^ 2) ≤ T * δ * (2 ^ 27 * T ^ 8) :=
    mul_le_mul hQδ hsum (add_nonneg (mul_nonneg hSH0 hTb0) (sq_nonneg _)) (mul_nonneg hT0 hδ.le)
  have h2 := mul_le_mul_of_nonneg_right h1 hp₀.le
  have e : T * δ * (2 ^ 27 * T ^ 8) * p₀ = δ / 8 * (2 ^ 30 * T ^ 9 * p₀) := by ring
  rw [e] at h2
  have := mul_le_mul_of_nonneg_left hT9p (by positivity : 0 ≤ δ / 8)
  linarith only [h2, this]

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
    (hδT : Real.log (16 / δ) ≤ 16 * T) (hAL : 2 ^ 30 * T ^ 9 ≤ X) :
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
    have l4 : 2 ^ 31 * T ^ 7 ≤ 2 * X * c ^ 2 := by
      have h7 : T ^ 7 ≤ T ^ 9 * c ^ 2 := by
        have h' : 1 ≤ (c * T) ^ 2 := one_le_pow₀ hcT
        have e : T ^ 9 * c ^ 2 = T ^ 7 * (c * T) ^ 2 := by ring
        rw [e]; exact le_mul_of_one_le_right (pow_nonneg hT0 7) h'
      have := mul_le_mul_of_nonneg_right hAL (sq_nonneg c)
      linarith only [h7, this]
    have l5 : 21 * T ^ 3 ≤ 2 ^ 31 * T ^ 7 := by
      have : T ^ 3 ≤ T ^ 7 := pow_le_pow_right₀ hT1 (by norm_num)
      linarith only [this, pow_nonneg hT0 7]
    linarith only [l3, l4, l5]
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
