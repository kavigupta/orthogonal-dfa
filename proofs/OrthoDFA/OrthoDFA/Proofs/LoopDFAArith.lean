import OrthoDFA.Proofs.LoopUniform

/-!
# The learner's DFA: its closed-form length

`dfa_cap_bound` turns `ℓ²⁰ E ≤ |Σ|^k` into the two inequalities `LearnerLoopDFASimple` asks of
the probes' prefix mass: with `ℓ` the longest probe plus one, `τ` is at least `1/(Θ ℓ³)`, the
suffix pool at most `ℓ³ P₀`, the prefix count at most `ℓ¹⁰ R₀` and `1/ρ` at most `ℓ⁴` times a
constant, so `4/(cap ρ) ≤ ℓ²⁰ E` with `E` free of `L`. With the prefix count and the budget per
`ℓ²`, `dfa_collide_arith` turns `ℓ²⁰ E₂ ≤ |Σ|^k` into the held-out strings' collision bound.
`dfa_pow` gets `ℓ²⁰ (E + E₂) ≤ |Σ|^k` from `L ≥ L₀`, the walk starting at half of `L`.
-/

noncomputable section

namespace OrthoDFA

lemma twenty_log_le {A ℓ : ℝ} (hA : 2 ≤ A) (hℓ : 0 < ℓ) :
    20 * Real.log ℓ ≤ 100 + ℓ * Real.log A / 4 := by
  have hl2 := Real.log_two_gt_d9
  have hlA : Real.log 2 ≤ Real.log A := Real.log_le_log (by norm_num) hA
  have ha : 0 < Real.log 2 / 160 := by positivity
  have h1 := Real.log_le_sub_one_of_pos (mul_pos hℓ ha)
  rw [Real.log_mul hℓ.ne' ha.ne'] at h1
  have h2 : -Real.log (Real.log 2 / 160) ≤ 6 := by
    rw [← Real.log_inv, inv_div, Real.log_le_iff_le_exp (by positivity)]
    have he := Real.exp_one_gt_d9
    have e : Real.exp 6 = Real.exp 1 ^ 6 := by rw [← Real.exp_nat_mul]; norm_num
    rw [e, div_le_iff₀ (by positivity)]
    have h6 : (2.7 : ℝ) ^ 6 ≤ Real.exp 1 ^ 6 := pow_le_pow_left₀ (by norm_num) (by linarith) 6
    nlinarith
  have h3 : ℓ * Real.log 2 ≤ ℓ * Real.log A := mul_le_mul_of_nonneg_left hlA hℓ.le
  have e : ℓ * (Real.log 2 / 160) = ℓ * Real.log 2 / 160 := by ring
  rw [e] at h1
  nlinarith

set_option maxHeartbeats 4000000 in -- many small nonlinear steps
lemma dfa_cap_bound {Q A ℓ Nn η₀ c pAP εd εcov δ' ι ρ s τ V Kp G Ce Θ K₁ B₁ P₀ R₀ E₁ E : ℝ}
    {m n : ℕ}
    (hQ : 1 ≤ Q) (hA : 2 ≤ A) (hℓ : 1 ≤ ℓ) (hN1 : 1 ≤ Nn) (hNℓ : Nn ≤ ℓ) (hη0 : 0 ≤ η₀)
    (hη : η₀ < 1 / 2) (hc : 0 < c) (hp : 0 < pAP)
    (hp1 : pAP ≤ 1)
    (hεd : εd = c / (34 * (A + 1) ^ 2 * (Q + 2) * Nn * ℓ))
    (hε0 : 0 < εcov) (hCe0 : 0 < Ce) (hεl : 1 / (Ce * ℓ ^ 3) ≤ εcov)
    (hδ'0 : 0 < δ') (hδ'1 : δ' ≤ 1)
    (hι : ι = min (εcov / 8) (1 / (3072 * ℓ * (Q + 1) * Q)))
    (hρ : ρ = εd ^ 2 / ((Q + 2) * A)) (hs : s = 1 / 2 - η₀) (hτ : τ = min (εcov / 2) (min s ι))
    (hV : V = (Q + 2) * ((Q + 2) * A + 1)) (hKp : Kp = V * (224 / εd + 1))
    (hG : G = Real.log (4 * V * (2 * (Q + 2) * A + 4) / δ'))
    (hΘ : Θ = 1 / s + 8 * Ce + 3072 * Q * (Q + 1))
    (hK₁ : K₁ = V * (7616 * (A + 1) ^ 2 * (Q + 2) / c + 1))
    (hB₁ : B₁ = 4 * (3 + Real.log (4 * K₁ * G * (Q + 3) * A ^ m / δ')))
    (hP₀ : P₀ = 256 * (2 * Θ + B₁) / (s ^ 2 * pAP)
      + 64 * Real.log ((Q + 4) / (δ' / 2)) / pAP ^ 2)
    (hR₀ : R₀ = 6000 * (Q + 2) ^ 2 * Θ ^ 3
      * (Real.log ((Q + 4) * P₀ * Θ / (δ' / 2 * (1 / 4) * (pAP / 2))) + 6) / s ^ 4)
    (hE₁ : E₁ = 2 ^ 24 * (Q + 3) ^ 2 * (Θ ^ 2 * (R₀ + 1) + (P₀ + 2) / pAP) / δ')
    (hE : E = 4 * (Q + 2) * A * (34 * (A + 1) ^ 2 * (Q + 2) / c) ^ 2 * E₁
      + 16384 * (Q + 2) ^ 2 * A ^ 2 * (34 * (A + 1) ^ 2 * (Q + 2) / c) ^ 2
        * Real.log ((Q + 3) * A / δ')) :
    let J := Q + 2
    let PB := 256 * (Real.log (2 / τ) + 4 * Real.log (4 * Kp * G * ℓ * (Q + 3) * A ^ m / δ'))
      / (s ^ 2 * pAP)
      + 64 * Real.log ((J + 2) / (δ' / 2)) / pAP ^ 2
    let RB := 6000 * J ^ 2 * Real.log ((J + 2) * PB / (δ' / 2 * (1 / 4) * (pAP / 2) * τ))
      / (s ^ 4 * τ ^ 3)
    let cap := min (min (δ' * τ ^ 2 / (2 ^ 22 * (J + 1) ^ 2 * (RB + 1)))
        (δ' * ι ^ 2 / (2 ^ 11 * (J + 1) * (RB + 1))))
      (min (min (δ' * pAP / (2 ^ 15 * (J + 1) * (PB + 1))) (ι / (12 * J ^ 2 * (RB + 1))))
        (min (pAP / (2 * J * (PB + 2))) (1 / (6 * J * (RB + 1) * ℓ))))
    RB ≤ ℓ ^ 10 * R₀ ∧ Kp ≤ ℓ ^ 2 * K₁ ∧ 0 ≤ R₀ ∧ 1 ≤ G ∧ 0 ≤ E
      ∧ (ℓ ^ 20 * E ≤ A ^ n → (A ^ n)⁻¹ ≤ cap * ρ / 4
        ∧ 16384 * ℓ * (Q + 2) * A * Real.log ((Q + 3) * A / δ') * (A ^ n)⁻¹ ≤ ρ) := by
  intro J PB RB cap
  have hJ : J = Q + 2 := rfl
  have hs0 : 0 < s := by rw [hs]; linarith
  have hs1 : s ≤ 1 / 2 := by rw [hs]; linarith
  have hA0 : 0 < A := by linarith
  have hN0 : 0 < Nn := by linarith
  have hℓ0 : 0 < ℓ := by linarith
  have hεd0 : 0 < εd := by rw [hεd]; positivity
  have hρ0 : 0 < ρ := by rw [hρ]; positivity
  have hAm : 1 ≤ A ^ m := one_le_pow₀ (by linarith)
  have hℓ3 : ℓ ≤ ℓ ^ 3 := le_self_pow₀ hℓ (by norm_num)
  have hℓ31 : 1 ≤ ℓ ^ 3 := one_le_pow₀ hℓ
  have h1s : 0 < 1 / s := by positivity
  have hQQ : 3072 ≤ 3072 * Q * (Q + 1) := by nlinarith only [hQ]
  have hΘs : 1 / s ≤ Θ := by rw [hΘ]; linarith
  have hΘC : 8 * Ce ≤ Θ := by rw [hΘ]; linarith
  have hΘQ : 3072 * Q * (Q + 1) ≤ Θ := by rw [hΘ]; linarith
  have hΘ1 : 3072 ≤ Θ := by linarith
  have hΘ0 : 0 < Θ := by linarith
  -- the band's far side, `τ`, is at least `1/(Θ ℓ³)`
  have hι0 : 0 < ι := by rw [hι]; exact lt_min (by positivity) (by positivity)
  have hιℓ : ι * ℓ ≤ 1 := by
    have h := min_le_right (εcov / 8) (1 / (3072 * ℓ * (Q + 1) * Q))
    rw [← hι, le_div_iff₀ (by positivity)] at h
    nlinarith only [h, hι0, hQQ, hℓ0, hQ]
  have hι1 : ι ≤ 1 := by nlinarith only [hιℓ, hι0, hℓ]
  have hτι : τ ≤ ι := by rw [hτ]; exact (min_le_right _ _).trans (min_le_right _ _)
  have hτ0 : 0 < τ := by rw [hτ]; exact lt_min (by positivity) (lt_min hs0 hι0)
  have hτ1 : τ ≤ 1 := hτι.trans hι1
  have hΘℓ : 0 < Θ * ℓ ^ 3 := by positivity
  have hτΘ : 1 / τ ≤ Θ * ℓ ^ 3 := by
    rw [div_le_iff₀ hτ0, ← div_le_iff₀' hΘℓ]
    have h1 : 1 / (Θ * ℓ ^ 3) ≤ 1 / Θ :=
      one_div_le_one_div_of_le hΘ0 (le_mul_of_one_le_right hΘ0.le hℓ31)
    have h2 : 1 / (Θ * ℓ ^ 3) ≤ εcov / 8 := by
      have : 1 / (Θ * ℓ ^ 3) ≤ 1 / (8 * Ce * ℓ ^ 3) :=
        one_div_le_one_div_of_le (by positivity) (mul_le_mul_of_nonneg_right hΘC (by positivity))
      have e : 1 / (8 * Ce * ℓ ^ 3) = 1 / (Ce * ℓ ^ 3) / 8 := by field_simp
      linarith
    have h3 : 1 / Θ ≤ s := by
      rw [div_le_iff₀ hΘ0]; rw [div_le_iff₀ hs0] at hΘs; linarith
    have h4 : 1 / (Θ * ℓ ^ 3) ≤ 1 / (3072 * ℓ * (Q + 1) * Q) :=
      one_div_le_one_div_of_le (by positivity) (by
        calc 3072 * ℓ * (Q + 1) * Q = (3072 * Q * (Q + 1)) * ℓ := by ring
          _ ≤ Θ * ℓ ^ 3 := mul_le_mul hΘQ hℓ3 hℓ0.le hΘ0.le)
    rw [hτ, hι]
    exact le_min (by linarith) (le_min (by linarith) (le_min (by linarith) h4))
  -- the level-5 budget per `ℓ`, and the suffix pool's bound `PB`
  have hV1 : 1 ≤ V := by
    rw [hV]
    have : 0 ≤ (Q + 2) * ((Q + 2) * A) := mul_nonneg (by linarith) (by nlinarith)
    nlinarith
  have hG1 : 1 ≤ G := by
    rw [hG]
    have hX : 3 ≤ 4 * V * (2 * (Q + 2) * A + 4) / δ' := by
      rw [le_div_iff₀ hδ'0]
      have : (4 : ℝ) ≤ 2 * (Q + 2) * A + 4 := by nlinarith
      nlinarith
    rw [Real.le_log_iff_exp_le (by linarith)]
    linarith [Real.exp_one_lt_d9]
  have hK₁0 : 0 ≤ K₁ := by rw [hK₁]; positivity
  have hKpl : Kp ≤ ℓ ^ 2 * K₁ := by
    have e : 224 / εd = 7616 * (A + 1) ^ 2 * (Q + 2) * Nn * ℓ / c := by
      rw [hεd]; field_simp; ring
    rw [hKp, e, hK₁]
    have hℓ2 : 1 ≤ ℓ ^ 2 := one_le_pow₀ hℓ
    have h1 : 7616 * (A + 1) ^ 2 * (Q + 2) * Nn * ℓ / c
        ≤ ℓ ^ 2 * (7616 * (A + 1) ^ 2 * (Q + 2) / c) := by
      rw [mul_div_assoc', div_le_div_iff_of_pos_right hc]
      have : Nn * ℓ ≤ ℓ ^ 2 := by nlinarith only [hNℓ, hℓ]
      have h0 : 0 ≤ 7616 * (A + 1) ^ 2 * (Q + 2) := by positivity
      nlinarith only [this, h0]
    have h2 : 0 ≤ 7616 * (A + 1) ^ 2 * (Q + 2) / c := by positivity
    nlinarith only [h1, hV1, hℓ2, h2]
  have hKp0 : 0 ≤ Kp := by rw [hKp]; positivity
  set cL := 4 * Real.log (4 * Kp * G * ℓ * (Q + 3) * A ^ m / δ') with hcLd
  have hKp1 : 1 ≤ Kp := by
    rw [hKp]
    have : 0 ≤ 224 / εd := by positivity
    nlinarith
  set X₁ := 4 * K₁ * G * (Q + 3) * A ^ m / δ' with hX₁d
  have hK₁1 : 1 ≤ K₁ := by
    rw [hK₁]
    have : 0 ≤ 7616 * (A + 1) ^ 2 * (Q + 2) / c := by positivity
    nlinarith
  have hX₁ : 1 ≤ X₁ := by
    rw [hX₁d, le_div_iff₀ hδ'0, one_mul]
    have h1 : 1 ≤ K₁ * G := one_le_mul_of_one_le_of_one_le hK₁1 hG1
    have h2 : 1 ≤ (Q + 3) * A ^ m := one_le_mul_of_one_le_of_one_le (by linarith) hAm
    have h3 := mul_le_mul h1 h2 zero_le_one (by linarith)
    nlinarith
  have hlX₁ : 0 ≤ Real.log X₁ := Real.log_nonneg hX₁
  have hB₀ : 0 ≤ B₁ := by rw [hB₁]; positivity
  have hcL0 : 0 ≤ cL := by
    rw [hcLd]
    refine mul_nonneg (by norm_num) (Real.log_nonneg ?_)
    rw [le_div_iff₀ hδ'0, one_mul]
    have h1 : 1 ≤ Kp * G := one_le_mul_of_one_le_of_one_le hKp1 hG1
    have h2 : 1 ≤ ℓ * (Q + 3) * A ^ m :=
      one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le hℓ (by linarith)) hAm
    have h3 := mul_le_mul h1 h2 zero_le_one (by linarith)
    nlinarith
  have hcL : cL ≤ ℓ ^ 3 * B₁ := by
    have h1 : cL ≤ 4 * Real.log (ℓ ^ 3 * X₁) := by
      rw [hcLd]
      refine mul_le_mul_of_nonneg_left (Real.log_le_log ?_ ?_) (by norm_num)
      · have : 0 < Kp := by linarith
        positivity
      · rw [hX₁d]
        have e : ℓ ^ 3 * (4 * K₁ * G * (Q + 3) * A ^ m / δ')
            = 4 * (ℓ ^ 2 * K₁) * G * ℓ * (Q + 3) * A ^ m / δ' := by ring
        rw [e]
        gcongr
    have h2 : Real.log (ℓ ^ 3 * X₁) = 3 * Real.log ℓ + Real.log X₁ := by
      rw [Real.log_mul (by positivity) (by linarith), Real.log_pow]; push_cast; ring
    have h3 : Real.log ℓ ≤ ℓ := by
      have := Real.log_le_sub_one_of_pos hℓ0; linarith
    have h4 : Real.log X₁ ≤ ℓ * Real.log X₁ := le_mul_of_one_le_left hlX₁ hℓ
    have h5 : cL ≤ ℓ * B₁ := by rw [hB₁]; nlinarith
    exact h5.trans (mul_le_mul_of_nonneg_right hℓ3 hB₀)
  have hJ2 : J + 2 = Q + 4 := by rw [hJ]; ring
  have hPB : PB = 256 * (Real.log (2 / τ) + cL) / (s ^ 2 * pAP)
      + 64 * Real.log ((Q + 4) / (δ' / 2)) / pAP ^ 2 := by
    show _ + 64 * Real.log ((J + 2) / (δ' / 2)) / pAP ^ 2 = _
    rw [hJ2]
  have hl2 := Real.log_two_gt_d9
  have hlτ2 : Real.log 2 ≤ Real.log (2 / τ) :=
    Real.log_le_log (by norm_num) (by rw [le_div_iff₀ hτ0]; linarith)
  have hlτ0 : 0 ≤ Real.log (2 / τ) := by linarith
  have hlτ : Real.log (2 / τ) ≤ 2 * (Θ * ℓ ^ 3) := by
    have h1 := Real.log_le_sub_one_of_pos (by positivity : 0 < 2 / τ)
    have h2 : 2 / τ = 2 * (1 / τ) := by ring
    rw [h2] at h1 ⊢
    linarith
  have hlJ0 : 0 ≤ Real.log ((Q + 4) / (δ' / 2)) :=
    Real.log_nonneg (by rw [le_div_iff₀ (by positivity)]; linarith)
  have hsp : s ^ 2 * pAP ≤ 1 / 4 := by
    have : s ^ 2 ≤ 1 / 4 := by nlinarith only [hs0, hs1]
    nlinarith only [this, hp1, sq_nonneg s]
  have hsp0 : 0 < s ^ 2 * pAP := by positivity
  have hPBle : PB ≤ ℓ ^ 3 * P₀ := by
    have e : ℓ ^ 3 * P₀ = 256 * (2 * (Θ * ℓ ^ 3) + ℓ ^ 3 * B₁) / (s ^ 2 * pAP)
        + ℓ ^ 3 * (64 * Real.log ((Q + 4) / (δ' / 2)) / pAP ^ 2) := by rw [hP₀]; ring
    rw [e, hPB]
    have h1 : 256 * (Real.log (2 / τ) + cL) / (s ^ 2 * pAP)
        ≤ 256 * (2 * (Θ * ℓ ^ 3) + ℓ ^ 3 * B₁) / (s ^ 2 * pAP) :=
      div_le_div_of_nonneg_right (by linarith) hsp0.le
    have h2 : 64 * Real.log ((Q + 4) / (δ' / 2)) / pAP ^ 2
        ≤ ℓ ^ 3 * (64 * Real.log ((Q + 4) / (δ' / 2)) / pAP ^ 2) :=
      le_mul_of_one_le_left (by positivity) hℓ31
    linarith
  have hPB1 : 1 ≤ PB := by
    rw [hPB]
    have h1 : 1 ≤ 256 * (Real.log (2 / τ) + cL) / (s ^ 2 * pAP) := by
      rw [le_div_iff₀ hsp0]; nlinarith only [hsp, hlτ2, hl2, hcL0]
    have h2 : 0 ≤ 64 * Real.log ((Q + 4) / (δ' / 2)) / pAP ^ 2 := by positivity
    linarith
  have hPB0 : 0 ≤ PB := by linarith
  have hP₀1 : 1 ≤ P₀ := by
    have h1 : 1 ≤ 256 * (2 * Θ + B₁) / (s ^ 2 * pAP) := by
      rw [le_div_iff₀ hsp0]; nlinarith only [hsp, hΘ1, hB₀]
    have h2 : 0 ≤ 64 * Real.log ((Q + 4) / (δ' / 2)) / pAP ^ 2 := by positivity
    rw [hP₀]; linarith
  -- the prefix count's bound `RB`
  obtain ⟨Y₀, hY₀⟩ : ∃ Y₀, Y₀ = (Q + 4) * P₀ * Θ / (δ' / 2 * (1 / 4) * (pAP / 2)) := ⟨_, rfl⟩
  have hD0 : 0 < δ' / 2 * (1 / 4) * (pAP / 2) := by positivity
  have hD1 : δ' / 2 * (1 / 4) * (pAP / 2) ≤ 1 := by nlinarith only [hδ'0, hδ'1, hp, hp1]
  have hY5 : 5 ≤ (Q + 4) * PB / (δ' / 2 * (1 / 4) * (pAP / 2) * τ) := by
    rw [le_div_iff₀ (by positivity)]
    have : δ' / 2 * (1 / 4) * (pAP / 2) * τ ≤ 1 := mul_le_one₀ hD1 hτ0.le hτ1
    nlinarith only [this, hQ, hPB1]
  have hYle : (Q + 4) * PB / (δ' / 2 * (1 / 4) * (pAP / 2) * τ) ≤ ℓ ^ 6 * Y₀ := by
    have e : (Q + 4) * PB / (δ' / 2 * (1 / 4) * (pAP / 2) * τ)
        = (Q + 4) * PB * (1 / τ) / (δ' / 2 * (1 / 4) * (pAP / 2)) := by field_simp
    have e2 : ℓ ^ 6 * Y₀ = (Q + 4) * (ℓ ^ 3 * P₀) * (Θ * ℓ ^ 3)
        / (δ' / 2 * (1 / 4) * (pAP / 2)) := by rw [hY₀]; ring
    rw [e, e2]
    refine div_le_div_of_nonneg_right ?_ hD0.le
    have h1 : 0 ≤ (Q + 4) * PB := by positivity
    calc (Q + 4) * PB * (1 / τ) ≤ (Q + 4) * PB * (Θ * ℓ ^ 3) :=
          mul_le_mul_of_nonneg_left hτΘ h1
      _ ≤ (Q + 4) * (ℓ ^ 3 * P₀) * (Θ * ℓ ^ 3) := by gcongr
  have hY₀1 : 1 ≤ Y₀ := by
    rw [hY₀, le_div_iff₀ hD0, one_mul]
    have h3 : 1 ≤ (Q + 4) * P₀ * Θ :=
      one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le (by linarith) hP₀1)
        (by linarith)
    linarith
  have hlY₀ : 0 ≤ Real.log Y₀ := Real.log_nonneg hY₀1
  have hlY0 : 0 ≤ Real.log ((Q + 4) * PB / (δ' / 2 * (1 / 4) * (pAP / 2) * τ)) :=
    Real.log_nonneg (by linarith)
  have hlY : Real.log ((Q + 4) * PB / (δ' / 2 * (1 / 4) * (pAP / 2) * τ))
      ≤ ℓ * (Real.log Y₀ + 6) := by
    set u := Real.log ℓ with hu
    have hu0 : 0 ≤ u := Real.log_nonneg hℓ
    have huL : u ≤ ℓ := by
      have := Real.log_le_sub_one_of_pos hℓ0
      rw [← hu] at this; linarith
    have h1 : Real.log ((Q + 4) * PB / (δ' / 2 * (1 / 4) * (pAP / 2) * τ))
        ≤ Real.log Y₀ + 6 * u := by
      have := Real.log_le_log (by linarith) hYle
      rw [Real.log_mul (by positivity) (by linarith), Real.log_pow, ← hu] at this
      push_cast at this
      linarith
    have h2 : Real.log Y₀ ≤ ℓ * Real.log Y₀ := le_mul_of_one_le_left hlY₀ hℓ
    nlinarith only [h1, h2, huL]
  have hR₀0 : 0 ≤ R₀ := by rw [hR₀, ← hY₀]; positivity
  have hRBle : RB ≤ ℓ ^ 10 * R₀ := by
    have hK : 0 ≤ 6000 * (Q + 2) ^ 2 / s ^ 4 := by positivity
    have e1 : RB = 6000 * (Q + 2) ^ 2 / s ^ 4 * (1 / τ) ^ 3
        * Real.log ((Q + 4) * PB / (δ' / 2 * (1 / 4) * (pAP / 2) * τ)) := by
      show 6000 * J ^ 2 * Real.log ((J + 2) * PB / (δ' / 2 * (1 / 4) * (pAP / 2) * τ))
        / (s ^ 4 * τ ^ 3) = _
      rw [hJ2, hJ]; field_simp
    have e2 : ℓ ^ 10 * R₀ = 6000 * (Q + 2) ^ 2 / s ^ 4 * (Θ * ℓ ^ 3) ^ 3
        * (ℓ * (Real.log Y₀ + 6)) := by rw [hR₀, ← hY₀]; ring
    rw [e1, e2]
    have h1 : (1 / τ) ^ 3 ≤ (Θ * ℓ ^ 3) ^ 3 := pow_le_pow_left₀ (by positivity) hτΘ 3
    exact mul_le_mul (mul_le_mul_of_nonneg_left h1 hK) hlY hlY0 (by positivity)
  have hRB0 : 0 ≤ RB := by
    show 0 ≤ 6000 * J ^ 2 * Real.log ((J + 2) * PB / (δ' / 2 * (1 / 4) * (pAP / 2) * τ))
      / (s ^ 4 * τ ^ 3)
    rw [hJ2]; positivity
  -- the cap is at least its first or its third term
  set c1 := δ' * τ ^ 2 / (2 ^ 22 * (J + 1) ^ 2 * (RB + 1)) with hc1
  set c3 := δ' * pAP / (2 ^ 15 * (J + 1) * (PB + 1)) with hc3
  have hJ1 : 3 ≤ J := by rw [hJ]; linarith
  have hτ2 : τ ^ 2 ≤ ι ^ 2 := pow_le_pow_left₀ hτ0.le hτι 2
  have hcap : min c1 c3 ≤ cap := by
    refine le_min (le_min (min_le_left _ _) ((min_le_left _ _).trans ?_))
      (le_min (le_min (min_le_right _ _) ((min_le_left _ _).trans ?_))
        (le_min ((min_le_right _ _).trans ?_) ((min_le_left _ _).trans ?_)))
    · rw [hc1, div_le_div_iff₀ (by positivity) (by positivity)]
      have h1 : 2 ^ 11 * (J + 1) * (RB + 1) ≤ 2 ^ 22 * (J + 1) ^ 2 * (RB + 1) := by
        have : (2 : ℝ) ^ 11 * (J + 1) ≤ 2 ^ 22 * (J + 1) ^ 2 := by nlinarith only [hJ1]
        exact mul_le_mul_of_nonneg_right this (by linarith)
      exact mul_le_mul (mul_le_mul_of_nonneg_left hτ2 hδ'0.le) h1 (by positivity) (by positivity)
    · rw [hc1, div_le_div_iff₀ (by positivity) (by positivity)]
      have h1 : δ' * τ ^ 2 ≤ ι := by
        have : τ ^ 2 ≤ τ := by nlinarith only [hτ0, hτ1]
        nlinarith only [this, hτι, hδ'1, hδ'0, hτ0]
      have h2 : 12 * J ^ 2 * (RB + 1) ≤ 2 ^ 22 * (J + 1) ^ 2 * (RB + 1) := by
        have : 12 * J ^ 2 ≤ 2 ^ 22 * (J + 1) ^ 2 := by nlinarith only [hJ1]
        exact mul_le_mul_of_nonneg_right this (by linarith)
      exact mul_le_mul h1 h2 (by positivity) hι0.le
    · rw [hc3, div_le_div_iff₀ (by positivity) (by positivity)]
      have h1 : δ' * pAP ≤ pAP := by nlinarith only [hδ'1, hp]
      have h2 : 2 * J * (PB + 2) ≤ 2 ^ 15 * (J + 1) * (PB + 1) := by nlinarith only [hJ1, hPB0]
      exact mul_le_mul h1 h2 (by positivity) hp.le
    · rw [hc1, div_le_div_iff₀ (by positivity) (by positivity), one_mul]
      have h1 : τ ^ 2 * ℓ ≤ 1 := by
        have : τ * ℓ ≤ 1 := (mul_le_mul_of_nonneg_right hτι hℓ0.le).trans hιℓ
        nlinarith only [this, hτ1, hτ0, hℓ0]
      have h2 : 6 * J ≤ 2 ^ 22 * (J + 1) ^ 2 := by nlinarith only [hJ1]
      have h3 : δ' * τ ^ 2 * ℓ ≤ 1 := by nlinarith only [h1, hδ'1, hδ'0, hτ0]
      have h4 := mul_le_mul_of_nonneg_right (mul_le_mul h3 h2 (by positivity) zero_le_one)
        (by linarith : (0 : ℝ) ≤ RB + 1)
      nlinarith only [h4]
  -- `4/c ≤ ℓ¹⁶ E₁` for both, and `1/ρ ≤ ℓ · 32(|Σ|+1)/(3c)`
  have hJ3 : J + 1 = Q + 3 := by rw [hJ]; ring
  have hw : 0 ≤ Θ ^ 2 * (R₀ + 1) := by positivity
  have hw2 : 0 ≤ (P₀ + 2) / pAP := by positivity
  have hℓ16 : 1 ≤ ℓ ^ 16 := one_le_pow₀ hℓ
  have hE₁0 : 0 ≤ E₁ := by rw [hE₁]; positivity
  have hb1 : 4 / c1 ≤ ℓ ^ 16 * E₁ := by
    have e : 4 / c1 = 2 ^ 24 * (Q + 3) ^ 2 * ((RB + 1) * (1 / τ) ^ 2) / δ' := by
      rw [hc1, hJ3]; field_simp; ring
    have e2 : ℓ ^ 16 * E₁ = 2 ^ 24 * (Q + 3) ^ 2
        * (ℓ ^ 16 * (Θ ^ 2 * (R₀ + 1) + (P₀ + 2) / pAP)) / δ' := by rw [hE₁]; ring
    rw [e, e2]
    refine div_le_div_of_nonneg_right ?_ hδ'0.le
    refine mul_le_mul_of_nonneg_left ?_ (by positivity)
    have hRB1 : RB + 1 ≤ ℓ ^ 10 * (R₀ + 1) := by
      have : (1 : ℝ) ≤ ℓ ^ 10 := one_le_pow₀ hℓ
      nlinarith only [hRBle, this, hR₀0]
    have hτ2' : (1 / τ) ^ 2 ≤ Θ ^ 2 * ℓ ^ 6 := by
      have := pow_le_pow_left₀ (by positivity) hτΘ 2
      rw [mul_pow, ← pow_mul] at this; exact this
    calc (RB + 1) * (1 / τ) ^ 2 ≤ (ℓ ^ 10 * (R₀ + 1)) * (Θ ^ 2 * ℓ ^ 6) :=
          mul_le_mul hRB1 hτ2' (by positivity) (by positivity)
      _ = ℓ ^ 16 * (Θ ^ 2 * (R₀ + 1)) := by ring
      _ ≤ ℓ ^ 16 * (Θ ^ 2 * (R₀ + 1) + (P₀ + 2) / pAP) :=
          mul_le_mul_of_nonneg_left (by linarith) (by positivity)
  have hb3 : 4 / c3 ≤ ℓ ^ 16 * E₁ := by
    have e : 4 / c3 = 2 ^ 17 * (Q + 3) * ((PB + 1) / pAP) / δ' := by
      rw [hc3, hJ3]; field_simp; ring
    have e2 : ℓ ^ 16 * E₁ = 2 ^ 24 * (Q + 3) ^ 2
        * (ℓ ^ 16 * (Θ ^ 2 * (R₀ + 1) + (P₀ + 2) / pAP)) / δ' := by rw [hE₁]; ring
    rw [e, e2]
    refine div_le_div_of_nonneg_right ?_ hδ'0.le
    have h1 : (PB + 1) / pAP ≤ ℓ ^ 16 * ((P₀ + 2) / pAP) := by
      have : PB + 1 ≤ ℓ ^ 16 * (P₀ + 2) := by
        have hl : ℓ ^ 3 ≤ ℓ ^ 16 := pow_le_pow_right₀ hℓ (by norm_num)
        nlinarith only [hPBle, hl, hP₀1, hℓ16, hℓ31]
      rw [mul_div_assoc']
      exact div_le_div_of_nonneg_right this hp.le
    have h2 : ℓ ^ 16 * ((P₀ + 2) / pAP)
        ≤ ℓ ^ 16 * (Θ ^ 2 * (R₀ + 1) + (P₀ + 2) / pAP) :=
      mul_le_mul_of_nonneg_left (by linarith) (by positivity)
    have h3 : 2 ^ 17 * (Q + 3) ≤ 2 ^ 24 * (Q + 3) ^ 2 := by nlinarith only [hQ]
    exact mul_le_mul h3 (h1.trans h2) (by positivity) (by positivity)
  set c₂ : ℝ := 34 * (A + 1) ^ 2 * (Q + 2) / c with hc₂
  have hc₂0 : 0 < c₂ := by positivity
  have hεdl : 1 / (c₂ * ℓ ^ 2) ≤ εd := by
    have e : 1 / (c₂ * ℓ ^ 2) = c / (34 * (A + 1) ^ 2 * (Q + 2) * ℓ ^ 2) := by
      rw [hc₂]; field_simp
    rw [e, hεd]
    have : Nn * ℓ ≤ ℓ ^ 2 := by nlinarith only [hNℓ, hℓ]
    have h0 : 0 < 34 * (A + 1) ^ 2 * (Q + 2) := by positivity
    refine div_le_div_of_nonneg_left hc.le (by positivity) ?_
    have := mul_le_mul_of_nonneg_left this h0.le
    linarith
  have hρinv : 1 / ρ ≤ ℓ ^ 4 * ((Q + 2) * A * c₂ ^ 2) := by
    have hρe : 1 / ρ = (Q + 2) * A * (1 / εd) ^ 2 := by rw [hρ]; field_simp
    have h1 : 1 / εd ≤ c₂ * ℓ ^ 2 := by
      rw [one_div_le hεd0 (by positivity)]; exact hεdl
    have h2 : (1 / εd) ^ 2 ≤ (c₂ * ℓ ^ 2) ^ 2 := pow_le_pow_left₀ (by positivity) h1 2
    rw [hρe]
    calc (Q + 2) * A * (1 / εd) ^ 2 ≤ (Q + 2) * A * (c₂ * ℓ ^ 2) ^ 2 :=
          mul_le_mul_of_nonneg_left h2 (by positivity)
      _ = ℓ ^ 4 * ((Q + 2) * A * c₂ ^ 2) := by ring
  have hlg0 : 0 < (Q + 3) * A / δ' := by positivity
  have hlg1 : 0 ≤ Real.log ((Q + 3) * A / δ') :=
    Real.log_nonneg (by rw [le_div_iff₀ hδ'0]; nlinarith only [hδ'1, hQ, hA])
  have hEge : 4 * (Q + 2) * A * c₂ ^ 2 * E₁ ≤ E := by
    rw [hE]
    have : 0 ≤ 16384 * (Q + 2) ^ 2 * A ^ 2 * c₂ ^ 2 * Real.log ((Q + 3) * A / δ') := by
      positivity
    linarith
  refine ⟨hRBle, hKpl, hR₀0, hG1, le_trans (by positivity) hEge, fun hpow => ?_⟩
  have hAn : 0 < A ^ n := by positivity
  have hfin : ∀ c' : ℝ, 0 < c' → 4 / c' ≤ ℓ ^ 16 * E₁ → (A ^ n)⁻¹ ≤ c' * ρ / 4 := by
    intro c' hc' h
    rw [← one_div, one_div_le hAn (by positivity)]
    have e : 1 / (c' * ρ / 4) = 4 / c' * (1 / ρ) := by field_simp
    rw [e]
    calc 4 / c' * (1 / ρ) ≤ ℓ ^ 16 * E₁ * (ℓ ^ 4 * ((Q + 2) * A * c₂ ^ 2)) :=
          mul_le_mul h hρinv (by positivity) (by positivity)
      _ = ℓ ^ 20 * ((Q + 2) * A * c₂ ^ 2 * E₁) := by ring
      _ ≤ ℓ ^ 20 * E := by
          refine mul_le_mul_of_nonneg_left ?_ (by positivity)
          have : (Q + 2) * A * c₂ ^ 2 * E₁ ≤ 4 * (Q + 2) * A * c₂ ^ 2 * E₁ := by
            have : 0 ≤ (Q + 2) * A * c₂ ^ 2 * E₁ := by positivity
            linarith
          linarith
      _ ≤ A ^ n := hpow
  refine ⟨?_, ?_⟩
  · have hc10 : 0 < c1 := by rw [hc1]; positivity
    have hc30 : 0 < c3 := by rw [hc3]; positivity
    have hm : (A ^ n)⁻¹ ≤ min c1 c3 * ρ / 4 := by
      rcases min_choice c1 c3 with h | h <;> rw [h]
      · exact hfin c1 hc10 hb1
      · exact hfin c3 hc30 hb3
    exact hm.trans (by gcongr)
  · -- level 5's prefix condition
    set W := 16384 * (Q + 2) * A * Real.log ((Q + 3) * A / δ') with hW
    have hW0 : 0 ≤ W := by positivity
    have hE2 : ℓ ^ 20 * (W * ((Q + 2) * A * c₂ ^ 2)) ≤ A ^ n := by
      refine le_trans (mul_le_mul_of_nonneg_left ?_ (by positivity)) hpow
      rw [hE]
      have h1 : 0 ≤ 4 * (Q + 2) * A * c₂ ^ 2 * E₁ := by positivity
      have e : W * ((Q + 2) * A * c₂ ^ 2)
          = 16384 * (Q + 2) ^ 2 * A ^ 2 * c₂ ^ 2 * Real.log ((Q + 3) * A / δ') := by
        rw [hW]; ring
      linarith
    rw [← div_eq_mul_inv, div_le_iff₀ hAn]
    have e : 16384 * ℓ * (Q + 2) * A * Real.log ((Q + 3) * A / δ') = ℓ * W := by rw [hW]; ring
    rw [e]
    have hρ0 : 0 < ρ := by rw [hρ]; positivity
    have h3 : ℓ * W * (1 / ρ) ≤ ℓ ^ 20 * (W * ((Q + 2) * A * c₂ ^ 2)) := by
      have hℓ5 : ℓ ^ 5 ≤ ℓ ^ 20 := pow_le_pow_right₀ hℓ (by norm_num)
      calc ℓ * W * (1 / ρ) ≤ ℓ * W * (ℓ ^ 4 * ((Q + 2) * A * c₂ ^ 2)) :=
            mul_le_mul_of_nonneg_left hρinv (by positivity)
        _ = ℓ ^ 5 * (W * ((Q + 2) * A * c₂ ^ 2)) := by ring
        _ ≤ ℓ ^ 20 * (W * ((Q + 2) * A * c₂ ^ 2)) :=
            mul_le_mul_of_nonneg_right hℓ5 (by positivity)
    have h4 : ℓ * W * (1 / ρ) ≤ A ^ n := h3.trans hE2
    rw [mul_one_div, div_le_iff₀ hρ0] at h4
    linarith

lemma dfa_collide_arith {Q A ℓ δ RB R₀ Kp K₁ G E₂ : ℝ} {n k : ℕ} (hQ : 0 ≤ Q) (hA : 0 < A)
    (hℓ : 1 ≤ ℓ) (hδ : 0 < δ) (hR₀ : 0 ≤ R₀) (hK₁ : 0 ≤ K₁) (hG : 0 ≤ G)
    (hRB : RB ≤ ℓ ^ 10 * R₀) (hKp : Kp ≤ ℓ ^ 2 * K₁)
    (hE₂ : E₂ = 4 * (Q + 1) * n * (6 * (Q + 2) * R₀ + K₁ * G + n) / δ)
    (hpow : ℓ ^ 20 * E₂ ≤ A ^ k) :
    (Q + 1) * n * (6 * (Q + 2) * RB + Kp * G + n) * (A ^ k)⁻¹ ≤ δ / 4 := by
  have hAk : 0 < A ^ k := by positivity
  have hn : (0 : ℝ) ≤ n := Nat.cast_nonneg _
  have hℓ10 : ℓ ^ 10 ≤ ℓ ^ 20 := pow_le_pow_right₀ hℓ (by norm_num)
  have hℓ1 : ℓ ^ 2 ≤ ℓ ^ 20 := pow_le_pow_right₀ hℓ (by norm_num)
  have hℓ17 : 1 ≤ ℓ ^ 20 := one_le_pow₀ hℓ
  have h1 : 6 * (Q + 2) * RB + Kp * G + n ≤ ℓ ^ 20 * (6 * (Q + 2) * R₀ + K₁ * G + n) := by
    have a1 : 6 * (Q + 2) * RB ≤ ℓ ^ 20 * (6 * (Q + 2) * R₀) := by
      have : RB ≤ ℓ ^ 20 * R₀ := hRB.trans (mul_le_mul_of_nonneg_right hℓ10 hR₀)
      nlinarith
    have a2 : Kp * G ≤ ℓ ^ 20 * (K₁ * G) := by
      have : Kp ≤ ℓ ^ 20 * K₁ := hKp.trans (mul_le_mul_of_nonneg_right hℓ1 hK₁)
      nlinarith
    have a3 : (n : ℝ) ≤ ℓ ^ 20 * n := le_mul_of_one_le_left hn hℓ17
    nlinarith
  rw [← div_eq_mul_inv, div_le_iff₀ hAk]
  have h2 : ℓ ^ 20 * E₂ * δ / 4
      = (Q + 1) * n * (ℓ ^ 20 * (6 * (Q + 2) * R₀ + K₁ * G + n)) := by
    rw [hE₂]; field_simp
  have h3 : (Q + 1) * n * (6 * (Q + 2) * RB + Kp * G + n)
      ≤ (Q + 1) * n * (ℓ ^ 20 * (6 * (Q + 2) * R₀ + K₁ * G + n)) :=
    mul_le_mul_of_nonneg_left h1 (by positivity)
  have h4 : ℓ ^ 20 * E₂ * δ / 4 ≤ δ / 4 * A ^ k := by nlinarith
  linarith

lemma dfa_pow {A ℓ E L Q : ℝ} {k : ℕ} (hA : 2 ≤ A) (hℓ : ℓ = L + Q + 4)
    (hℓ0 : 0 < ℓ) (hk : L ≤ 2 * k) (hL₀ : Q + 4 + 4 * (Real.log E + 100) / Real.log A ≤ L) :
    ℓ ^ 20 * E ≤ A ^ k := by
  rcases le_or_gt E 0 with hE | hE
  · have : ℓ ^ 20 * E ≤ 0 := mul_nonpos_of_nonneg_of_nonpos (by positivity) hE
    exact this.trans (by positivity)
  subst hℓ
  have hlA : 0 < Real.log A := Real.log_pos (by linarith)
  have h1 := twenty_log_le hA hℓ0
  have h2 : 4 * (Real.log E + 100) ≤ (L - Q - 4) * Real.log A := by
    have := sub_le_sub_right hL₀ (Q + 4)
    rw [add_sub_cancel_left, div_le_iff₀ hlA] at this
    linarith
  have h3 : L * Real.log A ≤ 2 * k * Real.log A := mul_le_mul_of_nonneg_right hk hlA.le
  rw [← Real.log_le_log_iff (by positivity) (by positivity), Real.log_mul (by positivity) hE.ne',
    Real.log_pow, Real.log_pow]
  push_cast
  nlinarith

end OrthoDFA

end
