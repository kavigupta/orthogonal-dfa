import OrthoDFA.Proofs.LoopSimple

/-!
# The learner's DFA: its closed-form length

With `ℓ` the longest probe plus one, `1/ε` is at most `ℓ aε` and `1/τ` and `1/ι` at most `ℓ Θ`, so
the suffix pool is at most `ℓ P₀`, the prefix count at most `ℓ³ R₀`, the budget at most `ℓ K₁`
and `1/cap` at most `ℓ⁵ E₁` (`dfa_rate_bounds`, `dfa_count_bounds`, `dfa_cap_le`). So
`ℓ⁶ (E + E₂) ≤ |Σ|^k` gives the three inequalities `LearnerLoopDFASimple` asks of the probes'
prefix mass (`dfa_conditions`), and `dfa_pow` gets it from `L ≥ L₀`, the walk starting at half
of `L`.
-/

noncomputable section

namespace OrthoDFA

lemma six_log_le {A ℓ : ℝ} (hA : 2 ≤ A) (hℓ : 0 < ℓ) :
    6 * Real.log ℓ ≤ 30 + ℓ * Real.log A / 4 := by
  have hl2 := Real.log_two_gt_d9
  have hlA : Real.log 2 ≤ Real.log A := Real.log_le_log (by norm_num) hA
  have ha : 0 < Real.log 2 / 24 := by positivity
  have h1 := Real.log_le_sub_one_of_pos (mul_pos hℓ ha)
  rw [Real.log_mul hℓ.ne' ha.ne'] at h1
  have h2 : -Real.log (Real.log 2 / 24) ≤ 4 := by
    rw [← Real.log_inv, inv_div, Real.log_le_iff_le_exp (by positivity)]
    have he := Real.exp_one_gt_d9
    have e : Real.exp 4 = Real.exp 1 ^ 4 := by rw [← Real.exp_nat_mul]; norm_num
    rw [e, div_le_iff₀ (by positivity)]
    have h4 : (2.7 : ℝ) ^ 4 ≤ Real.exp 1 ^ 4 := pow_le_pow_left₀ (by norm_num) (by linarith) 4
    nlinarith
  have h3 : ℓ * Real.log 2 ≤ ℓ * Real.log A := mul_le_mul_of_nonneg_left hlA hℓ.le
  have e : ℓ * (Real.log 2 / 24) = ℓ * Real.log 2 / 24 := by ring
  rw [e] at h1
  nlinarith

lemma dfa_pow {A ℓ E L Q : ℝ} {k : ℕ} (hA : 2 ≤ A) (hℓ : ℓ = L + Q + 4)
    (hℓ0 : 0 < ℓ) (hk : L ≤ 2 * k) (hL₀ : Q + 4 + 4 * (Real.log E + 30) / Real.log A ≤ L) :
    ℓ ^ 6 * E ≤ A ^ k := by
  rcases le_or_gt E 0 with hE | hE
  · have : ℓ ^ 6 * E ≤ 0 := mul_nonpos_of_nonneg_of_nonpos (by positivity) hE
    exact this.trans (by positivity)
  have hlA : 0 < Real.log A := Real.log_pos (by linarith)
  have h1 := six_log_le hA hℓ0
  have h2 : 4 * (Real.log E + 30) ≤ (L - Q - 4) * Real.log A := by
    have := sub_le_sub_right hL₀ (Q + 4)
    rw [add_sub_cancel_left, div_le_iff₀ hlA] at this
    linarith
  have h3 : Real.log (ℓ ^ 6 * E) ≤ Real.log (A ^ k) := by
    rw [Real.log_mul (pow_pos hℓ0 6).ne' hE.ne', Real.log_pow, Real.log_pow]
    push_cast
    have : L * Real.log A ≤ 2 * k * Real.log A := mul_le_mul_of_nonneg_right hk hlA.le
    have e1 : ℓ * Real.log A = L * Real.log A + Q * Real.log A + 4 * Real.log A := by
      rw [hℓ]; ring
    have e2 : (L - Q - 4) * Real.log A = L * Real.log A - Q * Real.log A - 4 * Real.log A := by
      ring
    linarith
  exact (Real.log_le_log_iff (by positivity) (by positivity)).1 h3

/-- The rates' inverses grow at most linearly in `ℓ`. -/
lemma dfa_rate_bounds {Q A ℓ N c ξ η₀ ε εcov ι τ s aε Ce Θ : ℝ} (hQ : 1 ≤ Q) (hA : 2 ≤ A)
    (hℓ : 1 ≤ ℓ) (hN1 : 1 ≤ N) (hNℓ : N ≤ ℓ) (hc : 0 < c) (hc1 : c ≤ 1) (hξ : 0 < ξ)
    (hη0 : 0 ≤ η₀) (hη : η₀ < 1 / 2)
    (he : ε = c / (4 * (A + 1) * N)) (hεc : εcov = min ε (ξ / (8 * N)))
    (hι : ι = min (εcov / 8) (ε / (13824 * Q * (Q + 1)))) (hs : s = 1 / 2 - η₀)
    (hτ : τ = min ((εcov / 2) * ((1 - 2 * η₀) / (1 - η₀)) ^ 2) ι)
    (haε : aε = 4 * (A + 1) / c) (hCe : Ce = aε + 8 / ξ)
    (hΘ : Θ = Ce / (2 * s ^ 2) + 8 * Ce + 13824 * Q * (Q + 1) * aε) :
    0 < ε ∧ 1 / ε ≤ aε * ℓ ∧ 0 < ι ∧ 1 / ι ≤ Θ * ℓ ∧ 0 < τ ∧ 1 / τ ≤ Θ * ℓ ∧ 1 ≤ Θ
      ∧ 0 < aε := by
  have hs0 : 0 < s := by rw [hs]; linarith
  have hs1 : s ≤ 1 / 2 := by rw [hs]; linarith
  have hε0 : 0 < ε := by rw [he]; positivity
  have haε0 : 0 < aε := by rw [haε]; positivity
  have hinvε : 1 / ε ≤ aε * ℓ := by
    rw [he, haε, one_div_div, div_le_iff₀ hc]
    have h1 : 4 * (A + 1) * N ≤ 4 * (A + 1) * ℓ := by gcongr
    have e : 4 * (A + 1) / c * ℓ * c = 4 * (A + 1) * ℓ := by field_simp
    rw [e]; exact h1
  have hCe0 : 0 < Ce := by rw [hCe]; positivity
  have haCe : aε ≤ Ce := by
    have : 0 ≤ 8 / ξ := by positivity
    rw [hCe]; linarith
  have h8Ce : 8 / ξ ≤ Ce := by rw [hCe]; linarith
  have hεc0 : 0 < εcov := by rw [hεc]; exact lt_min hε0 (by positivity)
  have hinvεc : 1 / εcov ≤ Ce * ℓ := by
    rw [hεc]
    rcases min_choice ε (ξ / (8 * N)) with h | h <;> rw [h]
    · exact hinvε.trans (mul_le_mul_of_nonneg_right haCe (by linarith))
    · rw [one_div_div]
      calc 8 * N / ξ ≤ 8 * ℓ / ξ := by gcongr
        _ = 8 / ξ * ℓ := by ring
        _ ≤ Ce * ℓ := mul_le_mul_of_nonneg_right h8Ce (by linarith)
  have hQQ : 0 < 13824 * Q * (Q + 1) := by positivity
  have hι0 : 0 < ι := by rw [hι]; exact lt_min (by positivity) (by positivity)
  have hΘ1 : 8 * Ce + 13824 * Q * (Q + 1) * aε ≤ Θ := by
    rw [hΘ]; have : 0 ≤ Ce / (2 * s ^ 2) := by positivity
    linarith
  have hinvι : 1 / ι ≤ Θ * ℓ := by
    rw [hι]
    rcases min_choice (εcov / 8) (ε / (13824 * Q * (Q + 1))) with h | h <;> rw [h]
    · rw [one_div_div]
      have h1 : 8 / εcov = 8 * (1 / εcov) := by ring
      have h2 : 8 * (1 / εcov) ≤ 8 * (Ce * ℓ) := by linarith
      have h3 : 8 * (Ce * ℓ) ≤ Θ * ℓ := by
        have : 8 * Ce ≤ Θ := by
          have : 0 ≤ 13824 * Q * (Q + 1) * aε := by positivity
          linarith
        nlinarith
      linarith
    · rw [one_div_div]
      have h1 : 13824 * Q * (Q + 1) / ε = 13824 * Q * (Q + 1) * (1 / ε) := by ring
      have h2 : 13824 * Q * (Q + 1) * (1 / ε) ≤ 13824 * Q * (Q + 1) * (aε * ℓ) :=
        mul_le_mul_of_nonneg_left hinvε hQQ.le
      have h3 : 13824 * Q * (Q + 1) * (aε * ℓ) ≤ Θ * ℓ := by
        have : 13824 * Q * (Q + 1) * aε ≤ Θ := by
          have : 0 ≤ 8 * Ce := by positivity
          linarith
        have e : 13824 * Q * (Q + 1) * (aε * ℓ) = 13824 * Q * (Q + 1) * aε * ℓ := by ring
        rw [e]
        exact mul_le_mul_of_nonneg_right this (by linarith)
      linarith
  have ha : 2 * s ≤ (1 - 2 * η₀) / (1 - η₀) := by
    rw [le_div_iff₀ (by linarith), hs]
    have := mul_le_mul_of_nonneg_left (show 1 - η₀ ≤ 1 by linarith)
      (show (0 : ℝ) ≤ 1 - 2 * η₀ by linarith)
    linarith
  have hτ0 : 0 < τ := by
    rw [hτ]
    exact lt_min (mul_pos (by positivity) (pow_pos (div_pos (by linarith) (by linarith)) 2)) hι0
  have hΘ2 : Ce / (2 * s ^ 2) ≤ Θ := by
    rw [hΘ]; have : 0 ≤ 8 * Ce + 13824 * Q * (Q + 1) * aε := by positivity
    linarith
  have hinvτ : 1 / τ ≤ Θ * ℓ := by
    rw [hτ]
    rcases min_choice ((εcov / 2) * ((1 - 2 * η₀) / (1 - η₀)) ^ 2) ι with h | h <;> rw [h]
    · have hsq : (2 * s) ^ 2 ≤ ((1 - 2 * η₀) / (1 - η₀)) ^ 2 :=
        pow_le_pow_left₀ (by positivity) ha 2
      have hlow : 2 * s ^ 2 * εcov ≤ (εcov / 2) * ((1 - 2 * η₀) / (1 - η₀)) ^ 2 := by
        have := mul_le_mul_of_nonneg_left hsq (show 0 ≤ εcov / 2 by positivity)
        nlinarith
      calc 1 / ((εcov / 2) * ((1 - 2 * η₀) / (1 - η₀)) ^ 2) ≤ 1 / (2 * s ^ 2 * εcov) :=
            one_div_le_one_div_of_le (by positivity) hlow
        _ = 1 / (2 * s ^ 2) * (1 / εcov) := by field_simp
        _ ≤ 1 / (2 * s ^ 2) * (Ce * ℓ) :=
            mul_le_mul_of_nonneg_left hinvεc (by positivity)
        _ = Ce / (2 * s ^ 2) * ℓ := by field_simp
        _ ≤ Θ * ℓ := mul_le_mul_of_nonneg_right hΘ2 (by linarith)
    · exact hinvι
  have hΘge : 1 ≤ Θ := by
    have hc4 : c ≤ 4 * (A + 1) := by linarith
    have h1 : 1 ≤ 4 * (A + 1) / c := (one_le_div hc).2 hc4
    have hCe1 : 1 ≤ Ce := by
      rw [hCe, haε]; exact h1.trans (le_add_of_nonneg_right (by positivity))
    calc (1 : ℝ) ≤ 8 * Ce := by linarith
      _ ≤ 8 * Ce + 13824 * Q * (Q + 1) * aε := le_add_of_nonneg_right (by positivity)
      _ ≤ Θ := hΘ1
  exact ⟨hε0, hinvε, hι0, hinvι, hτ0, hinvτ, hΘge, haε0⟩

lemma log_le_mul_of_le {x y X ℓ : ℝ} (hx : 0 < x) (hℓ : 1 ≤ ℓ)
    (hxy : x ≤ y * ℓ ^ 2) (hy : 0 < y) (hXy : Real.log y ≤ X) :
    Real.log x ≤ X + 2 * Real.log ℓ := by
  have hℓ0 : 0 < ℓ := by linarith
  calc Real.log x ≤ Real.log (y * ℓ ^ 2) := Real.log_le_log hx hxy
    _ = Real.log y + 2 * Real.log ℓ := by
        rw [Real.log_mul hy.ne' (by positivity), Real.log_pow]; push_cast; ring
    _ ≤ X + 2 * Real.log ℓ := by linarith

lemma add_log_le {X ℓ c : ℝ} (hX : 0 ≤ X) (hℓ : 1 ≤ ℓ) (hc : 0 ≤ c) :
    X + c * Real.log ℓ ≤ (X + c) * ℓ := by
  have hlog : Real.log ℓ ≤ ℓ := (Real.log_le_sub_one_of_pos (by linarith)).trans (by linarith)
  have h1 : X ≤ X * ℓ := le_mul_of_one_le_right hX hℓ
  have h2 : c * Real.log ℓ ≤ c * ℓ := mul_le_mul_of_nonneg_left hlog hc
  linarith

lemma budget_le {S ε aε ℓ X : ℝ} (hS : 0 ≤ S) (hε : 0 < ε) (hinv : 1 / ε ≤ aε * ℓ)
    (hℓ : 1 ≤ ℓ) (hX : 0 ≤ X) :
    36 * S / ε * X + S + 1 ≤ (36 * S * aε * X + S + 1) * ℓ := by
  have h1 : 36 * S / ε * X ≤ 36 * S * aε * X * ℓ := by
    have e : 36 * S / ε * X = 36 * S * X * (1 / ε) := by ring
    rw [e]
    have := mul_le_mul_of_nonneg_left hinv (show 0 ≤ 36 * S * X by positivity)
    linarith
  have h2 : S + 1 ≤ (S + 1) * ℓ := le_mul_of_one_le_right (by positivity) hℓ
  linarith

lemma far_le {B K Q c Am ε aε ℓ : ℝ} (hB : B ≤ K * ℓ) (hB0 : 0 ≤ B) (hQ : 0 ≤ Q) (hc : 0 ≤ c)
    (hAm : 0 ≤ Am) (hε : 0 < ε) (hinv : 1 / ε ≤ aε * ℓ) (hℓ : 1 ≤ ℓ) :
    B * ℓ * Q + c + (Q + 1) * Am / ε ≤ (K * Q + c + (Q + 1) * Am * aε) * ℓ ^ 2 := by
  have hℓ0 : 0 ≤ ℓ := by linarith
  have hℓ2 : ℓ ≤ ℓ ^ 2 := by nlinarith
  have h1 : B * ℓ * Q ≤ K * Q * ℓ ^ 2 := by
    have := mul_le_mul_of_nonneg_right hB (show 0 ≤ ℓ * Q by positivity)
    have e : K * ℓ * (ℓ * Q) = K * Q * ℓ ^ 2 := by ring
    linarith
  have h2 : c ≤ c * ℓ ^ 2 := le_mul_of_one_le_right hc (one_le_pow₀ hℓ)
  have h3 : (Q + 1) * Am / ε ≤ (Q + 1) * Am * aε * ℓ ^ 2 := by
    have e : (Q + 1) * Am / ε = (Q + 1) * Am * (1 / ε) := by ring
    rw [e]
    have h4 := mul_le_mul_of_nonneg_left hinv (show (0 : ℝ) ≤ (Q + 1) * Am by positivity)
    have haε : 0 ≤ aε := by
      have : 0 < 1 / ε := by positivity
      by_contra h; push Not at h; nlinarith
    have h5 : (Q + 1) * Am * (aε * ℓ) ≤ (Q + 1) * Am * aε * ℓ ^ 2 := by
      have := mul_le_mul_of_nonneg_left hℓ2 (show (0 : ℝ) ≤ (Q + 1) * Am * aε by positivity)
      linarith
    linarith
  have e : (K * Q + c + (Q + 1) * Am * aε) * ℓ ^ 2
      = K * Q * ℓ ^ 2 + c * ℓ ^ 2 + (Q + 1) * Am * aε * ℓ ^ 2 := by ring
  linarith

lemma pb_le {Q τ Θ ℓ W W₀ δ' s pAP : ℝ} (hQ : 0 ≤ Q) (hτ0 : 0 < τ) (hτ1 : τ ≤ 1)
    (hinvτ : 1 / τ ≤ Θ * ℓ) (hΘ : 1 ≤ Θ) (hℓ : 1 ≤ ℓ) (hW1 : 1 ≤ W) (hW₀1 : 1 ≤ W₀)
    (hWW : W ≤ W₀ * ℓ ^ 2)
    (hδ'0 : 0 < δ') (hδ'1 : δ' ≤ 1) (hs0 : 0 < s) (hs1 : s ≤ 1 / 2) (hp : 0 < pAP)
    (hp1 : pAP ≤ 1) :
    1 ≤ 256 * (Real.log (2 / τ) + 4 * Real.log (8 * W / δ')) / (s ^ 2 * pAP)
        + 64 * Real.log ((Q + 2 + 2) / (δ' / 2)) / pAP ^ 2
      ∧ 256 * (Real.log (2 / τ) + 4 * Real.log (8 * W / δ')) / (s ^ 2 * pAP)
        + 64 * Real.log ((Q + 2 + 2) / (δ' / 2)) / pAP ^ 2
      ≤ (256 * (Real.log (2 * Θ) + 4 * Real.log (8 * W₀ / δ') + 9) / (s ^ 2 * pAP)
        + 64 * Real.log ((Q + 4) / (δ' / 2)) / pAP ^ 2) * ℓ := by
  have hℓ0 : 0 < ℓ := by linarith
  have hW₀ : 0 < W₀ := by linarith
  have hQ4 : Q + 2 + 2 = Q + 4 := by ring
  rw [hQ4]
  have hZ : 0 ≤ 64 * Real.log ((Q + 4) / (δ' / 2)) / pAP ^ 2 :=
    div_nonneg (mul_nonneg (by norm_num) (Real.log_nonneg (by
      rw [le_div_iff₀ (by positivity)]; linarith))) (by positivity)
  have hl2 := Real.log_two_gt_d9
  have h2τ : Real.log 2 ≤ Real.log (2 / τ) :=
    Real.log_le_log (by norm_num) (by rw [le_div_iff₀ hτ0]; linarith)
  have h8W : 0 ≤ Real.log (8 * W / δ') :=
    Real.log_nonneg (by rw [le_div_iff₀ hδ'0]; linarith)
  have hsp : s ^ 2 * pAP ≤ 1 / 4 := by
    have h1 : s ^ 2 ≤ 1 / 4 := by nlinarith
    have h2 : s ^ 2 * pAP ≤ s ^ 2 * 1 := mul_le_mul_of_nonneg_left hp1 (by positivity)
    linarith
  have hsp0 : 0 < s ^ 2 * pAP := by positivity
  constructor
  · have h1 : 1 ≤ 256 * (Real.log (2 / τ) + 4 * Real.log (8 * W / δ')) / (s ^ 2 * pAP) := by
      rw [le_div_iff₀ hsp0]
      nlinarith
    linarith
  have hτ2 : Real.log (2 / τ) ≤ Real.log (2 * Θ) + Real.log ℓ := by
    have h : 2 / τ ≤ 2 * Θ * ℓ := by
      have e : 2 / τ = 2 * (1 / τ) := by ring
      rw [e]; linarith
    calc Real.log (2 / τ) ≤ Real.log (2 * Θ * ℓ) := Real.log_le_log (by positivity) h
      _ = Real.log (2 * Θ) + Real.log ℓ := Real.log_mul (by positivity) hℓ0.ne'
  have hW2 : Real.log (8 * W / δ') ≤ Real.log (8 * W₀ / δ') + 2 * Real.log ℓ :=
    log_le_mul_of_le (by positivity) hℓ
      (by rw [show 8 * W₀ / δ' * ℓ ^ 2 = 8 * (W₀ * ℓ ^ 2) / δ' by ring]; gcongr) (by positivity)
      le_rfl
  have hX : 0 ≤ Real.log (2 * Θ) + 4 * Real.log (8 * W₀ / δ') := by
    have : 0 ≤ Real.log (2 * Θ) := Real.log_nonneg (by linarith)
    have : 0 ≤ Real.log (8 * W₀ / δ') :=
      Real.log_nonneg (by rw [le_div_iff₀ hδ'0]; linarith)
    linarith
  have hsum : Real.log (2 / τ) + 4 * Real.log (8 * W / δ')
      ≤ (Real.log (2 * Θ) + 4 * Real.log (8 * W₀ / δ') + 9) * ℓ := by
    have := add_log_le hX hℓ (show (0 : ℝ) ≤ 9 by norm_num)
    linarith
  have hterm : 256 * (Real.log (2 / τ) + 4 * Real.log (8 * W / δ')) / (s ^ 2 * pAP)
      ≤ 256 * (Real.log (2 * Θ) + 4 * Real.log (8 * W₀ / δ') + 9) / (s ^ 2 * pAP) * ℓ := by
    rw [div_mul_eq_mul_div, mul_assoc]
    gcongr
  have hZℓ : 64 * Real.log ((Q + 4) / (δ' / 2)) / pAP ^ 2
      ≤ 64 * Real.log ((Q + 4) / (δ' / 2)) / pAP ^ 2 * ℓ := le_mul_of_one_le_right hZ hℓ
  linarith

lemma rb_le {Q τ Θ ℓ PB P₀ δ' s pAP : ℝ} (hQ : 0 ≤ Q) (hτ0 : 0 < τ) (hτ1 : τ ≤ 1)
    (hinvτ : 1 / τ ≤ Θ * ℓ) (hΘ : 1 ≤ Θ) (hℓ : 1 ≤ ℓ) (hPB1 : 1 ≤ PB) (hPB : PB ≤ P₀ * ℓ)
    (hP₀ : 1 ≤ P₀) (hδ'0 : 0 < δ') (hδ'1 : δ' ≤ 1) (hs0 : 0 < s) (hp : 0 < pAP)
    (hp1 : pAP ≤ 1) :
    0 ≤ 60000 * (Q + 2) ^ 2 * Real.log ((Q + 2 + 2) * PB / (δ' / 2 * (δ' / 4) * (pAP / 2) * τ))
        / (s ^ 4 * τ ^ 2)
      ∧ 60000 * (Q + 2) ^ 2 * Real.log ((Q + 2 + 2) * PB / (δ' / 2 * (δ' / 4) * (pAP / 2) * τ))
        / (s ^ 4 * τ ^ 2)
      ≤ 60000 * (Q + 2) ^ 2 * Θ ^ 2
        * (Real.log (16 * (Q + 4) * P₀ * Θ / (δ' ^ 2 * pAP)) + 2) / s ^ 4 * ℓ ^ 3 := by
  have hℓ0 : 0 < ℓ := by linarith
  have hD : δ' / 2 * (δ' / 4) * (pAP / 2) * τ = δ' ^ 2 * pAP * τ / 16 := by ring
  rw [hD, show Q + 2 + 2 = Q + 4 by ring]
  have hD0 : 0 < δ' ^ 2 * pAP * τ / 16 := by positivity
  have hD1 : δ' ^ 2 * pAP * τ / 16 ≤ 1 / 16 := by
    have h1 : δ' ^ 2 ≤ 1 := by nlinarith
    have h2 : δ' ^ 2 * pAP ≤ 1 := by nlinarith
    have h3 : δ' ^ 2 * pAP * τ ≤ 1 := by nlinarith [mul_nonneg (sq_nonneg δ') hp.le]
    linarith
  have harg1 : 1 ≤ (Q + 4) * PB / (δ' ^ 2 * pAP * τ / 16) := by
    rw [le_div_iff₀ hD0]
    nlinarith
  set y := 16 * (Q + 4) * P₀ * Θ / (δ' ^ 2 * pAP) with hy
  have hy1 : 1 ≤ y := by
    rw [hy, le_div_iff₀ (by positivity)]
    have h1 : δ' ^ 2 * pAP ≤ 1 := by nlinarith
    have h2 : 1 ≤ (Q + 4) * P₀ * Θ := by
      have : 1 ≤ (Q + 4) * P₀ := by nlinarith
      nlinarith
    nlinarith
  have hargy : (Q + 4) * PB / (δ' ^ 2 * pAP * τ / 16) ≤ y * ℓ ^ 2 := by
    have e : (Q + 4) * PB / (δ' ^ 2 * pAP * τ / 16) = 16 * (Q + 4) * PB / (δ' ^ 2 * pAP) * (1 / τ) := by
      field_simp
    rw [e, hy]
    have h1 : 16 * (Q + 4) * PB / (δ' ^ 2 * pAP) ≤ 16 * (Q + 4) * (P₀ * ℓ) / (δ' ^ 2 * pAP) := by
      gcongr
    calc 16 * (Q + 4) * PB / (δ' ^ 2 * pAP) * (1 / τ)
        ≤ 16 * (Q + 4) * (P₀ * ℓ) / (δ' ^ 2 * pAP) * (Θ * ℓ) := by
          gcongr
      _ = 16 * (Q + 4) * P₀ * Θ / (δ' ^ 2 * pAP) * ℓ ^ 2 := by ring
  have hlog := log_le_mul_of_le (by linarith) hℓ hargy (by linarith) le_rfl
  have hY : 0 ≤ Real.log y := Real.log_nonneg hy1
  have hlog' : Real.log ((Q + 4) * PB / (δ' ^ 2 * pAP * τ / 16)) ≤ (Real.log y + 2) * ℓ :=
    hlog.trans (add_log_le hY hℓ (by norm_num))
  have hlog0 : 0 ≤ Real.log ((Q + 4) * PB / (δ' ^ 2 * pAP * τ / 16)) := Real.log_nonneg harg1
  have hinv2 : 1 / τ ^ 2 ≤ Θ ^ 2 * ℓ ^ 2 := by
    rw [show 1 / τ ^ 2 = (1 / τ) ^ 2 by ring, show Θ ^ 2 * ℓ ^ 2 = (Θ * ℓ) ^ 2 by ring]
    exact pow_le_pow_left₀ (by positivity) hinvτ 2
  constructor
  · positivity
  · have e : 60000 * (Q + 2) ^ 2 * Real.log ((Q + 4) * PB / (δ' ^ 2 * pAP * τ / 16))
        / (s ^ 4 * τ ^ 2)
        = 60000 * (Q + 2) ^ 2 / s ^ 4 * (Real.log ((Q + 4) * PB / (δ' ^ 2 * pAP * τ / 16))
          * (1 / τ ^ 2)) := by field_simp
    rw [e]
    have e2 : 60000 * (Q + 2) ^ 2 * Θ ^ 2 * (Real.log y + 2) / s ^ 4 * ℓ ^ 3
        = 60000 * (Q + 2) ^ 2 / s ^ 4 * ((Real.log y + 2) * ℓ * (Θ ^ 2 * ℓ ^ 2)) := by ring
    rw [e2]
    gcongr

lemma cap_ge {Q τ ι Θ ℓ RB R₀ PB P₀ δ' pAP E₁ : ℝ} (hQ : 0 ≤ Q) (hτ0 : 0 < τ)
    (hinvτ : 1 / τ ≤ Θ * ℓ) (hι0 : 0 < ι) (hinvι : 1 / ι ≤ Θ * ℓ) (hΘ : 1 ≤ Θ) (hℓ : 1 ≤ ℓ)
    (hRB0 : 0 ≤ RB) (hRB : RB ≤ R₀ * ℓ ^ 3) (hPB0 : 0 ≤ PB) (hPB : PB ≤ P₀ * ℓ)
    (hP₀ : 0 ≤ P₀) (hR₀ : 0 ≤ R₀) (hδ'0 : 0 < δ') (hδ'1 : δ' ≤ 1) (hp : 0 < pAP)
    (hp1 : pAP ≤ 1)
    (hE₁ : E₁ = 2 ^ 23 * (Q + 3) ^ 2 * (Θ ^ 2 * (R₀ + 1) + (P₀ + 2) / pAP) / δ') :
    1 / (E₁ * ℓ ^ 5) ≤ min (min (δ' * τ ^ 2 / (2 ^ 22 * (Q + 2 + 1) ^ 2 * (RB + 1)))
        (δ' * ι ^ 2 / (2 ^ 11 * (Q + 2 + 1) * (RB + 1))))
      (min (min (δ' * pAP / (2 ^ 15 * (Q + 2 + 1) * (PB + 1)))
          (ι / (12 * (Q + 2) ^ 2 * (RB + 1))))
        (min (pAP / (2 * (Q + 2) * (PB + 2))) (1 / (6 * (Q + 2) * (RB + 1))))) := by
  have hℓ0 : 0 < ℓ := by linarith
  rw [show Q + 2 + 1 = Q + 3 by ring]
  set U := Θ ^ 2 * (R₀ + 1) + (P₀ + 2) / pAP with hU
  have hU1 : Θ ^ 2 * (R₀ + 1) ≤ U := by
    have : 0 ≤ (P₀ + 2) / pAP := by positivity
    rw [hU]; linarith
  have hU2 : (P₀ + 2) / pAP ≤ U := by
    have : 0 ≤ Θ ^ 2 * (R₀ + 1) := by positivity
    rw [hU]; linarith
  have hE₁' : δ' * E₁ = 2 ^ 23 * (Q + 3) ^ 2 * U := by rw [hE₁]; field_simp
  have hE₁0 : 0 < E₁ := by rw [hE₁]; positivity
  have hτΘ : 1 ≤ τ * Θ * ℓ := by
    have := mul_le_mul_of_nonneg_left hinvτ hτ0.le
    rw [mul_one_div_cancel hτ0.ne'] at this
    linarith
  have hιΘ : 1 ≤ ι * Θ * ℓ := by
    have := mul_le_mul_of_nonneg_left hinvι hι0.le
    rw [mul_one_div_cancel hι0.ne'] at this
    linarith
  have hℓ3 : 1 ≤ ℓ ^ 3 := one_le_pow₀ hℓ
  have hℓ5 : ℓ ^ 3 ≤ ℓ ^ 5 := pow_le_pow_right₀ hℓ (by norm_num)
  have hℓ1 : ℓ ≤ ℓ ^ 5 := le_self_pow₀ hℓ (by norm_num)
  have hRB1 : RB + 1 ≤ (R₀ + 1) * ℓ ^ 3 := by nlinarith
  have hPB1 : PB + 2 ≤ (P₀ + 2) * ℓ := by nlinarith
  have hΘ2 : 1 ≤ Θ ^ 2 := one_le_pow₀ hΘ
  have hQ3 : (1 : ℝ) ≤ (Q + 3) := by linarith
  have hD : 0 < E₁ * ℓ ^ 5 := by positivity
  have hEℓ : ∀ c : ℝ, c * (E₁ * ℓ ^ 5) = 2 ^ 23 * (Q + 3) ^ 2 * U * c * ℓ ^ 5 / δ' := by
    intro c
    rw [show E₁ = 2 ^ 23 * (Q + 3) ^ 2 * U / δ' by rw [← hE₁']; field_simp]
    ring
  have hQQ : Q + 3 ≤ (Q + 3) ^ 2 := by nlinarith
  have hQ2 : Q + 2 ≤ (Q + 3) ^ 2 := by nlinarith
  have hU0 : 0 ≤ U := (by positivity : (0 : ℝ) ≤ Θ ^ 2 * (R₀ + 1)).trans hU1
  have hR1U : R₀ + 1 ≤ U := (le_mul_of_one_le_left (by positivity) hΘ2).trans hU1
  have hΘU : Θ * (R₀ + 1) ≤ U := (mul_le_mul_of_nonneg_right
    (le_self_pow₀ hΘ (by norm_num) : Θ ≤ Θ ^ 2) (by positivity)).trans hU1
  have hPU : P₀ + 2 ≤ U * pAP := by
    have := mul_le_mul_of_nonneg_right hU2 hp.le
    rwa [div_mul_cancel₀ _ hp.ne'] at this
  refine le_min (le_min ?_ ?_) (le_min (le_min ?_ ?_) (le_min ?_ ?_))
  · rw [div_le_div_iff₀ hD (by positivity), one_mul]
    have h1 : 1 ≤ (τ * Θ * ℓ) ^ 2 := one_le_pow₀ hτΘ
    have a1 : 2 ^ 22 * (Q + 3) ^ 2 * (RB + 1) ≤ 2 ^ 22 * (Q + 3) ^ 2 * ((R₀ + 1) * ℓ ^ 3) :=
      mul_le_mul_of_nonneg_left hRB1 (by positivity)
    have a2 : 2 ^ 22 * (Q + 3) ^ 2 * ((R₀ + 1) * ℓ ^ 3)
        ≤ 2 ^ 22 * (Q + 3) ^ 2 * ((R₀ + 1) * ℓ ^ 3) * (τ * Θ * ℓ) ^ 2 :=
      le_mul_of_one_le_right (by positivity) h1
    have a3 : 2 ^ 22 * (Q + 3) ^ 2 * ((R₀ + 1) * ℓ ^ 3) * (τ * Θ * ℓ) ^ 2
        = 2 ^ 22 * (Q + 3) ^ 2 * (Θ ^ 2 * (R₀ + 1)) * τ ^ 2 * ℓ ^ 5 := by ring
    have a4 : 2 ^ 22 * (Q + 3) ^ 2 * (Θ ^ 2 * (R₀ + 1)) * τ ^ 2 * ℓ ^ 5
        ≤ 2 ^ 22 * (Q + 3) ^ 2 * U * τ ^ 2 * ℓ ^ 5 := by gcongr
    have a5 : 0 ≤ 2 ^ 22 * (Q + 3) ^ 2 * U * τ ^ 2 * ℓ ^ 5 := by positivity
    have a6 : δ' * τ ^ 2 * (E₁ * ℓ ^ 5) = 2 ^ 23 * (Q + 3) ^ 2 * U * τ ^ 2 * ℓ ^ 5 := by
      rw [show δ' * τ ^ 2 * (E₁ * ℓ ^ 5) = (δ' * E₁) * τ ^ 2 * ℓ ^ 5 by ring, hE₁']
    have a7 : 2 ^ 23 * (Q + 3) ^ 2 * U * τ ^ 2 * ℓ ^ 5
        = 2 * (2 ^ 22 * (Q + 3) ^ 2 * U * τ ^ 2 * ℓ ^ 5) := by ring
    linarith
  · rw [div_le_div_iff₀ hD (by positivity), one_mul]
    have h1 : 1 ≤ (ι * Θ * ℓ) ^ 2 := one_le_pow₀ hιΘ
    have a1 : 2 ^ 11 * (Q + 3) * (RB + 1) ≤ 2 ^ 11 * (Q + 3) ^ 2 * ((R₀ + 1) * ℓ ^ 3) :=
      mul_le_mul (mul_le_mul_of_nonneg_left hQQ (by norm_num)) hRB1 (by positivity) (by positivity)
    have a2 : 2 ^ 11 * (Q + 3) ^ 2 * ((R₀ + 1) * ℓ ^ 3)
        ≤ 2 ^ 11 * (Q + 3) ^ 2 * ((R₀ + 1) * ℓ ^ 3) * (ι * Θ * ℓ) ^ 2 :=
      le_mul_of_one_le_right (by positivity) h1
    have a3 : 2 ^ 11 * (Q + 3) ^ 2 * ((R₀ + 1) * ℓ ^ 3) * (ι * Θ * ℓ) ^ 2
        = 2 ^ 11 * (Q + 3) ^ 2 * (Θ ^ 2 * (R₀ + 1)) * ι ^ 2 * ℓ ^ 5 := by ring
    have a4 : 2 ^ 11 * (Q + 3) ^ 2 * (Θ ^ 2 * (R₀ + 1)) * ι ^ 2 * ℓ ^ 5
        ≤ 2 ^ 11 * (Q + 3) ^ 2 * U * ι ^ 2 * ℓ ^ 5 := by gcongr
    have a5 : 0 ≤ 2 ^ 11 * (Q + 3) ^ 2 * U * ι ^ 2 * ℓ ^ 5 := by positivity
    have a6 : δ' * ι ^ 2 * (E₁ * ℓ ^ 5) = 2 ^ 23 * (Q + 3) ^ 2 * U * ι ^ 2 * ℓ ^ 5 := by
      rw [show δ' * ι ^ 2 * (E₁ * ℓ ^ 5) = (δ' * E₁) * ι ^ 2 * ℓ ^ 5 by ring, hE₁']
    have a7 : 2 ^ 23 * (Q + 3) ^ 2 * U * ι ^ 2 * ℓ ^ 5
        = 2 ^ 12 * (2 ^ 11 * (Q + 3) ^ 2 * U * ι ^ 2 * ℓ ^ 5) := by ring
    linarith
  · rw [div_le_div_iff₀ hD (by positivity), one_mul]
    have h4 : PB + 1 ≤ (P₀ + 2) * ℓ ^ 5 :=
      (by linarith : PB + 1 ≤ (P₀ + 2) * ℓ).trans (mul_le_mul_of_nonneg_left hℓ1 (by positivity))
    have a1 : 2 ^ 15 * (Q + 3) * (PB + 1) ≤ 2 ^ 15 * (Q + 3) ^ 2 * ((P₀ + 2) * ℓ ^ 5) :=
      mul_le_mul (mul_le_mul_of_nonneg_left hQQ (by norm_num)) h4 (by positivity) (by positivity)
    have a2 : 2 ^ 15 * (Q + 3) ^ 2 * ((P₀ + 2) * ℓ ^ 5)
        ≤ 2 ^ 15 * (Q + 3) ^ 2 * (U * pAP * ℓ ^ 5) := by gcongr
    have a5 : 0 ≤ 2 ^ 15 * (Q + 3) ^ 2 * (U * pAP * ℓ ^ 5) := by positivity
    have a6 : δ' * pAP * (E₁ * ℓ ^ 5) = 2 ^ 23 * (Q + 3) ^ 2 * U * pAP * ℓ ^ 5 := by
      rw [show δ' * pAP * (E₁ * ℓ ^ 5) = (δ' * E₁) * pAP * ℓ ^ 5 by ring, hE₁']
    have a7 : 2 ^ 23 * (Q + 3) ^ 2 * U * pAP * ℓ ^ 5
        = 2 ^ 8 * (2 ^ 15 * (Q + 3) ^ 2 * (U * pAP * ℓ ^ 5)) := by ring
    linarith
  · rw [div_le_div_iff₀ hD (by positivity), one_mul, hEℓ]
    rw [le_div_iff₀ hδ'0]
    have h4 : 1 ≤ ι * Θ * ℓ := hιΘ
    have a1 : 12 * (Q + 2) ^ 2 * (RB + 1) ≤ 12 * (Q + 3) ^ 2 * ((R₀ + 1) * ℓ ^ 3) :=
      mul_le_mul (mul_le_mul_of_nonneg_left (pow_le_pow_left₀ (by linarith) (by linarith) 2)
        (by norm_num)) hRB1 (by positivity) (by positivity)
    have a2 : 12 * (Q + 3) ^ 2 * ((R₀ + 1) * ℓ ^ 3)
        ≤ 12 * (Q + 3) ^ 2 * ((R₀ + 1) * ℓ ^ 3) * (ι * Θ * ℓ) :=
      le_mul_of_one_le_right (by positivity) h4
    have a3 : 12 * (Q + 3) ^ 2 * ((R₀ + 1) * ℓ ^ 3) * (ι * Θ * ℓ)
        = 12 * (Q + 3) ^ 2 * (Θ * (R₀ + 1)) * ι * ℓ ^ 4 := by ring
    have a4 : 12 * (Q + 3) ^ 2 * (Θ * (R₀ + 1)) * ι * ℓ ^ 4
        ≤ 12 * (Q + 3) ^ 2 * U * ι * ℓ ^ 5 :=
      mul_le_mul (mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left hΘU (by positivity))
        hι0.le) (pow_le_pow_right₀ hℓ (by norm_num)) (by positivity) (by positivity)
    have a5 : 12 * (Q + 3) ^ 2 * U * ι * ℓ ^ 5 * δ' ≤ 12 * (Q + 3) ^ 2 * U * ι * ℓ ^ 5 :=
      mul_le_of_le_one_right (by positivity) hδ'1
    have a6 : 0 ≤ 12 * (Q + 3) ^ 2 * U * ι * ℓ ^ 5 := by positivity
    have a7 : 2 ^ 23 * (Q + 3) ^ 2 * U * ι * ℓ ^ 5
        = 2 ^ 23 / 12 * (12 * (Q + 3) ^ 2 * U * ι * ℓ ^ 5) := by ring
    have a8 : 12 * (Q + 2) ^ 2 * (RB + 1) * δ' ≤ 12 * (Q + 2) ^ 2 * (RB + 1) :=
      mul_le_of_le_one_right (by positivity) hδ'1
    linarith
  · rw [div_le_div_iff₀ hD (by positivity), one_mul, hEℓ]
    rw [le_div_iff₀ hδ'0]
    have h4 : PB + 2 ≤ (P₀ + 2) * ℓ ^ 5 :=
      hPB1.trans (mul_le_mul_of_nonneg_left hℓ1 (by positivity))
    have a1 : 2 * (Q + 2) * (PB + 2) ≤ 2 * (Q + 3) ^ 2 * ((P₀ + 2) * ℓ ^ 5) :=
      mul_le_mul (mul_le_mul_of_nonneg_left hQ2 (by norm_num)) h4 (by positivity) (by positivity)
    have a2 : 2 * (Q + 3) ^ 2 * ((P₀ + 2) * ℓ ^ 5) ≤ 2 * (Q + 3) ^ 2 * (U * pAP * ℓ ^ 5) := by
      gcongr
    have a6 : 0 ≤ 2 * (Q + 3) ^ 2 * (U * pAP * ℓ ^ 5) := by positivity
    have a7 : 2 ^ 23 * (Q + 3) ^ 2 * U * pAP * ℓ ^ 5
        = 2 ^ 22 * (2 * (Q + 3) ^ 2 * (U * pAP * ℓ ^ 5)) := by ring
    have a8 : 2 * (Q + 2) * (PB + 2) * δ' ≤ 2 * (Q + 2) * (PB + 2) :=
      mul_le_of_le_one_right (by positivity) hδ'1
    linarith
  · rw [div_le_div_iff₀ hD (by positivity), one_mul, hEℓ]
    rw [le_div_iff₀ hδ'0]
    have h4 : RB + 1 ≤ (R₀ + 1) * ℓ ^ 5 :=
      hRB1.trans (mul_le_mul_of_nonneg_left hℓ5 (by positivity))
    have a1 : 6 * (Q + 2) * (RB + 1) ≤ 6 * (Q + 3) ^ 2 * ((R₀ + 1) * ℓ ^ 5) :=
      mul_le_mul (mul_le_mul_of_nonneg_left hQ2 (by norm_num)) h4 (by positivity) (by positivity)
    have a2 : 6 * (Q + 3) ^ 2 * ((R₀ + 1) * ℓ ^ 5) ≤ 6 * (Q + 3) ^ 2 * (U * ℓ ^ 5) := by gcongr
    have a6 : 0 ≤ 6 * (Q + 3) ^ 2 * (U * ℓ ^ 5) := by positivity
    have a7 : 2 ^ 23 * (Q + 3) ^ 2 * U * 1 * ℓ ^ 5 = 2 ^ 23 / 6 * (6 * (Q + 3) ^ 2 * (U * ℓ ^ 5)) := by
      ring
    have a8 : 6 * (Q + 2) * (RB + 1) * δ' ≤ 6 * (Q + 2) * (RB + 1) :=
      mul_le_of_le_one_right (by positivity) hδ'1
    linarith

lemma dfa_final {Ak ℓ ε cap aε E₁ E E₂ lg Q n J RB R₀ B₃ K₁ δ : ℝ} (hAk : 0 < Ak)
    (hℓ : 1 ≤ ℓ) (hε0 : 0 < ε) (hinvε : 1 / ε ≤ aε * ℓ) (haε : 0 < aε)
    (hcap : 1 / (E₁ * ℓ ^ 5) ≤ cap) (hE₁ : 0 < E₁) (hlg : 0 ≤ lg)
    (hE : E = 96 * aε * E₁ + 768 * lg * aε)
    (hE₂ : E₂ = 4 * (Q + 1) * n * (6 * (Q + 2) * R₀ + K₁ + n) / δ) (hQ : 0 ≤ Q) (hn : 0 ≤ n)
    (hδ : 0 < δ) (hJ : J = Q + 2) (hRB0 : 0 ≤ RB) (hRB : RB ≤ R₀ * ℓ ^ 3) (hB₃0 : 0 ≤ B₃)
    (hB₃ : B₃ ≤ K₁ * ℓ) (hpow : ℓ ^ 6 * (E + E₂) ≤ Ak) :
    Ak⁻¹ ≤ cap * ε / 96 ∧ 768 * lg * Ak⁻¹ ≤ ε
      ∧ (Q + 1) * n * (6 * J * RB + B₃ + n) * Ak⁻¹ ≤ δ / 4 := by
  have hℓ0 : 0 < ℓ := by linarith
  have hR₀ : 0 ≤ R₀ := hRB0.trans hRB |> fun h => by
    by_contra h'; push Not at h'
    have : R₀ * ℓ ^ 3 < 0 := mul_neg_of_neg_of_pos h' (by positivity)
    linarith
  have hK₁ : 0 ≤ K₁ := by
    by_contra h'; push Not at h'
    have : K₁ * ℓ < 0 := mul_neg_of_neg_of_pos h' hℓ0
    linarith
  have hE₂0 : 0 ≤ E₂ := by rw [hE₂]; positivity
  have hE0 : 0 ≤ E := by rw [hE]; positivity
  have hℓ6 : 1 ≤ ℓ ^ 6 := one_le_pow₀ hℓ
  have hinvA : Ak⁻¹ ≤ (ℓ ^ 6 * (E + E₂))⁻¹ ∨ E + E₂ = 0 := by
    rcases eq_or_lt_of_le (add_nonneg hE0 hE₂0) with h | h
    · exact Or.inr h.symm
    · exact Or.inl (inv_anti₀ (by positivity) hpow)
  have hε : 1 / (aε * ℓ) ≤ ε := by
    rw [div_le_iff₀ (by positivity)]
    rw [div_le_iff₀ hε0] at hinvε
    linarith
  refine ⟨?_, ?_, ?_⟩
  · -- the cap
    have hc : 1 / (96 * aε * E₁ * ℓ ^ 6) ≤ cap * ε / 96 := by
      have h1 : 1 / (96 * aε * E₁ * ℓ ^ 6) = 1 / (E₁ * ℓ ^ 5) * (1 / (aε * ℓ)) / 96 := by
        field_simp
      rw [h1]
      have hcap0 : 0 ≤ cap := (by positivity : (0 : ℝ) ≤ 1 / (E₁ * ℓ ^ 5)).trans hcap
      gcongr
    refine le_trans ?_ hc
    rw [one_div]
    refine inv_anti₀ (by positivity) (le_trans ?_ hpow)
    have : 96 * aε * E₁ ≤ E + E₂ := by
      rw [hE]; have : 0 ≤ 768 * lg * aε := by positivity
      linarith
    have := mul_le_mul_of_nonneg_left this (show 0 ≤ ℓ ^ 6 by positivity)
    linarith
  · -- the prefix mass against the noise's tails
    rcases eq_or_lt_of_le hlg with h0 | hlg0
    · rw [← h0]; simp [hε0.le]
    have h1 : 768 * lg * aε * ℓ ≤ ℓ ^ 6 * (E + E₂) := by
      have : 768 * lg * aε ≤ E + E₂ := by
        rw [hE]; have : 0 ≤ 96 * aε * E₁ := by positivity
        linarith
      have h2 : ℓ ≤ ℓ ^ 6 := le_self_pow₀ hℓ (by norm_num)
      calc 768 * lg * aε * ℓ ≤ (E + E₂) * ℓ ^ 6 :=
            mul_le_mul this h2 hℓ0.le (by positivity)
        _ = ℓ ^ 6 * (E + E₂) := by ring
    have h3 : 768 * lg * Ak⁻¹ ≤ 1 / (aε * ℓ) := by
      rw [le_div_iff₀ (by positivity)]
      have : Ak⁻¹ * Ak = 1 := inv_mul_cancel₀ hAk.ne'
      have h4 : 768 * lg * aε * ℓ ≤ Ak := h1.trans hpow
      calc 768 * lg * Ak⁻¹ * (aε * ℓ) = (768 * lg * aε * ℓ) * Ak⁻¹ := by ring
        _ ≤ Ak * Ak⁻¹ := mul_le_mul_of_nonneg_right h4 (by positivity)
        _ = 1 := mul_inv_cancel₀ hAk.ne'
    exact h3.trans hε
  · -- the held-out strings' collisions
    have h1 : (Q + 1) * n * (6 * J * RB + B₃ + n) ≤ ℓ ^ 3 * (E₂ * δ / 4) := by
      have hℓ3 : 1 ≤ ℓ ^ 3 := one_le_pow₀ hℓ
      have hℓ13 : ℓ ≤ ℓ ^ 3 := le_self_pow₀ hℓ (by norm_num)
      have e : E₂ * δ / 4 = (Q + 1) * n * (6 * (Q + 2) * R₀ + K₁ + n) := by
        rw [hE₂]; field_simp
      rw [e, hJ]
      have a1 : 6 * (Q + 2) * RB ≤ ℓ ^ 3 * (6 * (Q + 2) * R₀) := by
        have := mul_le_mul_of_nonneg_left hRB (show 0 ≤ 6 * (Q + 2) by positivity)
        linarith
      have a2 : B₃ ≤ ℓ ^ 3 * K₁ := by
        have := mul_le_mul_of_nonneg_left hℓ13 hK₁
        linarith
      have a3 : (n : ℝ) ≤ ℓ ^ 3 * n := le_mul_of_one_le_left hn hℓ3
      have h5 : 6 * (Q + 2) * RB + B₃ + n ≤ ℓ ^ 3 * (6 * (Q + 2) * R₀ + K₁ + n) := by linarith
      have := mul_le_mul_of_nonneg_left h5 (show 0 ≤ (Q + 1) * n by positivity)
      linarith
    have h2 : ℓ ^ 3 * (E₂ * δ / 4) ≤ Ak * (δ / 4) := by
      have hℓ36 : ℓ ^ 3 ≤ ℓ ^ 6 := pow_le_pow_right₀ hℓ (by norm_num)
      have : ℓ ^ 3 * E₂ ≤ Ak := by
        have := mul_le_mul_of_nonneg_right hℓ36 hE₂0
        have : ℓ ^ 6 * E₂ ≤ ℓ ^ 6 * (E + E₂) :=
          mul_le_mul_of_nonneg_left (by linarith) (by positivity)
        linarith
      have := mul_le_mul_of_nonneg_right this (show 0 ≤ δ / 4 by positivity)
      calc ℓ ^ 3 * (E₂ * δ / 4) = ℓ ^ 3 * E₂ * (δ / 4) := by ring
        _ ≤ Ak * (δ / 4) := this
    calc (Q + 1) * n * (6 * J * RB + B₃ + n) * Ak⁻¹ ≤ Ak * (δ / 4) * Ak⁻¹ :=
          mul_le_mul_of_nonneg_right (h1.trans h2) (by positivity)
      _ = δ / 4 := by field_simp

lemma log_budget_nonneg {S δ' : ℝ} (hS : 1 ≤ S) (hδ'0 : 0 < δ') (hδ'1 : δ' ≤ 1) :
    0 ≤ Real.log (16 * S * (S + 1) / δ') :=
  Real.log_nonneg (by rw [le_div_iff₀ hδ'0]; nlinarith)

lemma log_count_nonneg {Q P₀ Θ δ' pAP : ℝ} (hQ : 0 ≤ Q) (hP₀ : 1 ≤ P₀) (hΘ : 1 ≤ Θ)
    (hδ'0 : 0 < δ') (hδ'1 : δ' ≤ 1) (hp : 0 < pAP) (hp1 : pAP ≤ 1) :
    0 ≤ Real.log (16 * (Q + 4) * P₀ * Θ / (δ' ^ 2 * pAP)) := by
  refine Real.log_nonneg ?_
  rw [le_div_iff₀ (by positivity)]
  have h1 : δ' ^ 2 * pAP ≤ 1 := by nlinarith
  have h2 : 1 ≤ (Q + 4) * P₀ * Θ := by
    have : 1 ≤ (Q + 4) * P₀ := by nlinarith
    nlinarith
  nlinarith

lemma pool_ge_one {Q s pAP Θ W₀ δ' : ℝ} (hQ : 0 ≤ Q) (hs0 : 0 < s) (hs1 : s ≤ 1 / 2)
    (hp : 0 < pAP) (hp1 : pAP ≤ 1) (hΘ : 1 ≤ Θ) (hW₀ : 1 ≤ W₀) (hδ'0 : 0 < δ')
    (hδ'1 : δ' ≤ 1) :
    1 ≤ 256 * (Real.log (2 * Θ) + 4 * Real.log (8 * W₀ / δ') + 9) / (s ^ 2 * pAP)
      + 64 * Real.log ((Q + 4) / (δ' / 2)) / pAP ^ 2 := by
  have h1 : 0 ≤ Real.log (2 * Θ) := Real.log_nonneg (by linarith)
  have h2 : 0 ≤ Real.log (8 * W₀ / δ') := Real.log_nonneg (by rw [le_div_iff₀ hδ'0]; linarith)
  have h3 : 0 ≤ 64 * Real.log ((Q + 4) / (δ' / 2)) / pAP ^ 2 :=
    div_nonneg (mul_nonneg (by norm_num) (Real.log_nonneg (by
      rw [le_div_iff₀ (by positivity)]; linarith))) (by positivity)
  have hsp : s ^ 2 * pAP ≤ 1 := by
    have : s ^ 2 ≤ 1 := by nlinarith
    nlinarith
  have h4 : 1 ≤ 256 * (Real.log (2 * Θ) + 4 * Real.log (8 * W₀ / δ') + 9) / (s ^ 2 * pAP) := by
    rw [le_div_iff₀ (by positivity)]; nlinarith
  linarith

lemma rate_le_one {c A N : ℝ} (hc1 : c ≤ 1) (hA : 2 ≤ A) (hN : 1 ≤ N) :
    c / (4 * (A + 1) * N) ≤ 1 := by
  rw [div_le_one (by positivity)]
  have : (1 : ℝ) ≤ 4 * (A + 1) * N :=
    one_le_mul_of_one_le_of_one_le (by linarith) hN
  linarith

lemma kc_ge_one {Q A : ℝ} {m e : ℕ} (hQ : 0 ≤ Q) (hA : 0 ≤ A) :
    (1 : ℝ) ≤ (1 + (Q + 1) ^ 3 * A) ^ m * (Q + 3) ^ e := by
  have h1 : (1 : ℝ) ≤ (1 + (Q + 1) ^ 3 * A) ^ m :=
    one_le_pow₀ (by have : 0 ≤ (Q + 1) ^ 3 * A := by positivity
                    linarith)
  have h2 : (1 : ℝ) ≤ (Q + 3) ^ e := one_le_pow₀ (by linarith)
  exact one_le_mul_of_one_le_of_one_le h1 h2

end OrthoDFA






end
