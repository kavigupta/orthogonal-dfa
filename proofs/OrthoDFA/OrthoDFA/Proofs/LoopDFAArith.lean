import Mathlib.Analysis.SpecialFunctions.Log.Base
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

/-- `6 log ℓ` lies under its tangent at `T`, so `L ≥ L₀` gives `6 log ℓ + log E ≤ (L/2) log A`. -/
lemma dfa_pow {A ℓ E L Q T : ℝ} {k : ℕ} (hA : 2 ≤ A) (hℓ : ℓ = L + Q + 4) (hQ : 0 ≤ Q)
    (hL : 0 ≤ L) (hE : 1 ≤ E) (hk : L ≤ 2 * k)
    (hT : T = 2 * (Real.log E + 12) / Real.log A + Q + 4)
    (hL₀ : 2 * (Real.log E + 6 * Real.log T) / Real.log A
      + 144 * (Real.log T - 2) / (Real.log A * (T * Real.log A - 12)) ≤ L) :
    ℓ ^ 6 * E ≤ A ^ k := by
  have ha : 0 < Real.log A := Real.log_pos (by linarith)
  have hx : 0 ≤ Real.log E := Real.log_nonneg hE
  have hTa : T * Real.log A = 2 * (Real.log E + 12) + (Q + 4) * Real.log A := by
    rw [hT]; field_simp; ring
  have hT0 : 0 < T := by rw [hT]; positivity
  have hd : 0 < T * Real.log A - 12 := by nlinarith
  have hℓ0 : 0 < ℓ := by rw [hℓ]; linarith
  have htan : Real.log ℓ ≤ Real.log T + ℓ / T - 1 := by
    have := Real.log_le_sub_one_of_pos (div_pos hℓ0 hT0)
    rw [Real.log_div hℓ0.ne' hT0.ne'] at this
    linarith
  have h1 : 2 * (Real.log E + 6 * Real.log T) * (T * Real.log A - 12)
      + 144 * (Real.log T - 2) ≤ L * Real.log A * (T * Real.log A - 12) := by
    have e : 2 * (Real.log E + 6 * Real.log T) / Real.log A
        + 144 * (Real.log T - 2) / (Real.log A * (T * Real.log A - 12))
        = (2 * (Real.log E + 6 * Real.log T) * (T * Real.log A - 12) + 144 * (Real.log T - 2))
          / (Real.log A * (T * Real.log A - 12)) := by
      rw [eq_div_iff (mul_pos ha hd).ne', add_mul, div_mul_cancel₀ _ (mul_pos ha hd).ne']
      field_simp
    rw [e, div_le_iff₀ (mul_pos ha hd)] at hL₀
    linarith
  have h2 : 2 * T * (Real.log E + 6 * Real.log T - 6) + 12 * (Q + 4)
      ≤ L * (T * Real.log A - 12) := by
    refine le_of_mul_le_mul_left ?_ ha
    linear_combination h1 - 12 * hTa
  have h3 : 6 * Real.log ℓ + Real.log E ≤ L * Real.log A / 2 := by
    have e : ℓ / T * T = ℓ := div_mul_cancel₀ ℓ hT0.ne'
    have h4 : 2 * T * (6 * Real.log ℓ + Real.log E) ≤ T * (L * Real.log A) := by
      nlinarith
    nlinarith
  have h5 : Real.log (ℓ ^ 6 * E) ≤ Real.log (A ^ k) := by
    rw [Real.log_mul (pow_pos hℓ0 6).ne' (by linarith), Real.log_pow, Real.log_pow]
    have : L * Real.log A ≤ 2 * k * Real.log A := mul_le_mul_of_nonneg_right hk ha.le
    push_cast
    linarith
  exact (Real.log_le_log_iff (by positivity) (by positivity)).1 h5

/-- The rates' inverses grow at most linearly in `ℓ`. -/
lemma dfa_rate_bounds {Q A ℓ N c ξ η₀ ε εcov ι τ s aε Ce Θ : ℝ} (hQ : 1 ≤ Q) (hA : 2 ≤ A)
    (hℓ : 1 ≤ ℓ) (hN1 : 1 ≤ N) (hNℓ : N ≤ ℓ) (hc : 0 < c) (hc1 : c ≤ 1) (hξ : 0 < ξ)
    (hη0 : 0 ≤ η₀) (hη : η₀ < 1 / 2)
    (he : ε = c / (4 * (A + 1) * N)) (hεc : εcov = min ε (ξ / (8 * N)))
    (hι : ι = min (εcov / 8) (ε / (4608 * Q * (Q + 1)))) (hs : s = 1 / 2 - η₀)
    (hτ : τ = min ((εcov / 2) * ((1 - 2 * η₀) / (1 - η₀)) ^ 2) ι)
    (haε : aε = 4 * (A + 1) / c) (hCe : Ce = aε + 8 / ξ)
    (hΘ : Θ = Ce / (2 * s ^ 2) + 8 * Ce + 4608 * Q * (Q + 1) * aε) :
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
  have hQQ : 0 < 4608 * Q * (Q + 1) := by positivity
  have hι0 : 0 < ι := by rw [hι]; exact lt_min (by positivity) (by positivity)
  have hΘ1 : 8 * Ce + 4608 * Q * (Q + 1) * aε ≤ Θ := by
    rw [hΘ]; have : 0 ≤ Ce / (2 * s ^ 2) := by positivity
    linarith
  have hinvι : 1 / ι ≤ Θ * ℓ := by
    rw [hι]
    rcases min_choice (εcov / 8) (ε / (4608 * Q * (Q + 1))) with h | h <;> rw [h]
    · rw [one_div_div]
      have h1 : 8 / εcov = 8 * (1 / εcov) := by ring
      have h2 : 8 * (1 / εcov) ≤ 8 * (Ce * ℓ) := by linarith
      have h3 : 8 * (Ce * ℓ) ≤ Θ * ℓ := by
        have : 8 * Ce ≤ Θ := by
          have : 0 ≤ 4608 * Q * (Q + 1) * aε := by positivity
          linarith
        nlinarith
      linarith
    · rw [one_div_div]
      have h1 : 4608 * Q * (Q + 1) / ε = 4608 * Q * (Q + 1) * (1 / ε) := by ring
      have h2 : 4608 * Q * (Q + 1) * (1 / ε) ≤ 4608 * Q * (Q + 1) * (aε * ℓ) :=
        mul_le_mul_of_nonneg_left hinvε hQQ.le
      have h3 : 4608 * Q * (Q + 1) * (aε * ℓ) ≤ Θ * ℓ := by
        have : 4608 * Q * (Q + 1) * aε ≤ Θ := by
          have : 0 ≤ 8 * Ce := by positivity
          linarith
        have e : 4608 * Q * (Q + 1) * (aε * ℓ) = 4608 * Q * (Q + 1) * aε * ℓ := by ring
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
    rw [hΘ]; have : 0 ≤ 8 * Ce + 4608 * Q * (Q + 1) * aε := by positivity
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
      _ ≤ 8 * Ce + 4608 * Q * (Q + 1) * aε := le_add_of_nonneg_right (by positivity)
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
    1 ≤ (50 * max (Real.log (20 / τ)) (Real.log (8 * W / δ')) / s ^ 2 + 20 / s + 28) / pAP
        + 64 * Real.log ((Q + 2 + 2) / (δ' / 2)) / pAP ^ 2
      ∧ (50 * max (Real.log (20 / τ)) (Real.log (8 * W / δ')) / s ^ 2 + 20 / s + 28) / pAP
        + 64 * Real.log ((Q + 2 + 2) / (δ' / 2)) / pAP ^ 2
      ≤ ((50 * (max (Real.log (20 * Θ)) (Real.log (8 * W₀ / δ')) + 2) / s ^ 2 + 20 / s + 28) / pAP
        + 64 * Real.log ((Q + 4) / (δ' / 2)) / pAP ^ 2) * ℓ := by
  have hℓ0 : 0 < ℓ := by linarith
  have hW₀ : 0 < W₀ := by linarith
  have hQ4 : Q + 2 + 2 = Q + 4 := by ring
  rw [hQ4]
  have hZ : 0 ≤ 64 * Real.log ((Q + 4) / (δ' / 2)) / pAP ^ 2 :=
    div_nonneg (mul_nonneg (by norm_num) (Real.log_nonneg (by
      rw [le_div_iff₀ (by positivity)]; linarith))) (by positivity)
  have h20τ : 0 ≤ Real.log (20 / τ) :=
    Real.log_nonneg (by rw [le_div_iff₀ hτ0]; linarith)
  have h8W : 0 ≤ Real.log (8 * W / δ') :=
    Real.log_nonneg (by rw [le_div_iff₀ hδ'0]; linarith)
  have hs20 : 0 ≤ 20 / s := by positivity
  have hs2 : 0 < s ^ 2 := by positivity
  constructor
  · have hm0 : 0 ≤ max (Real.log (20 / τ)) (Real.log (8 * W / δ')) := le_max_of_le_left h20τ
    have hN0 : 0 ≤ 50 * max (Real.log (20 / τ)) (Real.log (8 * W / δ')) / s ^ 2 :=
      div_nonneg (mul_nonneg (by norm_num) hm0) hs2.le
    have h1 : 1 ≤ (50 * max (Real.log (20 / τ)) (Real.log (8 * W / δ')) / s ^ 2 + 20 / s + 28)
        / pAP := by
      rw [le_div_iff₀ hp]
      linarith
    linarith
  have hτ2 : Real.log (20 / τ) ≤ Real.log (20 * Θ) + Real.log ℓ := by
    have h : 20 / τ ≤ 20 * Θ * ℓ := by
      have e : 20 / τ = 20 * (1 / τ) := by ring
      rw [e]; linarith
    calc Real.log (20 / τ) ≤ Real.log (20 * Θ * ℓ) := Real.log_le_log (by positivity) h
      _ = Real.log (20 * Θ) + Real.log ℓ := Real.log_mul (by positivity) hℓ0.ne'
  have hW2 : Real.log (8 * W / δ') ≤ Real.log (8 * W₀ / δ') + 2 * Real.log ℓ :=
    log_le_mul_of_le (by positivity) hℓ
      (by rw [show 8 * W₀ / δ' * ℓ ^ 2 = 8 * (W₀ * ℓ ^ 2) / δ' by ring]; gcongr) (by positivity)
      le_rfl
  have hl0 : 0 ≤ Real.log (20 * Θ) := Real.log_nonneg (by linarith)
  have hlℓ : 0 ≤ Real.log ℓ := Real.log_nonneg hℓ
  have hmax : max (Real.log (20 / τ)) (Real.log (8 * W / δ'))
      ≤ max (Real.log (20 * Θ)) (Real.log (8 * W₀ / δ')) + 2 * Real.log ℓ := by
    refine max_le ?_ ?_
    · have := le_max_left (Real.log (20 * Θ)) (Real.log (8 * W₀ / δ'))
      linarith
    · have := le_max_right (Real.log (20 * Θ)) (Real.log (8 * W₀ / δ'))
      linarith
  have hm₀ : 0 ≤ max (Real.log (20 * Θ)) (Real.log (8 * W₀ / δ')) := le_max_of_le_left hl0
  have hX : 0 ≤ 50 * max (Real.log (20 * Θ)) (Real.log (8 * W₀ / δ')) / s ^ 2 + 20 / s + 28 := by
    have : 0 ≤ 50 * max (Real.log (20 * Θ)) (Real.log (8 * W₀ / δ')) / s ^ 2 :=
      div_nonneg (mul_nonneg (by norm_num) hm₀) hs2.le
    linarith
  have hsum : 50 * max (Real.log (20 / τ)) (Real.log (8 * W / δ')) / s ^ 2 + 20 / s + 28
      ≤ (50 * (max (Real.log (20 * Θ)) (Real.log (8 * W₀ / δ')) + 2) / s ^ 2 + 20 / s + 28)
        * ℓ := by
    have h1 : 50 * max (Real.log (20 / τ)) (Real.log (8 * W / δ')) / s ^ 2
        ≤ 50 * (max (Real.log (20 * Θ)) (Real.log (8 * W₀ / δ')) + 2 * Real.log ℓ) / s ^ 2 := by
      gcongr
    have h2 := add_log_le hX hℓ (show (0 : ℝ) ≤ 100 / s ^ 2 by positivity)
    have e : 50 * (max (Real.log (20 * Θ)) (Real.log (8 * W₀ / δ')) + 2 * Real.log ℓ) / s ^ 2
        + 20 / s + 28
        = (50 * max (Real.log (20 * Θ)) (Real.log (8 * W₀ / δ')) / s ^ 2 + 20 / s + 28)
          + 100 / s ^ 2 * Real.log ℓ := by ring
    have e2 : (50 * max (Real.log (20 * Θ)) (Real.log (8 * W₀ / δ')) / s ^ 2 + 20 / s + 28)
        + 100 / s ^ 2
        = 50 * (max (Real.log (20 * Θ)) (Real.log (8 * W₀ / δ')) + 2) / s ^ 2 + 20 / s + 28 := by
      ring
    rw [← e2]
    linarith
  have hterm : (50 * max (Real.log (20 / τ)) (Real.log (8 * W / δ')) / s ^ 2 + 20 / s + 28) / pAP
      ≤ (50 * (max (Real.log (20 * Θ)) (Real.log (8 * W₀ / δ')) + 2) / s ^ 2 + 20 / s + 28)
        / pAP * ℓ := by
    rw [div_mul_eq_mul_div]
    gcongr
  have hZℓ : 64 * Real.log ((Q + 4) / (δ' / 2)) / pAP ^ 2
      ≤ 64 * Real.log ((Q + 4) / (δ' / 2)) / pAP ^ 2 * ℓ := le_mul_of_one_le_right hZ hℓ
  linarith

lemma rb_le {Q τ Θ ℓ PB P₀ δ' s pAP : ℝ} (hQ : 0 ≤ Q) (hτ0 : 0 < τ) (hτ1 : τ ≤ 1)
    (hinvτ : 1 / τ ≤ Θ * ℓ) (hΘ : 1 ≤ Θ) (hℓ : 1 ≤ ℓ) (hPB1 : 1 ≤ PB) (hPB : PB ≤ P₀ * ℓ)
    (hP₀ : 1 ≤ P₀) (hδ'0 : 0 < δ') (hδ'1 : δ' ≤ 1) (hs0 : 0 < s) (hp : 0 < pAP)
    (hp1 : pAP ≤ 1) :
    0 ≤ 161000 * (Q + 2) ^ 2 * Real.log ((Q + 2 + 2) * PB / (δ' / 2 * (δ' / 4) * (pAP / 2) * τ))
        / (s ^ 4 * τ ^ 2)
      ∧ 161000 * (Q + 2) ^ 2 * Real.log ((Q + 2 + 2) * PB / (δ' / 2 * (δ' / 4) * (pAP / 2) * τ))
        / (s ^ 4 * τ ^ 2)
      ≤ 161000 * (Q + 2) ^ 2 * Θ ^ 2
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
  · have e : 161000 * (Q + 2) ^ 2 * Real.log ((Q + 4) * PB / (δ' ^ 2 * pAP * τ / 16))
        / (s ^ 4 * τ ^ 2)
        = 161000 * (Q + 2) ^ 2 / s ^ 4 * (Real.log ((Q + 4) * PB / (δ' ^ 2 * pAP * τ / 16))
          * (1 / τ ^ 2)) := by field_simp
    rw [e]
    have e2 : 161000 * (Q + 2) ^ 2 * Θ ^ 2 * (Real.log y + 2) / s ^ 4 * ℓ ^ 3
        = 161000 * (Q + 2) ^ 2 / s ^ 4 * ((Real.log y + 2) * ℓ * (Θ ^ 2 * ℓ ^ 2)) := by ring
    rw [e2]
    gcongr

/-- The band's ceiling `W` at the length `ℓ`, within `ℓ²` of its length-free bound. -/
lemma w_le {ℓ δ' Wm : ℝ} (hℓ1 : 1 ≤ ℓ) (hWm : 0 ≤ Wm) :
    max (Real.exp 1 * Real.sqrt δ' * Wm / 2) 1
      ≤ (Real.exp 1 * Real.sqrt δ' * Wm / 2 + 1) * ℓ ^ 2 := by
  have hℓ21 : 1 ≤ ℓ ^ 2 := one_le_pow₀ hℓ1
  have h0b : 0 ≤ Real.exp 1 * Real.sqrt δ' * Wm / 2 := by positivity
  refine (max_le (by linarith) (by linarith)).trans
    (le_mul_of_one_le_right (by positivity) hℓ21)

/-- `3 + 2 log₂ (RB + 1)` grows at most linearly in `ℓ` when `RB ≤ R₀ ℓ³`. -/
lemma logRB_le {RB R₀ ℓ : ℝ} (hRB0 : 0 ≤ RB) (hRB : RB + 1 ≤ (R₀ + 1) * ℓ ^ 3) (hR₀ : 0 ≤ R₀)
    (hℓ : 1 ≤ ℓ) :
    3 + 2 * Real.logb 2 (RB + 1) ≤ (13 + 2 * Real.logb 2 (R₀ + 1)) * ℓ := by
  have hℓ0 : 0 < ℓ := by linarith
  have hl2 : 0.6931471803 < Real.log 2 := Real.log_two_gt_d9
  have h1 : Real.logb 2 (RB + 1) ≤ Real.logb 2 ((R₀ + 1) * ℓ ^ 3) :=
    Real.logb_le_logb_of_le (by norm_num) (by linarith) hRB
  rw [Real.logb_mul (by positivity) (by positivity), Real.logb_pow] at h1
  have h2 : Real.logb 2 ℓ ≤ 3 / 2 * (ℓ - 1) := by
    rw [Real.logb, div_le_iff₀ (Real.log_pos (by norm_num))]
    have := Real.log_le_sub_one_of_pos hℓ0
    nlinarith
  have h3 : 0 ≤ Real.logb 2 (R₀ + 1) := Real.logb_nonneg (by norm_num) (by linarith)
  push_cast at h1
  nlinarith

set_option maxHeartbeats 1000000 in -- eight terms, each a chain of products
lemma cap_ge {Q τ ι mn s Θ ℓ RB R₀ PB P₀ δ' pAP E₁ : ℝ} (hQ : 0 ≤ Q) (hτ0 : 0 < τ)
    (hinvτ : 1 / τ ≤ Θ * ℓ) (hι0 : 0 < ι) (hinvι : 1 / ι ≤ Θ * ℓ) (hΘ : 1 ≤ Θ) (hℓ : 1 ≤ ℓ)
    (hRB0 : 0 ≤ RB) (hRB : RB ≤ R₀ * ℓ ^ 3) (hPB0 : 0 ≤ PB) (hPB : PB ≤ P₀ * ℓ)
    (hP₀ : 0 ≤ P₀) (hR₀ : 0 ≤ R₀) (hδ'0 : 0 < δ') (hδ'1 : δ' ≤ 1) (hp : 0 < pAP)
    (hp1 : pAP ≤ 1) (hs0 : 0 < s) (hs1 : s ≤ 1 / 2) (hτm : τ ≤ mn)
    (hE₁ : E₁ = (Q + 3) ^ 2 * (2 ^ 20 * Θ * (R₀ + 1) / s
      + 2 ^ 36 * Θ ^ 2 * (13 + 2 * Real.logb 2 (R₀ + 1)) / δ' + 2 ^ 16 * (P₀ + 2) / (pAP * δ'))) :
    1 / (E₁ * ℓ ^ 5) ≤ min (min (min (τ * s / (2 ^ 19 * (Q + 2 + 3) * (RB + 1)))
          (δ' * mn ^ 2 / (2 ^ 35 * (Q + 2) * (3 + 2 * Real.logb 2 (RB + 1)))))
        (min (δ' * (ι / 2) ^ 2 / (2 ^ 5 * (Q + 2 + 1) * (3 + 2 * Real.logb 2 (RB + 1))))
          (ι / 2 / (8 * (Q + 2) * (RB + 1)))))
      (min (min (δ' * pAP / (2 ^ 15 * (Q + 2 + 1) * (PB + 1)))
          (ι / (12 * (Q + 2) ^ 2 * (RB + 1))))
        (min (pAP / (2 * (Q + 2) * (PB + 2))) (1 / (6 * (Q + 2) * (RB + 1))))) := by
  have hℓ0 : 0 < ℓ := by linarith
  have hmn0 : 0 < mn := lt_of_lt_of_le hτ0 hτm
  set G : ℝ := 13 + 2 * Real.logb 2 (R₀ + 1) with hG
  have hG0 : 13 ≤ G := by
    have : 0 ≤ Real.logb 2 (R₀ + 1) := Real.logb_nonneg (by norm_num) (by linarith)
    rw [hG]; linarith
  set X1 : ℝ := 2 ^ 20 * Θ * (R₀ + 1) / s with hX1
  set X2 : ℝ := 2 ^ 36 * Θ ^ 2 * G / δ' with hX2
  set X3 : ℝ := 2 ^ 16 * (P₀ + 2) / (pAP * δ') with hX3
  have hX10 : 0 ≤ X1 := by positivity
  have hX20 : 0 ≤ X2 := by positivity
  have hX30 : 0 ≤ X3 := by positivity
  have hQ3 : (0 : ℝ) < (Q + 3) ^ 2 := by positivity
  have hE₁0 : 0 < E₁ := by rw [hE₁]; positivity
  have hD : 0 < E₁ * ℓ ^ 5 := by positivity
  -- each term `a/b` of the cap is at least `1/(E₁ ℓ⁵)` once `b ≤ a (Q + 3)² Xᵢ ℓ⁵`
  have key : ∀ a b X : ℝ, 0 < b → 0 ≤ a → X ≤ X1 + X2 + X3 → b ≤ a * ((Q + 3) ^ 2 * X * ℓ ^ 5) →
      1 / (E₁ * ℓ ^ 5) ≤ a / b := by
    intro a b X hb ha hX h
    rw [div_le_div_iff₀ hD hb, one_mul]
    refine h.trans (mul_le_mul_of_nonneg_left ?_ ha)
    have h1 : (Q + 3) ^ 2 * X * ℓ ^ 5 ≤ (Q + 3) ^ 2 * (X1 + X2 + X3) * ℓ ^ 5 := by gcongr
    have h2 : E₁ * ℓ ^ 5 = (Q + 3) ^ 2 * (X1 + X2 + X3) * ℓ ^ 5 := by rw [hE₁]
    linarith only [h1, h2]
  have h1X1 : X1 ≤ X1 + X2 + X3 := by linarith
  have h1X2 : X2 ≤ X1 + X2 + X3 := by linarith
  have h1X3 : X3 ≤ X1 + X2 + X3 := by linarith
  have hτΘ : 1 ≤ τ * Θ * ℓ := by
    have := mul_le_mul_of_nonneg_left hinvτ hτ0.le
    rw [mul_one_div_cancel hτ0.ne'] at this
    linarith
  have hιΘ : 1 ≤ ι * Θ * ℓ := by
    have := mul_le_mul_of_nonneg_left hinvι hι0.le
    rw [mul_one_div_cancel hι0.ne'] at this
    linarith
  have hmnΘ : 1 ≤ mn * Θ * ℓ := le_trans hτΘ (by gcongr)
  have hℓ3 : 1 ≤ ℓ ^ 3 := one_le_pow₀ hℓ
  have hRB1 : RB + 1 ≤ (R₀ + 1) * ℓ ^ 3 := by
    have e : (R₀ + 1) * ℓ ^ 3 = R₀ * ℓ ^ 3 + ℓ ^ 3 := by ring
    linarith only [hRB, hℓ3, e]
  have hPB1 : PB + 2 ≤ (P₀ + 2) * ℓ := by
    have e : (P₀ + 2) * ℓ = P₀ * ℓ + 2 * ℓ := by ring
    linarith only [hPB, hℓ, e]
  have hlg := logRB_le hRB0 hRB1 hR₀ hℓ
  rw [← hG] at hlg
  have hlgpos : 0 < 3 + 2 * Real.logb 2 (RB + 1) := by
    have : 0 ≤ Real.logb 2 (RB + 1) := Real.logb_nonneg (by norm_num) (by linarith)
    linarith
  have hQa : Q + 5 ≤ (Q + 3) ^ 2 := by nlinarith only [hQ]
  have hQb : Q + 3 ≤ (Q + 3) ^ 2 := by nlinarith only [hQ]
  have hQc : (Q + 2) ^ 2 ≤ (Q + 3) ^ 2 := by nlinarith only [hQ]
  have hℓ1 : ℓ ≤ ℓ ^ 5 := le_self_pow₀ hℓ (by norm_num)
  have hℓ34 : ℓ ^ 3 ≤ ℓ ^ 5 := pow_le_pow_right₀ hℓ (by norm_num)
  have hℓ2 : 1 ≤ ℓ ^ 2 := one_le_pow₀ hℓ
  refine le_min (le_min (le_min ?_ ?_) (le_min ?_ ?_)) (le_min (le_min ?_ ?_) (le_min ?_ ?_))
  · -- the redraws
    refine key _ _ X1 (by positivity) (by positivity) h1X1 ?_
    have e : τ * s * ((Q + 3) ^ 2 * X1 * ℓ ^ 5)
        = 2 ^ 20 * (Q + 3) ^ 2 * ((τ * Θ * ℓ) * ((R₀ + 1) * ℓ ^ 3) * ℓ) := by
      rw [hX1]; field_simp
    rw [e]
    have h1 : RB + 1 ≤ (τ * Θ * ℓ) * ((R₀ + 1) * ℓ ^ 3) * ℓ := by
      have h := mul_le_mul hτΘ hRB1 (by positivity) (by positivity)
      rw [one_mul] at h
      exact h.trans (le_mul_of_one_le_right (by positivity) hℓ)
    have h2 : 2 ^ 19 * (Q + 2 + 3) ≤ 2 ^ 20 * (Q + 3) ^ 2 := by linarith only [hQa, hQ]
    exact mul_le_mul h2 h1 (by positivity) (by positivity)
  · -- the hash
    refine key _ _ X2 (by positivity) (by positivity) h1X2 ?_
    have e : δ' * mn ^ 2 * ((Q + 3) ^ 2 * X2 * ℓ ^ 5)
        = 2 ^ 36 * (Q + 3) ^ 2 * ((mn * Θ * ℓ) ^ 2 * (G * ℓ) * ℓ ^ 2) := by
      rw [hX2]; field_simp
    rw [e]
    have h1 : 3 + 2 * Real.logb 2 (RB + 1) ≤ (mn * Θ * ℓ) ^ 2 * (G * ℓ) * ℓ ^ 2 := by
      have hm2 : 1 ≤ (mn * Θ * ℓ) ^ 2 := one_le_pow₀ hmnΘ
      have : G * ℓ ≤ (mn * Θ * ℓ) ^ 2 * (G * ℓ) := le_mul_of_one_le_left (by positivity) hm2
      have : (mn * Θ * ℓ) ^ 2 * (G * ℓ) ≤ (mn * Θ * ℓ) ^ 2 * (G * ℓ) * ℓ ^ 2 :=
        le_mul_of_one_le_right (by positivity) hℓ2
      linarith
    have h2 : 2 ^ 35 * (Q + 2) ≤ 2 ^ 36 * (Q + 3) ^ 2 := by linarith only [hQb, hQ]
    exact mul_le_mul h2 h1 hlgpos.le (by positivity)
  · -- the quality's ladder tail
    refine key _ _ X2 (by positivity) (by positivity) h1X2 ?_
    have e : δ' * (ι / 2) ^ 2 * ((Q + 3) ^ 2 * X2 * ℓ ^ 5)
        = 2 ^ 34 * (Q + 3) ^ 2 * ((ι * Θ * ℓ) ^ 2 * (G * ℓ) * ℓ ^ 2) := by
      rw [hX2]; field_simp
    rw [e]
    have h1 : 3 + 2 * Real.logb 2 (RB + 1) ≤ (ι * Θ * ℓ) ^ 2 * (G * ℓ) * ℓ ^ 2 := by
      have hm2 : 1 ≤ (ι * Θ * ℓ) ^ 2 := one_le_pow₀ hιΘ
      have : G * ℓ ≤ (ι * Θ * ℓ) ^ 2 * (G * ℓ) := le_mul_of_one_le_left (by positivity) hm2
      have : (ι * Θ * ℓ) ^ 2 * (G * ℓ) ≤ (ι * Θ * ℓ) ^ 2 * (G * ℓ) * ℓ ^ 2 :=
        le_mul_of_one_le_right (by positivity) hℓ2
      linarith
    have h2 : 2 ^ 5 * (Q + 2 + 1) ≤ 2 ^ 34 * (Q + 3) ^ 2 := by linarith only [hQb, hQ]
    exact mul_le_mul h2 h1 hlgpos.le (by positivity)
  · -- the quality's table mass
    refine key _ _ X1 (by positivity) (by positivity) h1X1 ?_
    have e : ι / 2 * ((Q + 3) ^ 2 * X1 * ℓ ^ 5)
        = 2 ^ 19 / s * (Q + 3) ^ 2 * ((ι * Θ * ℓ) * ((R₀ + 1) * ℓ ^ 3) * ℓ) := by
      rw [hX1]; field_simp
    rw [e]
    have h1 : RB + 1 ≤ (ι * Θ * ℓ) * ((R₀ + 1) * ℓ ^ 3) * ℓ := by
      have h := mul_le_mul hιΘ hRB1 (by positivity) (by positivity)
      rw [one_mul] at h
      exact h.trans (le_mul_of_one_le_right (by positivity) hℓ)
    have h2 : 8 * (Q + 2) ≤ 2 ^ 19 / s * (Q + 3) ^ 2 := by
      rw [div_mul_eq_mul_div, le_div_iff₀ hs0]
      have := mul_le_mul_of_nonneg_left hs1 (by positivity : (0 : ℝ) ≤ 8 * (Q + 2))
      linarith only [this, hQb, hQ]
    exact mul_le_mul h2 h1 (by positivity) (by positivity)
  · -- the suffix collision
    refine key _ _ X3 (by positivity) (by positivity) h1X3 ?_
    have e : δ' * pAP * ((Q + 3) ^ 2 * X3 * ℓ ^ 5) = 2 ^ 16 * (Q + 3) ^ 2 * ((P₀ + 2) * ℓ ^ 5) := by
      rw [hX3]; field_simp
    rw [e]
    have h1 : PB + 1 ≤ (P₀ + 2) * ℓ ^ 5 := by
      have h := mul_le_mul_of_nonneg_left hℓ1 (by positivity : (0 : ℝ) ≤ P₀ + 2)
      linarith only [h, hPB1]
    have h2 : 2 ^ 15 * (Q + 2 + 1) ≤ 2 ^ 16 * (Q + 3) ^ 2 := by linarith only [hQb, hQ]
    exact mul_le_mul h2 h1 (by positivity) (by positivity)
  · -- the earlier rounds' prefixes
    refine key _ _ X1 (by positivity) (by positivity) h1X1 ?_
    have e : ι * ((Q + 3) ^ 2 * X1 * ℓ ^ 5)
        = 2 ^ 20 / s * (Q + 3) ^ 2 * ((ι * Θ * ℓ) * ((R₀ + 1) * ℓ ^ 3) * ℓ) := by
      rw [hX1]; field_simp
    rw [e]
    have h1 : RB + 1 ≤ (ι * Θ * ℓ) * ((R₀ + 1) * ℓ ^ 3) * ℓ := by
      have h := mul_le_mul hιΘ hRB1 (by positivity) (by positivity)
      rw [one_mul] at h
      exact h.trans (le_mul_of_one_le_right (by positivity) hℓ)
    have h2 : 12 * (Q + 2) ^ 2 ≤ 2 ^ 20 / s * (Q + 3) ^ 2 := by
      rw [div_mul_eq_mul_div, le_div_iff₀ hs0]
      have := mul_le_mul_of_nonneg_left hs1 (by positivity : (0 : ℝ) ≤ 12 * (Q + 2) ^ 2)
      linarith only [this, hQc, sq_nonneg (Q + 2), sq_nonneg (Q + 3)]
    exact mul_le_mul h2 h1 (by positivity) (by positivity)
  · -- the earlier rounds' suffixes
    refine key _ _ X3 (by positivity) (by positivity) h1X3 ?_
    have e : pAP * ((Q + 3) ^ 2 * X3 * ℓ ^ 5) = 2 ^ 16 / δ' * (Q + 3) ^ 2 * ((P₀ + 2) * ℓ ^ 5) := by
      rw [hX3]; field_simp
    rw [e]
    have h1 : PB + 2 ≤ (P₀ + 2) * ℓ ^ 5 :=
      hPB1.trans (mul_le_mul_of_nonneg_left hℓ1 (by positivity))
    have h2 : 2 * (Q + 2) ≤ 2 ^ 16 / δ' * (Q + 3) ^ 2 := by
      rw [div_mul_eq_mul_div, le_div_iff₀ hδ'0]
      have := mul_le_mul_of_nonneg_left hδ'1 (by positivity : (0 : ℝ) ≤ 2 * (Q + 2))
      linarith only [this, hQb, hQ]
    exact mul_le_mul h2 h1 (by positivity) (by positivity)
  · -- the probes' avoided mass
    refine key _ _ X1 (by positivity) zero_le_one h1X1 ?_
    have e : 1 * ((Q + 3) ^ 2 * X1 * ℓ ^ 5)
        = 2 ^ 20 * Θ / s * (Q + 3) ^ 2 * ((R₀ + 1) * ℓ ^ 3 * ℓ ^ 2) := by
      rw [hX1]; field_simp
    rw [e]
    have h1 : RB + 1 ≤ (R₀ + 1) * ℓ ^ 3 * ℓ ^ 2 :=
      hRB1.trans (le_mul_of_one_le_right (by positivity) hℓ2)
    have h2 : 6 * (Q + 2) ≤ 2 ^ 20 * Θ / s * (Q + 3) ^ 2 := by
      rw [div_mul_eq_mul_div, le_div_iff₀ hs0]
      have h3 := mul_le_mul_of_nonneg_left hs1 (by positivity : (0 : ℝ) ≤ 6 * (Q + 2))
      have h4 : 2 ^ 20 * (Q + 3) ^ 2 ≤ 2 ^ 20 * Θ * (Q + 3) ^ 2 := by
        have := mul_le_mul_of_nonneg_left hΘ (by positivity : (0 : ℝ) ≤ 2 ^ 20 * (Q + 3) ^ 2)
        linarith only [this]
      linarith only [h3, h4, hQb, hQ]
    exact mul_le_mul h2 h1 (by positivity) (by positivity)

lemma dfa_final {Ak ℓ ε cap aε E₁ E E₂ lg Q n J RB R₀ B₃ K₁ δ : ℝ} (hAk : 0 < Ak)
    (hℓ : 1 ≤ ℓ) (hε0 : 0 < ε) (hinvε : 1 / ε ≤ aε * ℓ) (haε : 0 < aε)
    (hcap : 1 / (E₁ * ℓ ^ 5) ≤ cap) (hE₁ : 0 < E₁) (hlg : 0 ≤ lg)
    (hE : E = 1536 * aε * E₁ + 1536 * lg * aε)
    (hE₂ : E₂ = 4 * (Q + 1) * n * (6 * (Q + 2) * R₀ + K₁ + n) / δ) (hQ : 0 ≤ Q) (hn : 0 ≤ n)
    (hδ : 0 < δ) (hJ : J = Q + 2) (hRB0 : 0 ≤ RB) (hRB : RB ≤ R₀ * ℓ ^ 3) (hB₃0 : 0 ≤ B₃)
    (hB₃ : B₃ ≤ K₁ * ℓ) (hpow : ℓ ^ 6 * (E + E₂) ≤ Ak) :
    Ak⁻¹ ≤ cap * ε / 1536 ∧ 1536 * lg * Ak⁻¹ ≤ ε
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
    have hc : 1 / (1536 * aε * E₁ * ℓ ^ 6) ≤ cap * ε / 1536 := by
      have h1 : 1 / (1536 * aε * E₁ * ℓ ^ 6) = 1 / (E₁ * ℓ ^ 5) * (1 / (aε * ℓ)) / 1536 := by
        field_simp
      rw [h1]
      have hcap0 : 0 ≤ cap := (by positivity : (0 : ℝ) ≤ 1 / (E₁ * ℓ ^ 5)).trans hcap
      gcongr
    refine le_trans ?_ hc
    rw [one_div]
    refine inv_anti₀ (by positivity) (le_trans ?_ hpow)
    have : 1536 * aε * E₁ ≤ E + E₂ := by
      rw [hE]; have : 0 ≤ 1536 * lg * aε := by positivity
      linarith
    have := mul_le_mul_of_nonneg_left this (show 0 ≤ ℓ ^ 6 by positivity)
    linarith
  · -- the prefix mass against the noise's tails
    rcases eq_or_lt_of_le hlg with h0 | hlg0
    · rw [← h0]; simp [hε0.le]
    have h1 : 1536 * lg * aε * ℓ ≤ ℓ ^ 6 * (E + E₂) := by
      have : 1536 * lg * aε ≤ E + E₂ := by
        rw [hE]; have : 0 ≤ 1536 * aε * E₁ := by positivity
        linarith
      have h2 : ℓ ≤ ℓ ^ 6 := le_self_pow₀ hℓ (by norm_num)
      calc 1536 * lg * aε * ℓ ≤ (E + E₂) * ℓ ^ 6 :=
            mul_le_mul this h2 hℓ0.le (by positivity)
        _ = ℓ ^ 6 * (E + E₂) := by ring
    have h3 : 1536 * lg * Ak⁻¹ ≤ 1 / (aε * ℓ) := by
      rw [le_div_iff₀ (by positivity)]
      have : Ak⁻¹ * Ak = 1 := inv_mul_cancel₀ hAk.ne'
      have h4 : 1536 * lg * aε * ℓ ≤ Ak := h1.trans hpow
      calc 1536 * lg * Ak⁻¹ * (aε * ℓ) = (1536 * lg * aε * ℓ) * Ak⁻¹ := by ring
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
    1 ≤ (50 * (max (Real.log (20 * Θ)) (Real.log (8 * W₀ / δ')) + 2) / s ^ 2 + 20 / s + 28) / pAP
      + 64 * Real.log ((Q + 4) / (δ' / 2)) / pAP ^ 2 := by
  have h1 : 0 ≤ Real.log (20 * Θ) := Real.log_nonneg (by linarith)
  have hm : 0 ≤ max (Real.log (20 * Θ)) (Real.log (8 * W₀ / δ')) := le_max_of_le_left h1
  have h3 : 0 ≤ 64 * Real.log ((Q + 4) / (δ' / 2)) / pAP ^ 2 :=
    div_nonneg (mul_nonneg (by norm_num) (Real.log_nonneg (by
      rw [le_div_iff₀ (by positivity)]; linarith))) (by positivity)
  have h5 : 0 ≤ 50 * (max (Real.log (20 * Θ)) (Real.log (8 * W₀ / δ')) + 2) / s ^ 2 :=
    div_nonneg (mul_nonneg (by norm_num) (by linarith)) (by positivity)
  have h6 : 0 ≤ 20 / s := by positivity
  have h4 : 1 ≤ (50 * (max (Real.log (20 * Θ)) (Real.log (8 * W₀ / δ')) + 2) / s ^ 2 + 20 / s + 28)
      / pAP := by
    rw [le_div_iff₀ hp]; linarith
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

lemma pool_le {Q τ Θ ℓ W W₀ δ' s pAP : ℝ} (hτ0 : 0 < τ) (hinvτ : 1 / τ ≤ Θ * ℓ) (hΘ : 1 ≤ Θ)
    (hℓ : 1 ≤ ℓ) (hW1 : 1 ≤ W) (hWW : W ≤ W₀ * ℓ ^ 2) (hδ'0 : 0 < δ') (hs0 : 0 < s)
    (hp : 0 < pAP) :
    (50 * max (Real.log (20 / τ)) (Real.log (8 * W / δ')) / s ^ 2 + 20 / s + 28) / pAP
        + 64 * Real.log ((Q + 2 + 2) / (δ' / 2)) / pAP ^ 2
      ≤ (50 * (max (Real.log (20 * Θ)) (Real.log (8 * W₀ / δ')) + 2 * Real.log ℓ) / s ^ 2
          + 20 / s + 28) / pAP
        + 64 * Real.log ((Q + 4) / (δ' / 2)) / pAP ^ 2 := by
  have hℓ0 : 0 < ℓ := by linarith
  have hW₀ : 0 < W₀ := by
    by_contra h
    push Not at h
    nlinarith [sq_nonneg ℓ]
  rw [show Q + 2 + 2 = Q + 4 by ring]
  have hτ2 : Real.log (20 / τ) ≤ Real.log (20 * Θ) + Real.log ℓ := by
    have h : 20 / τ ≤ 20 * Θ * ℓ := by
      have e : 20 / τ = 20 * (1 / τ) := by ring
      rw [e]; linarith
    calc Real.log (20 / τ) ≤ Real.log (20 * Θ * ℓ) := Real.log_le_log (by positivity) h
      _ = Real.log (20 * Θ) + Real.log ℓ := Real.log_mul (by positivity) hℓ0.ne'
  have hW2 : Real.log (8 * W / δ') ≤ Real.log (8 * W₀ / δ') + 2 * Real.log ℓ :=
    log_le_mul_of_le (by positivity) hℓ
      (by rw [show 8 * W₀ / δ' * ℓ ^ 2 = 8 * (W₀ * ℓ ^ 2) / δ' by ring]; gcongr) (by positivity)
      le_rfl
  have hlℓ : 0 ≤ Real.log ℓ := Real.log_nonneg hℓ
  have hmax : max (Real.log (20 / τ)) (Real.log (8 * W / δ'))
      ≤ max (Real.log (20 * Θ)) (Real.log (8 * W₀ / δ')) + 2 * Real.log ℓ := by
    refine max_le ?_ ?_
    · have := le_max_left (Real.log (20 * Θ)) (Real.log (8 * W₀ / δ'))
      linarith
    · have := le_max_right (Real.log (20 * Θ)) (Real.log (8 * W₀ / δ'))
      linarith
  gcongr

lemma count_le {Q τ Θ ℓ PB PBf δ' s pAP : ℝ} (hQ : 0 ≤ Q) (hτ0 : 0 < τ) (hτ1 : τ ≤ 1)
    (hinvτ : 1 / τ ≤ Θ * ℓ) (hΘ : 1 ≤ Θ) (hℓ : 1 ≤ ℓ) (hPB1 : 1 ≤ PB) (hPB : PB ≤ PBf)
    (hδ'0 : 0 < δ') (hδ'1 : δ' ≤ 1) (hs0 : 0 < s) (hp : 0 < pAP) (hp1 : pAP ≤ 1) :
    161000 * (Q + 2) ^ 2 * Real.log ((Q + 2 + 2) * PB / (δ' / 2 * (δ' / 4) * (pAP / 2) * τ))
        / (s ^ 4 * τ ^ 2)
      ≤ 161000 * (Q + 2) ^ 2 * Θ ^ 2 * ℓ ^ 2
        * Real.log (16 * (Q + 4) * PBf * Θ * ℓ / (δ' ^ 2 * pAP)) / s ^ 4 := by
  have hD : δ' / 2 * (δ' / 4) * (pAP / 2) * τ = δ' ^ 2 * pAP * τ / 16 := by ring
  rw [hD, show Q + 2 + 2 = Q + 4 by ring]
  have hD0 : 0 < δ' ^ 2 * pAP * τ / 16 := by positivity
  have harg1 : 1 ≤ (Q + 4) * PB / (δ' ^ 2 * pAP * τ / 16) := by
    rw [le_div_iff₀ hD0]
    have h1 : δ' ^ 2 ≤ 1 := by nlinarith
    have h2 : δ' ^ 2 * pAP ≤ 1 := by nlinarith
    have h3 : δ' ^ 2 * pAP * τ ≤ 1 := by nlinarith [mul_nonneg (sq_nonneg δ') hp.le]
    nlinarith
  have harg : (Q + 4) * PB / (δ' ^ 2 * pAP * τ / 16)
      ≤ 16 * (Q + 4) * PBf * Θ * ℓ / (δ' ^ 2 * pAP) := by
    have e : (Q + 4) * PB / (δ' ^ 2 * pAP * τ / 16)
        = 16 * (Q + 4) * PB / (δ' ^ 2 * pAP) * (1 / τ) := by
      field_simp
    have e2 : 16 * (Q + 4) * PBf * Θ * ℓ / (δ' ^ 2 * pAP)
        = 16 * (Q + 4) * PBf / (δ' ^ 2 * pAP) * (Θ * ℓ) := by ring
    have hPBf : 0 ≤ PBf := by linarith
    rw [e, e2]
    gcongr
  have hlog := Real.log_le_log (by linarith) harg
  have hlog0 : 0 ≤ Real.log ((Q + 4) * PB / (δ' ^ 2 * pAP * τ / 16)) := Real.log_nonneg harg1
  have hinv2 : 1 / τ ^ 2 ≤ Θ ^ 2 * ℓ ^ 2 := by
    rw [show 1 / τ ^ 2 = (1 / τ) ^ 2 by ring, show Θ ^ 2 * ℓ ^ 2 = (Θ * ℓ) ^ 2 by ring]
    exact pow_le_pow_left₀ (by positivity) hinvτ 2
  have e : 161000 * (Q + 2) ^ 2 * Real.log ((Q + 4) * PB / (δ' ^ 2 * pAP * τ / 16))
      / (s ^ 4 * τ ^ 2)
      = 161000 * (Q + 2) ^ 2 / s ^ 4 * (Real.log ((Q + 4) * PB / (δ' ^ 2 * pAP * τ / 16))
        * (1 / τ ^ 2)) := by field_simp
  rw [e]
  have e2 : 161000 * (Q + 2) ^ 2 * Θ ^ 2 * ℓ ^ 2
      * Real.log (16 * (Q + 4) * PBf * Θ * ℓ / (δ' ^ 2 * pAP)) / s ^ 4
      = 161000 * (Q + 2) ^ 2 / s ^ 4 * (Real.log (16 * (Q + 4) * PBf * Θ * ℓ / (δ' ^ 2 * pAP))
        * (Θ ^ 2 * ℓ ^ 2)) := by ring
  rw [e2]
  have := hlog0.trans hlog
  gcongr

end OrthoDFA






end
