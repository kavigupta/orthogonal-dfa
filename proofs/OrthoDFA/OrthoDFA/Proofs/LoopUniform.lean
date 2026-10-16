import OrthoDFA.Proofs.LoopSimple
import OrthoDFA.Proofs.UniformStrings

/-!
# The loop on uniform probes and suffixes

`cap_bound` turns `L ≥ L₀` into the probes' prefix mass at `L − 1` being under the simple claim's
`cap · ρ / 4`: the claim's cap is at least its first or its third term, `4/(cap ρ) ≤ (L + 1)⁶ E`
with `E` free of `L`, so `L ≥ L₀` gives `(L − 1) log |Σ| ≥ 6 log (L + 1) + log E`.
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory
open scoped ENNReal

section Cap

lemma seven_log_le {A L : ℝ} (hA : 2 ≤ A) (hL : 0 < L) :
    7 * Real.log L ≤ 28 + L * Real.log A / 8 := by
  have hl2 := Real.log_two_gt_d9
  have hlA : Real.log 2 ≤ Real.log A := Real.log_le_log (by norm_num) hA
  have hlA0 : 0 < Real.log A := by linarith
  have hx : 0 < L * Real.log A / 56 := by positivity
  have h1 := Real.log_le_sub_one_of_pos hx
  rw [Real.log_div (by positivity) (by norm_num), Real.log_mul hL.ne' hlA0.ne'] at h1
  have h2 : Real.log (1 / 2) ≤ Real.log (Real.log A) := Real.log_le_log (by norm_num) (by linarith)
  rw [one_div, Real.log_inv] at h2
  have h3 : Real.log 56 + Real.log 2 ≤ 5 := by
    rw [← Real.log_mul (by norm_num) (by norm_num), Real.log_le_iff_le_exp (by norm_num)]
    have he := Real.exp_one_gt_d9
    have : Real.exp 5 = Real.exp 1 ^ 5 := by rw [← Real.exp_nat_mul]; norm_num
    rw [this]
    have h27 : (2.7 : ℝ) ≤ Real.exp 1 := by linarith
    nlinarith [pow_le_pow_left₀ (by norm_num : (0 : ℝ) ≤ 2.7) h27 5]
  nlinarith

set_option maxHeartbeats 2000000 in -- many small nonlinear steps
/-- The simple claim's cap is at least its first or its third term, and is large once
`L ≥ L₀`. -/
lemma cap_bound {Q A L η₀ εcov εd δ pAP δ' ι ρ s Θ P₀ R₀ E : ℝ} {n : ℕ} (Bc cL : ℝ)
    (hQ : 1 ≤ Q) (hA : 2 ≤ A) (hLn : L = n + 1) (hη0 : 0 ≤ η₀) (hη : η₀ < 1 / 2)
    (hε : 0 < εcov) (hε1 : εcov ≤ 1) (hεd : 0 < εd) (hεd1 : εd ≤ 1) (hδ : 0 < δ) (hδ1 : δ ≤ 1)
    (hp : 0 < pAP) (hp1 : pAP ≤ 1)
    (hs : s = 1 / 2 - η₀) (hδ' : δ' = δ / (2 * (Q + 1)))
    (hι : ι = min (εcov / 8) (1 / (3072 * (L + 1) * (Q + 1) * Q)))
    (hρ : ρ = 3 * εd / 16) (hBc : 1 ≤ Bc) (hcL : cL = (L + 1) * Bc)
    (hΘ : Θ = 1 / s + 8 / εcov + 3072 * Q * (Q + 1))
    (hP₀ : P₀ = 256 * (2 * Θ + Bc) / (s ^ 2 * pAP)
      + 64 * Real.log ((Q + 4) / (δ' / 2)) / pAP ^ 2)
    (hR₀ : R₀ = 6000 * (Q + 2) ^ 2 * Θ ^ 3
      * Real.log ((Q + 4) * P₀ * Θ / (δ' / 2 * (1 / 4) * (pAP / 2))) / s ^ 4)
    (hE : E = 2 ^ 24 * (Q + 3) ^ 2 * (Θ ^ 2 * (R₀ + 1) + (P₀ + 1) / pAP + A ^ 2) / (δ' * ρ))
    (hL₀ : 2 + 28 * (Real.log E + 24) / (25 * Real.log A) ≤ L) :
    let τ := min (εcov / 2) (min s ι)
    let J := Q + 2
    let PB := 256 * (Real.log (2 / τ) + cL) / (s ^ 2 * pAP)
      + 64 * Real.log ((J + 2) / (δ' / 2)) / pAP ^ 2
    let RB := 6000 * J ^ 2 * Real.log ((J + 2) * PB / (δ' / 2 * (1 / 4) * (pAP / 2) * τ))
      / (s ^ 4 * τ ^ 3)
    let cap := min (min (δ' * τ ^ 2 / (2 ^ 22 * (J + 1) ^ 2 * (RB + 1)))
        (δ' * ι ^ 2 / (2 ^ 11 * (J + 1) * (RB + 1))))
      (min (min (δ' * pAP / (2 ^ 15 * (J + 1) * (PB + 1))) (ι / (12 * J ^ 2 * (RB + 1))))
        (min (pAP / (2 * J * (PB + 2))) (1 / (6 * J * (RB + 1) * (L + 1)))))
    1 / A ^ n ≤ cap * ρ / 4
      ∧ 16384 * (L + 1) * (Q + 2) * A * Real.log ((Q + 3) * A / δ') * (1 / A ^ n) ≤ εd := by
  intro τ J PB RB cap
  have hτ : τ = min (εcov / 2) (min s ι) := rfl
  have hJ : J = Q + 2 := rfl
  -- the problem's sizes
  have hs0 : 0 < s := by rw [hs]; linarith
  have hs1 : s ≤ 1 / 2 := by rw [hs]; linarith
  have hA0 : 0 < A := by linarith
  have hδ'0 : 0 < δ' := by rw [hδ']; positivity
  have hδ'1 : δ' ≤ 1 := by rw [hδ', div_le_one (by positivity)]; linarith
  have hρ0 : 0 < ρ := by rw [hρ]; positivity
  have hρ1 : ρ ≤ 1 := by rw [hρ]; linarith
  have hL1 : 1 ≤ L := by rw [hLn]; linarith [(Nat.cast_nonneg n : (0 : ℝ) ≤ n)]
  have hℓ0 : 0 < L + 1 := by linarith
  have h8ε : 0 < 8 / εcov := by positivity
  have h1s : 0 < 1 / s := by positivity
  have hQQ : 3072 ≤ 3072 * Q * (Q + 1) := by nlinarith only [hQ]
  have hΘs : 1 / s ≤ Θ := by rw [hΘ]; linarith
  have hΘε : 8 / εcov ≤ Θ := by rw [hΘ]; linarith
  have hΘQ : 3072 * Q * (Q + 1) ≤ Θ := by rw [hΘ]; linarith
  have hΘ1 : 3072 ≤ Θ := by linarith
  have hΘ0 : 0 < Θ := by linarith
  -- the band's far side, `τ`, is at least `1/(Θ (L + 1))`
  have hι0 : 0 < ι := by rw [hι]; exact lt_min (by positivity) (by positivity)
  have hι1 : ι ≤ 1 := by rw [hι]; exact (min_le_left _ _).trans (by linarith)
  have hιL : ι * (L + 1) ≤ 1 := by
    have h := min_le_right (εcov / 8) (1 / (3072 * (L + 1) * (Q + 1) * Q))
    rw [← hι, le_div_iff₀ (by positivity)] at h
    nlinarith only [h, hι0, hQQ, hℓ0, hQ]
  have hτι : τ ≤ ι := by rw [hτ]; exact (min_le_right _ _).trans (min_le_right _ _)
  have hτ0 : 0 < τ := by rw [hτ]; exact lt_min (by positivity) (lt_min hs0 hι0)
  have hτ1 : τ ≤ 1 := hτι.trans hι1
  have hΘℓ : 0 < Θ * (L + 1) := by positivity
  have hτΘ : 1 / τ ≤ Θ * (L + 1) := by
    rw [div_le_iff₀ hτ0, ← div_le_iff₀' hΘℓ]
    have h1 : 1 / (Θ * (L + 1)) ≤ 1 / Θ :=
      one_div_le_one_div_of_le hΘ0 (le_mul_of_one_le_right hΘ0.le (by linarith))
    have h2 : 1 / Θ ≤ εcov / 8 := by
      rw [div_le_div_iff₀ hΘ0 (by norm_num)]
      rw [div_le_iff₀ hε] at hΘε; linarith
    have h3 : 1 / Θ ≤ s := by
      rw [div_le_iff₀ hΘ0]; rw [div_le_iff₀ hs0] at hΘs; linarith
    have h4 : 1 / (Θ * (L + 1)) ≤ 1 / (3072 * (L + 1) * (Q + 1) * Q) :=
      one_div_le_one_div_of_le (by positivity) (by nlinarith only [hΘQ, hℓ0])
    rw [hτ, hι]
    exact le_min (by linarith) (le_min (by linarith) (le_min (by linarith) h4))
  -- the suffix pool's bound `PB`
  have hJ2 : J + 2 = Q + 4 := by rw [hJ]; ring
  have hPB : PB = 256 * (Real.log (2 / τ) + cL) / (s ^ 2 * pAP)
      + 64 * Real.log ((Q + 4) / (δ' / 2)) / pAP ^ 2 := by
    show _ + 64 * Real.log ((J + 2) / (δ' / 2)) / pAP ^ 2 = _
    rw [hJ2]
  have hlτ0 : 0 ≤ Real.log (2 / τ) := Real.log_nonneg (by rw [le_div_iff₀ hτ0]; linarith)
  have hlτ : Real.log (2 / τ) ≤ 2 * (Θ * (L + 1)) := by
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
  have hPBle : PB ≤ (L + 1) * P₀ := by
    have e : (L + 1) * P₀ = 256 * (2 * (Θ * (L + 1)) + cL) / (s ^ 2 * pAP)
        + (L + 1) * (64 * Real.log ((Q + 4) / (δ' / 2)) / pAP ^ 2) := by rw [hP₀, hcL]; ring
    rw [e, hPB]
    have h1 : 256 * (Real.log (2 / τ) + cL) / (s ^ 2 * pAP)
        ≤ 256 * (2 * (Θ * (L + 1)) + cL) / (s ^ 2 * pAP) :=
      div_le_div_of_nonneg_right (by linarith) hsp0.le
    have h2 : 64 * Real.log ((Q + 4) / (δ' / 2)) / pAP ^ 2
        ≤ (L + 1) * (64 * Real.log ((Q + 4) / (δ' / 2)) / pAP ^ 2) :=
      le_mul_of_one_le_left (by positivity) (by linarith)
    linarith
  have hcL1 : 1 ≤ cL := by rw [hcL]; nlinarith only [hBc, hL1]
  have hPBc : cL ≤ PB := by
    have h1 : cL ≤ 256 * (Real.log (2 / τ) + cL) / (s ^ 2 * pAP) := by
      rw [le_div_iff₀ hsp0]
      nlinarith only [hcL1, hsp, hlτ0]
    have h2 : 0 ≤ 64 * Real.log ((Q + 4) / (δ' / 2)) / pAP ^ 2 := by positivity
    rw [hPB]; linarith
  have hPB1 : 1 ≤ PB := hcL1.trans hPBc
  have hPB0 : 0 ≤ PB := by linarith
  have hP₀1 : 1 ≤ P₀ := by
    have hx : 0 ≤ 2 * Θ + Bc := by linarith
    have h1 : 2 * Θ + Bc ≤ 256 * (2 * Θ + Bc) / (s ^ 2 * pAP) := by
      rw [le_div_iff₀ hsp0]; nlinarith only [hx, hsp]
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
  have hYle : (Q + 4) * PB / (δ' / 2 * (1 / 4) * (pAP / 2) * τ) ≤ (L + 1) ^ 2 * Y₀ := by
    have e : (Q + 4) * PB / (δ' / 2 * (1 / 4) * (pAP / 2) * τ)
        = (Q + 4) * PB * (1 / τ) / (δ' / 2 * (1 / 4) * (pAP / 2)) := by field_simp
    have e2 : (L + 1) ^ 2 * Y₀ = (Q + 4) * ((L + 1) * P₀) * (Θ * (L + 1))
        / (δ' / 2 * (1 / 4) * (pAP / 2)) := by rw [hY₀]; ring
    rw [e, e2]
    refine div_le_div_of_nonneg_right ?_ hD0.le
    have h1 : 0 ≤ (Q + 4) * PB := by positivity
    calc (Q + 4) * PB * (1 / τ) ≤ (Q + 4) * PB * (Θ * (L + 1)) :=
          mul_le_mul_of_nonneg_left hτΘ h1
      _ ≤ (Q + 4) * ((L + 1) * P₀) * (Θ * (L + 1)) := by
          gcongr
  have hY₀4 : 4 ≤ Real.log Y₀ := by
    rw [Real.le_log_iff_exp_le (by rw [hY₀]; positivity)]
    have he := Real.exp_one_lt_d9
    have : Real.exp 4 = Real.exp 1 ^ 4 := by rw [← Real.exp_nat_mul]; norm_num
    rw [this]
    have h1 : Real.exp 1 ^ 4 ≤ 3 ^ 4 := pow_le_pow_left₀ (Real.exp_pos 1).le (by linarith) 4
    have h2 : (Q + 4) * P₀ * Θ ≤ Y₀ := by
      rw [hY₀, le_div_iff₀ hD0]
      have : 0 ≤ (Q + 4) * P₀ * Θ := by positivity
      nlinarith only [this, hD1]
    have h3 : 5 * 1 * 3072 ≤ (Q + 4) * P₀ * Θ := by
      have := mul_le_mul (mul_le_mul (by linarith : (5 : ℝ) ≤ Q + 4) hP₀1 (by norm_num)
        (by linarith)) hΘ1 (by norm_num) (by positivity)
      linarith
    linarith
  have hlY0 : 0 ≤ Real.log ((Q + 4) * PB / (δ' / 2 * (1 / 4) * (pAP / 2) * τ)) :=
    Real.log_nonneg (by linarith)
  have hlY : Real.log ((Q + 4) * PB / (δ' / 2 * (1 / 4) * (pAP / 2) * τ))
      ≤ (L + 1) * Real.log Y₀ := by
    set u := Real.log (L + 1) with hu
    have hu0 : 0 ≤ u := Real.log_nonneg (by linarith)
    have huL : u ≤ L := by
      have := Real.log_le_sub_one_of_pos hℓ0
      rw [← hu] at this; linarith
    have h1 : Real.log ((Q + 4) * PB / (δ' / 2 * (1 / 4) * (pAP / 2) * τ))
        ≤ Real.log Y₀ + 2 * u := by
      have := Real.log_le_log (by linarith) hYle
      rw [Real.log_mul (by positivity) (by rw [hY₀]; positivity), Real.log_pow, Nat.cast_ofNat,
        ← hu] at this
      linarith
    nlinarith only [h1, hu0, huL, hY₀4]
  have hR₀0 : 0 ≤ R₀ := by rw [hR₀]; have := hY₀4; rw [← hY₀]; positivity
  have hRBle : RB ≤ (L + 1) ^ 4 * R₀ := by
    have hK : 0 ≤ 6000 * (Q + 2) ^ 2 / s ^ 4 := by positivity
    have e1 : RB = 6000 * (Q + 2) ^ 2 / s ^ 4 * (1 / τ) ^ 3
        * Real.log ((Q + 4) * PB / (δ' / 2 * (1 / 4) * (pAP / 2) * τ)) := by
      show 6000 * J ^ 2 * Real.log ((J + 2) * PB / (δ' / 2 * (1 / 4) * (pAP / 2) * τ))
        / (s ^ 4 * τ ^ 3) = _
      rw [hJ2, hJ]; field_simp
    have e2 : (L + 1) ^ 4 * R₀ = 6000 * (Q + 2) ^ 2 / s ^ 4 * (Θ * (L + 1)) ^ 3
        * ((L + 1) * Real.log Y₀) := by rw [hR₀, ← hY₀]; ring
    rw [e1, e2]
    have h1 : (1 / τ) ^ 3 ≤ (Θ * (L + 1)) ^ 3 := pow_le_pow_left₀ (by positivity) hτΘ 3
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
      have h1 : τ ^ 2 * (L + 1) ≤ 1 := by
        have : τ * (L + 1) ≤ 1 := (mul_le_mul_of_nonneg_right hτι hℓ0.le).trans hιL
        nlinarith only [this, hτ1, hτ0, hℓ0]
      have h2 : 6 * J ≤ 2 ^ 22 * (J + 1) ^ 2 := by nlinarith only [hJ1]
      have h3 : δ' * τ ^ 2 * (L + 1) ≤ 1 := by nlinarith only [h1, hδ'1, hδ'0, hτ0]
      have h4 := mul_le_mul_of_nonneg_right (mul_le_mul h3 h2 (by positivity) zero_le_one)
        (by linarith : (0 : ℝ) ≤ RB + 1)
      nlinarith only [h4]
  -- `4/(c ρ) ≤ (L + 1)⁶ E` for both
  have hEp : 0 < E := by rw [hE]; positivity
  have hℓ6 : 1 ≤ (L + 1) ^ 6 := one_le_pow₀ (by linarith)
  have hw : 0 ≤ Θ ^ 2 * (R₀ + 1) := by positivity
  have hw2 : 0 ≤ (P₀ + 1) / pAP := by positivity
  have hw3 : 0 ≤ A ^ 2 := by positivity
  have hb1 : 4 / (c1 * ρ) ≤ (L + 1) ^ 6 * E := by
    have e : 4 / (c1 * ρ) = 2 ^ 24 * (J + 1) ^ 2 * (RB + 1) * (1 / τ) ^ 2 / (δ' * ρ) := by
      rw [hc1]; field_simp; ring
    rw [e, hE]
    have hRB1 : RB + 1 ≤ (L + 1) ^ 4 * (R₀ + 1) := by
      have : (1 : ℝ) ≤ (L + 1) ^ 4 := one_le_pow₀ (by linarith)
      nlinarith only [hRBle, this, hR₀0]
    have hτ2' : (1 / τ) ^ 2 ≤ Θ ^ 2 * (L + 1) ^ 2 := by
      rw [← mul_pow]; exact pow_le_pow_left₀ (by positivity) hτΘ 2
    have hJ3 : J + 1 = Q + 3 := by rw [hJ]; ring
    rw [hJ3, ← mul_div_assoc]
    refine div_le_div_of_nonneg_right ?_ (by positivity)
    have h1 : (RB + 1) * (1 / τ) ^ 2 ≤ (L + 1) ^ 6 * (Θ ^ 2 * (R₀ + 1)) := by
      calc (RB + 1) * (1 / τ) ^ 2 ≤ ((L + 1) ^ 4 * (R₀ + 1)) * (Θ ^ 2 * (L + 1) ^ 2) :=
            mul_le_mul hRB1 hτ2' (by positivity) (by positivity)
        _ = (L + 1) ^ 6 * (Θ ^ 2 * (R₀ + 1)) := by ring
    have h2 : (L + 1) ^ 6 * (Θ ^ 2 * (R₀ + 1))
        ≤ (L + 1) ^ 6 * (Θ ^ 2 * (R₀ + 1) + (P₀ + 1) / pAP + A ^ 2) :=
      mul_le_mul_of_nonneg_left (by linarith) (by positivity)
    have h3 := mul_le_mul_of_nonneg_left (h1.trans h2)
      (by positivity : (0 : ℝ) ≤ 2 ^ 24 * (Q + 3) ^ 2)
    nlinarith only [h3]
  have hb3 : 4 / (c3 * ρ) ≤ (L + 1) ^ 6 * E := by
    have hJ3 : J + 1 = Q + 3 := by rw [hJ]; ring
    have e : 4 / (c3 * ρ) = 2 ^ 17 * (Q + 3) * ((PB + 1) / pAP) / (δ' * ρ) := by
      rw [hc3, hJ3]; field_simp; ring
    have e2 : (L + 1) ^ 6 * E = (L + 1) ^ 6 * (2 ^ 24 * (Q + 3) ^ 2
        * (Θ ^ 2 * (R₀ + 1) + (P₀ + 1) / pAP + A ^ 2)) / (δ' * ρ) := by rw [hE]; ring
    rw [e, e2]
    refine div_le_div_of_nonneg_right ?_ (by positivity)
    have h1 : (PB + 1) / pAP ≤ (L + 1) ^ 6 * ((P₀ + 1) / pAP) := by
      have : PB + 1 ≤ (L + 1) ^ 6 * (P₀ + 1) := by
        have hl : L + 1 ≤ (L + 1) ^ 6 := le_self_pow₀ (by linarith) (by norm_num)
        nlinarith only [hPBle, hl, hP₀1, hℓ6]
      rw [mul_div_assoc']
      exact div_le_div_of_nonneg_right this hp.le
    have h2 : (L + 1) ^ 6 * ((P₀ + 1) / pAP)
        ≤ (L + 1) ^ 6 * (Θ ^ 2 * (R₀ + 1) + (P₀ + 1) / pAP + A ^ 2) :=
      mul_le_mul_of_nonneg_left (by linarith) (by positivity)
    have h3 : 2 ^ 17 * (Q + 3) ≤ 2 ^ 24 * (Q + 3) ^ 2 := by nlinarith only [hQ]
    have h4 := mul_le_mul h3 (h1.trans h2) (by positivity) (by positivity)
    nlinarith only [h4]
  -- `L ≥ L₀` makes `A^n ≥ (L + 1)⁶ E`
  have hlA0 : 0 < Real.log A := Real.log_pos (by linarith)
  have hpow : (L + 1) ^ 6 * E ≤ A ^ n := by
    have h2 : 7 * Real.log (L + 1) ≤ 28 + (L + 1) * Real.log A / 8 := seven_log_le hA hℓ0
    have h3 : Real.log E + 24 ≤ (L - 2) * (25 * Real.log A) / 28 := by
      have := (le_sub_iff_add_le'.mpr hL₀)
      rw [div_le_iff₀ (by positivity)] at this
      linarith
    rw [← Real.log_le_log_iff (by positivity) (by positivity), Real.log_mul (by positivity)
      hEp.ne', Real.log_pow, Real.log_pow]
    have : (n : ℝ) = L - 1 := by rw [hLn]; ring
    rw [this]
    push_cast
    nlinarith only [h2, h3, hlA0]
  have hAn : 0 < A ^ n := by positivity
  refine ⟨?_, ?_⟩
  · have hc10 : 0 < c1 := by rw [hc1]; positivity
    have hc30 : 0 < c3 := by rw [hc3]; positivity
    have hm : 1 / A ^ n ≤ min c1 c3 * ρ / 4 := by
      rcases min_choice c1 c3 with h | h <;> rw [h]
      · rw [one_div_le (by positivity) (by positivity)]
        have : 1 / (c1 * ρ / 4) = 4 / (c1 * ρ) := by field_simp
        rw [this]; exact hb1.trans hpow
      · rw [one_div_le (by positivity) (by positivity)]
        have : 1 / (c3 * ρ / 4) = 4 / (c3 * ρ) := by field_simp
        rw [this]; exact hb3.trans hpow
    exact hm.trans (by gcongr)
  · -- level 5's prefix condition
    have hlg0 : 0 < (Q + 3) * A / δ' := by positivity
    have hlg : Real.log ((Q + 3) * A / δ') ≤ (Q + 3) * A / δ' := by
      linarith [Real.log_le_sub_one_of_pos hlg0]
    have hlg1 : 0 ≤ Real.log ((Q + 3) * A / δ') :=
      Real.log_nonneg (by rw [le_div_iff₀ hδ'0]; nlinarith only [hδ'1, hQ, hA])
    have hEn : 3072 * (Q + 2) * A * Real.log ((Q + 3) * A / δ') / ρ ≤ E := by
      rw [hE, div_le_div_iff₀ hρ0 (by positivity)]
      have h1 : 3072 * (Q + 2) * A * Real.log ((Q + 3) * A / δ') * δ'
          ≤ 3072 * (Q + 2) * (Q + 3) * A ^ 2 := by
        have := mul_le_mul_of_nonneg_left hlg (by positivity : (0 : ℝ) ≤ 3072 * (Q + 2) * A * δ')
        have e : 3072 * (Q + 2) * A * δ' * ((Q + 3) * A / δ')
            = 3072 * (Q + 2) * (Q + 3) * A ^ 2 := by
          field_simp
        nlinarith only [this, e]
      have h2 : 3072 * (Q + 2) * (Q + 3) * A ^ 2 ≤ 2 ^ 24 * (Q + 3) ^ 2 * A ^ 2 := by
        have : 3072 * (Q + 2) * (Q + 3) ≤ 2 ^ 24 * (Q + 3) ^ 2 := by nlinarith only [hQ]
        exact mul_le_mul_of_nonneg_right this hw3
      have h3 : 2 ^ 24 * (Q + 3) ^ 2 * A ^ 2
          ≤ 2 ^ 24 * (Q + 3) ^ 2 * (Θ ^ 2 * (R₀ + 1) + (P₀ + 1) / pAP + A ^ 2) :=
        mul_le_mul_of_nonneg_left (by linarith) (by positivity)
      have h4 := mul_le_mul_of_nonneg_right (h1.trans (h2.trans h3)) hρ0.le
      nlinarith only [h4]
    have hℓ1 : L + 1 ≤ (L + 1) ^ 6 := le_self_pow₀ (by linarith) (by norm_num)
    have h5 : (L + 1) * E ≤ A ^ n :=
      (mul_le_mul_of_nonneg_right hℓ1 hEp.le).trans hpow
    have h6 : 3072 * (Q + 2) * A * Real.log ((Q + 3) * A / δ') ≤ E * ρ := (div_le_iff₀ hρ0).1 hEn
    have h7 := mul_le_mul_of_nonneg_left h6 hℓ0.le
    have h8 := mul_le_mul_of_nonneg_right h5 hρ0.le
    have hεd' : εd = 16 * ρ / 3 := by rw [hρ]; ring
    rw [← mul_div_assoc, mul_one, div_le_iff₀ hAn, hεd']
    nlinarith only [h7, h8]

end Cap

set_option maxHeartbeats 1000000 in -- unfolding the simple claim's cap
theorem learner_loop_uniform_holds : LearnerLoopUniform := by
  intro α _ _ _ σ _ Ω _ μ _ M O P η₀ εcov εd δ pAP Q A L s δ' ρ V Kp G Θ P₀ R₀ E L₀
    hL hη hη₀ hε hε1 hεd hεd1 hδ hδ1 hpAP hA2 hD hpAPD hL₀
  classical
  have : Nonempty σ := ⟨M.start⟩
  have hQ1 : (1 : ℝ) ≤ Q := by
    show (1 : ℝ) ≤ (Fintype.card σ : ℝ); exact_mod_cast Fintype.card_pos
  have hA : (2 : ℝ) ≤ A := by show (2 : ℝ) ≤ (Fintype.card α : ℝ); exact_mod_cast hA2
  have hη0 : 0 ≤ η₀ := le_trans (eta_nonneg O) hη
  have hp1 : pAP ≤ 1 := hpAPD.trans measureReal_le_one
  have hδ'0 : 0 < δ' := by show 0 < δ / (2 * (Q + 1)); positivity
  have hδ'1 : δ' ≤ 1 := by
    show δ / (2 * (Q + 1)) ≤ 1
    rw [div_le_one (by positivity)]; linarith
  have hρ0 : 0 < ρ := by change 0 < 3 * εd / 16; positivity
  have hρ1 : ρ ≤ 1 := by change 3 * εd / 16 ≤ 1; linarith
  -- the band's far side, per `L + 1`
  have hV1 : 1 ≤ V := by
    change 1 ≤ (Q + 2) * ((Q + 2) * A + 1)
    have : 0 ≤ (Q + 2) * ((Q + 2) * A) := mul_nonneg (by linarith) (by nlinarith)
    nlinarith
  have hKp1 : 1 ≤ Kp := by
    change 1 ≤ V * (224 / εd + 1)
    have : 0 ≤ 224 / εd := by positivity
    nlinarith
  have hG1 : 1 ≤ G := by
    change 1 ≤ Real.log (4 * V * (2 * (Q + 2) * A + 4) / δ')
    have hX : 3 ≤ 4 * V * (2 * (Q + 2) * A + 4) / δ' := by
      rw [le_div_iff₀ hδ'0]
      have : (4 : ℝ) ≤ 2 * (Q + 2) * A + 4 := by nlinarith
      nlinarith
    rw [Real.le_log_iff_exp_le (by linarith)]
    linarith [Real.exp_one_lt_d9]
  set Bc : ℝ := 8 * Kp * G * (Q + 3) * A ^ (Fintype.card σ + 2) / δ' with hBcd
  have hBc1 : 1 ≤ Bc := by
    rw [hBcd, le_div_iff₀ hδ'0, one_mul]
    have hA1 : 1 ≤ A ^ (Fintype.card σ + 2) := one_le_pow₀ (by linarith)
    have h1 : 1 ≤ Kp * G := one_le_mul_of_one_le_of_one_le hKp1 hG1
    have h2 : 1 ≤ (Q + 3) * A ^ (Fintype.card σ + 2) :=
      one_le_mul_of_one_le_of_one_le (by linarith) hA1
    have h3 := mul_le_mul h1 h2 zero_le_one (by linarith)
    nlinarith
  -- the probes are at least 2 long
  have hL2 : 2 ≤ P.L := by
    have hs0' : 0 < 1 / 2 - η₀ := by linarith
    have hΘ1 : 1 ≤ Θ := by
      show 1 ≤ 1 / (1 / 2 - η₀) + 8 / εcov + 3072 * Q * (Q + 1)
      have : 0 < 1 / (1 / 2 - η₀) := one_div_pos.2 hs0'
      have : 0 < 8 / εcov := by positivity
      nlinarith [hQ1]
    have hs2 : (1 / 2 - η₀) ^ 2 * pAP ≤ 1 := by
      have : (1 / 2 - η₀) ^ 2 ≤ 1 := by nlinarith
      nlinarith
    have hlg : 0 ≤ Real.log ((Q + 4) / (δ' / 2)) :=
      Real.log_nonneg (by rw [le_div_iff₀ (by positivity)]; linarith)
    have hP1 : 1 ≤ P₀ := by
      change 1 ≤ 256 * (2 * Θ + Bc) / ((1 / 2 - η₀) ^ 2 * pAP)
        + 64 * Real.log ((Q + 4) / (δ' / 2)) / pAP ^ 2
      have h1 : 1 ≤ 256 * (2 * Θ + Bc) / ((1 / 2 - η₀) ^ 2 * pAP) := by
        rw [le_div_iff₀ (by positivity), one_mul]
        linarith
      have h2 : 0 ≤ 64 * Real.log ((Q + 4) / (δ' / 2)) / pAP ^ 2 := by positivity
      linarith
    have hR₀0 : 0 ≤ R₀ := by
      show 0 ≤ 6000 * (Q + 2) ^ 2 * Θ ^ 3
        * Real.log ((Q + 4) * P₀ * Θ / (δ' / 2 * (1 / 4) * (pAP / 2))) / (1 / 2 - η₀) ^ 4
      have : 0 ≤ Real.log ((Q + 4) * P₀ * Θ / (δ' / 2 * (1 / 4) * (pAP / 2))) := by
        refine Real.log_nonneg ?_
        rw [le_div_iff₀ (by positivity), one_mul]
        have h1 : δ' / 2 * (1 / 4) * (pAP / 2) ≤ 1 := by nlinarith
        have h2 : 1 ≤ (Q + 4) * P₀ * Θ :=
          one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le (by linarith) hP1) hΘ1
        linarith
      have : 0 ≤ Θ := by linarith
      positivity
    have hE1 : 1 ≤ E := by
      show 1 ≤ 2 ^ 24 * (Q + 3) ^ 2 * (Θ ^ 2 * (R₀ + 1) + (P₀ + 1) / pAP + A ^ 2) / (δ' * ρ)
      rw [le_div_iff₀ (by positivity), one_mul]
      have h1 : δ' * ρ ≤ 1 := mul_le_one₀ hδ'1 hρ0.le hρ1
      have h2 : 1 ≤ 2 ^ 24 * (Q + 3) ^ 2 * (Θ ^ 2 * (R₀ + 1) + (P₀ + 1) / pAP + A ^ 2) := by
        have : (1 : ℝ) ≤ (Q + 3) ^ 2 := by nlinarith only [hQ1]
        have : 1 ≤ A ^ 2 := by nlinarith only [hA]
        have : 0 ≤ Θ ^ 2 * (R₀ + 1) := by positivity
        have : 0 ≤ (P₀ + 1) / pAP := by positivity
        nlinarith
      linarith
    have hlA : 0 < Real.log A := Real.log_pos (by linarith)
    have : 0 ≤ 28 * (Real.log E + 24) / (25 * Real.log A) := by
      have := Real.log_nonneg hE1
      positivity
    have h : (2 : ℝ) ≤ P.L := le_trans (by linarith) hL₀
    exact_mod_cast h
  have hLn : L = ((P.L - 1 : ℕ) : ℝ) + 1 := by
    show (P.L : ℝ) = _
    rw [Nat.cast_sub (by omega)]; ring
  obtain ⟨hcap, hnoise⟩ := cap_bound (ι := min (εcov / 8) (1 / (3072 * (L + 1) * (Q + 1) * Q)))
    Bc (8 * Kp * G * (L + 1) * (Q + 3) * A ^ (Fintype.card σ + 2) / δ')
    hQ1 hA hLn hη0 hη₀ hε hε1 hεd hεd1 hδ hδ1 hpAP hp1 rfl rfl rfl rfl hBc1 (by rw [hBcd]; ring)
    rfl rfl rfl rfl hL₀
  have hpre : prefixMax P.D (P.L - 1) ≤ 1 / A ^ (P.L - 1) := by
    rw [hD]; exact prefixMax_uniformStrings_le (Nat.sub_le _ _)
  obtain ⟨settings, hw, hd, hb, hmain⟩ := learner_loop_simple_holds M O P η₀ εcov εd δ pAP
    (P.L - 1) hL hη hη₀ hε hε1 hεd hεd1 hδ hδ1 hpAP (by omega) (by omega)
    (hpre.trans hcap)
    ((mul_le_mul_of_nonneg_left hpre (by
      have : 0 ≤ Real.log ((Q + 3) * A / δ') :=
        Real.log_nonneg (by rw [le_div_iff₀ hδ'0]; nlinarith only [hδ'1, hQ1, hA])
      have : (0 : ℝ) ≤ Q + 2 := by linarith
      have : (0 : ℝ) ≤ A := by linarith
      have : (0 : ℝ) ≤ L + 1 := by positivity
      positivity)).trans hnoise)
  refine ⟨settings, hw, hd, hb, ?_⟩
  have hA0 : 0 < 1 / A ^ (P.L - 1) := by positivity
  have key : ∀ c : ℝ, 1 / A ^ (P.L - 1) ≤ c * ρ / 4 → c * ρ / 4 ≤ c := fun c hc => by
    have h0 : 0 < c * ρ := by linarith [hA0.trans_le hc]
    have hc0 : 0 < c := pos_of_mul_pos_left h0 hρ0.le
    nlinarith
  refine hmain _ inferInstance (uniformStrings_length _) hpAPD fun v => ?_
  refine (uniformStrings_singleton_le _ v).trans (le_trans ?_ (hcap.trans (key _ hcap)))
  exact one_div_pow_le_one_div_pow_of_le (le_trans (by norm_num) hA) (by omega)

end OrthoDFA
