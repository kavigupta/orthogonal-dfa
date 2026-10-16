import OrthoDFA.Proofs.RandomRound

/-!
# Level 3's claim: the proofs

`RandomRoundCorrect` is `round_le_at` at the knobs `Ns = N₁ = ⌈8Λ / searchShare⌉`,
`Λ = log(16 S (S + 1) / δ)`, `θr = ε/16`, `εd' = ε/4`, `θpt' = 1/2 + middleRate`, and the claim's
`G`. Then `noiseRisk` is `δ/4`, and under the `j`-th hypothesis each way a stretch outlives `Ns`
probes has chance at most half its tests' level `a_j` (`stretch_le`), as `Λ ≥ log(2 / a_j)`, so a
stretch risks at most `6 a_j`, and these sum to at most `3δ/4`.
-/

namespace OrthoDFA

namespace Random

open MeasureTheory

theorem sum_level (C : Level3Settings) (N : ℕ) :
    ∑ j ∈ Finset.range N, C.level j = C.δ / 8 * (N / (N + 1)) := by
  induction N with
  | zero => simp
  | succ N ih =>
    rw [Finset.sum_range_succ, ih]
    unfold Level3Settings.level
    push_cast
    field_simp
    ring

/-- `middles_fall` at the alternative `2μ`. -/
theorem middles_fall_of {p μ s : ℝ} (hp : 0 ≤ p) (hμ0 : 0 < μ) (hμ : μ ≤ 1 / 4) (hs : 0 ≤ s)
    (h : 2 * p ≤ s * μ) :
    p * (2 * μ / μ) * ((1 - 2 * μ) / (1 - μ)) ^ (s - 1)
      + (1 - p) * ((1 - 2 * μ) / (1 - μ)) ^ s ≤ 1 := by
  set r := (1 - 2 * μ) / (1 - μ) with hr
  have h1μ : 0 < 1 - μ := by linarith
  have h12μ : 0 < 1 - 2 * μ := by linarith
  have hr0 : 0 < r := div_pos h12μ h1μ
  have hrs : r ^ (s - 1) = r ^ s / r := Real.rpow_sub_one hr0.ne' s
  have hsum : p * (2 * μ / μ) * (r ^ s / r) + (1 - p) * r ^ s
      = r ^ s * (1 + p / (1 - 2 * μ)) := by
    rw [hr]; field_simp; ring
  rw [hrs, hsum]
  have hrpow : r ^ s ≤ Real.exp (-(s * μ)) := by
    rw [Real.rpow_def_of_pos hr0]
    apply Real.exp_le_exp.2
    have hl : Real.log r ≤ r - 1 := Real.log_le_sub_one_of_pos hr0
    have hr1 : r - 1 ≤ -μ := by
      rw [hr, div_sub_one h1μ.ne', div_le_iff₀ h1μ]; nlinarith
    nlinarith
  have hq0 : 0 ≤ 1 + p / (1 - 2 * μ) := by
    have := div_nonneg hp h12μ.le
    linarith
  have hq : 1 + p / (1 - 2 * μ) ≤ Real.exp (2 * p) := by
    have : p / (1 - 2 * μ) ≤ 2 * p := by
      rw [div_le_iff₀ h12μ]; nlinarith
    linarith [Real.add_one_le_exp (2 * p)]
  calc r ^ s * (1 + p / (1 - 2 * μ)) ≤ Real.exp (-(s * μ)) * Real.exp (2 * p) :=
        mul_le_mul hrpow hq hq0 (Real.exp_pos _).le
    _ = Real.exp (2 * p - s * μ) := by rw [← Real.exp_add]; ring_nf
    _ ≤ 1 := Real.exp_le_one_iff.2 (by linarith)

theorem pow_le_exp {b c : ℝ} (hb : 0 ≤ b) (h : b ≤ Real.exp c) (n : ℕ) :
    b ^ n ≤ Real.exp (n * c) := by
  rw [Real.exp_nat_mul]
  exact pow_le_pow_left₀ hb h n

/-- Where `Λ ≥ log(2 / a_j)` and `Ns ≥ 8Λ / searchShare`, a stretch at the `j`-th hypothesis risks
at most `6 a_j` at the knobs `θr = ε/16`, `εd' = ε/4` and `θpt' = 1/2 + middleRate`. -/
theorem stretch_le (C : Level3Settings) (hδ : 0 < C.δ) (hss : 0 < C.searchShare)
    (hρ : 8 * C.searchShare ≤ C.disagreementRate) (hρ1 : C.disagreementRate < 1)
    (hμ0 : 0 < C.middleRate) (hμ : C.middleRate ≤ 1 / 4) {Λ : ℝ} {Ns j : ℕ} (hΛ0 : 0 ≤ Λ)
    (hΛ : Real.log (2 / C.level j) ≤ Λ) (hNs : 8 * Λ / C.searchShare ≤ Ns) :
    stretchRisk C (C.level j) Ns Ns (C.disagreementRate / 16) (C.disagreementRate / 4)
      (1 / 2 + C.middleRate) ≤ 6 * C.level j := by
  have ha : 0 < C.level j := level_pos C hδ j
  unfold stretchRisk Level3Settings.disagreementAlternative Level3Settings.middleAlternative
  set a := C.level j
  set ρ := C.disagreementRate
  set μ := C.middleRate
  set ss := C.searchShare
  set k := ⌈ss * Ns⌉₊
  have hρ0 : 0 < ρ := by linarith
  have hN0 : (0 : ℝ) ≤ Ns := Nat.cast_nonneg _
  have hssN : 8 * Λ ≤ ss * Ns := by rw [div_le_iff₀ hss] at hNs; linarith
  have hρN : 64 * Λ ≤ ρ * Ns := by nlinarith
  have hE : Real.exp (-Λ) ≤ a / 2 := by
    calc Real.exp (-Λ) ≤ Real.exp (-Real.log (2 / a)) := Real.exp_le_exp.2 (by linarith)
      _ = a / 2 := by rw [Real.exp_neg, Real.exp_log (by positivity), inv_div]
  have hE2 : Real.exp (-(2 * Λ)) ≤ a ^ 2 / 4 := by
    rw [show -(2 * Λ) = -Λ + -Λ by ring, Real.exp_add]
    have := Real.exp_pos (-Λ)
    nlinarith
  -- no clean disagreement
  have h2 : (1 - ρ / 16) ^ Ns ≤ a / 2 := by
    refine (pow_le_exp (c := -(ρ / 16)) (by linarith)
      (by linarith [Real.add_one_le_exp (-(ρ / 16))]) Ns).trans
      (le_trans (Real.exp_le_exp.2 ?_) hE)
    nlinarith
  -- success silent
  have hBD : ρ / 4 * (ρ / (ρ / 2)) + (1 - ρ / 4) * ((1 - ρ) / (1 - ρ / 2)) ≤ 1 - ρ / 4 := by
    have e : ρ / (ρ / 2) = 2 := by field_simp
    rw [e]
    have hd : 0 < 1 - ρ / 2 := by linarith
    have : (1 - ρ / 4) * ((1 - ρ) / (1 - ρ / 2)) ≤ 1 - 3 * ρ / 4 := by
      rw [← mul_div_assoc, div_le_iff₀ hd]; nlinarith
    linarith
  have hBD0 : 0 ≤ ρ / 4 * (ρ / (ρ / 2)) + (1 - ρ / 4) * ((1 - ρ) / (1 - ρ / 2)) := by
    have : 0 ≤ (1 - ρ) / (1 - ρ / 2) := div_nonneg (by linarith) (by linarith)
    have : 0 ≤ 1 - ρ / 4 := by linarith
    positivity
  have h3 : (ρ / 4 * (ρ / (ρ / 2)) + (1 - ρ / 4) * ((1 - ρ) / (1 - ρ / 2))) ^ Ns / a ≤ a / 2 := by
    rw [div_le_iff₀ ha]
    refine (pow_le_exp (c := -(ρ / 4)) hBD0
      (hBD.trans (by linarith [Real.add_one_le_exp (-(ρ / 4))])) Ns).trans
      (le_trans (Real.exp_le_exp.2 ?_) (hE2.trans (by nlinarith)))
    nlinarith
  -- too few searches
  have hk1 : ((k - 1 : ℕ) : ℝ) ≤ ss * Ns := by
    rcases Nat.eq_zero_or_pos k with h | h
    · rw [h]; simp only [Nat.zero_sub, Nat.cast_zero]; positivity
    · rw [Nat.cast_sub h, Nat.cast_one]
      linarith [Nat.ceil_lt_add_one (show 0 ≤ ss * Ns by positivity)]
  have hlog2 : Real.log 2 ≤ 3 / 4 := by linarith [Real.log_two_lt_d9]
  have hlog20 : 0 ≤ Real.log 2 := Real.log_nonneg (by norm_num)
  have h4 : (2 : ℝ) ^ (k - 1) * (1 - ρ / 4 / 2) ^ Ns ≤ a / 2 := by
    have e2 : (2 : ℝ) ^ (k - 1) = Real.exp ((k - 1 : ℕ) * Real.log 2) := by
      rw [Real.exp_nat_mul, Real.exp_log two_pos]
    rw [e2]
    have hp := pow_le_exp (c := -(ρ / 4 / 2)) (show 0 ≤ 1 - ρ / 4 / 2 by linarith)
      (by linarith [Real.add_one_le_exp (-(ρ / 4 / 2))]) Ns
    calc _ ≤ Real.exp ((k - 1 : ℕ) * Real.log 2) * Real.exp (Ns * -(ρ / 4 / 2)) :=
          mul_le_mul_of_nonneg_left hp (Real.exp_pos _).le
      _ = Real.exp ((k - 1 : ℕ) * Real.log 2 + Ns * -(ρ / 4 / 2)) := by rw [Real.exp_add]
      _ ≤ Real.exp (-Λ) := Real.exp_le_exp.2 (by nlinarith)
      _ ≤ a / 2 := hE
  -- middles silent
  have hBM : (1 / 2 + μ) * (μ / (2 * μ)) + (1 - (1 / 2 + μ)) * ((1 - μ) / (1 - 2 * μ)) = 3 / 4 := by
    have : (1 : ℝ) - 2 * μ ≠ 0 := by linarith
    field_simp
    ring
  have h5 : ((1 / 2 + μ) * (μ / (2 * μ)) + (1 - (1 / 2 + μ)) * ((1 - μ) / (1 - 2 * μ))) ^ k / a
      ≤ a / 2 := by
    rw [hBM, div_le_iff₀ ha]
    have hk : ss * Ns ≤ k := Nat.le_ceil _
    refine (pow_le_exp (c := -(1 / 4)) (by norm_num)
      (by linarith [Real.add_one_le_exp (-(1 / 4 : ℝ))]) k).trans
      (le_trans (Real.exp_le_exp.2 ?_) (hE2.trans (by nlinarith)))
    nlinarith
  linarith

theorem randomRoundCorrect_holds : RandomRoundCorrect := by
  intro α _ _ σ _ M side U θ Ω _ μ _ R P C y p₀
  dsimp only
  intro hpre hθ hδ hss hssρ hρ1 hμ hs1 he0 hps hpe hpm hP
  set nQ := Fintype.card σ
  set nα := Fintype.card α
  set S : ℕ := nQ * ((nQ + 2) * nα + 1) + 2 * nα + 1 with hS
  set δ := C.δ
  set ρ := C.disagreementRate
  set K : ℝ := (1 + (nQ + 1) ^ 3 * nα) ^ nQ with hK
  set G := 3 * θ * (P.L + 1) * (nQ + 1) + p₀ * Real.log (4 * K / δ) with hG
  set p := max (2 * G) y with hp
  by_cases hδ1 : 1 ≤ δ
  · calc _ ≤ ∫⁻ _, 1 ∂μ := lintegral_mono fun _ => prob_le_one
      _ = 1 := by simp
      _ ≤ _ := by rw [← ENNReal.ofReal_one]; exact ENNReal.ofReal_le_ofReal hδ1
  push Not at hδ1
  have hp₀ : 0 < p₀ :=
    (prefixMax_pos P.D C.walkStart).trans_le (ciSup_le hpre)
  have hK1 : 1 ≤ K := one_le_pow₀ (by
    have : (0 : ℝ) ≤ (nQ + 1) ^ 3 * nα := by positivity
    linarith)
  have hlogK : 0 < Real.log (4 * K / δ) :=
    Real.log_pos (by rw [one_lt_div hδ]; linarith)
  have hG0 : 0 < G := by
    have : 0 ≤ 3 * θ * (P.L + 1) * (nQ + 1) := by positivity
    have : 0 < p₀ * Real.log (4 * K / δ) := mul_pos hp₀ hlogK
    linarith
  have hpG : 2 * G ≤ p := le_max_left _ _
  have hpy : y ≤ p := le_max_right _ _
  have hμ0 : 0 < C.middleRate := by
    by_contra h
    push Not at h
    nlinarith
  have hρ0 : 0 < ρ := by linarith
  have hS1 : (1 : ℝ) ≤ S := by
    have : 1 ≤ S := by omega
    exact_mod_cast this
  set Λ := Real.log (16 * S * (S + 1) / δ) with hΛ
  have hΛ0 : 0 ≤ Λ := Real.log_nonneg (by rw [le_div_iff₀ hδ]; nlinarith)
  set Ns := ⌈8 * Λ / C.searchShare⌉₊ with hNs
  have hNs0 : 1 ≤ Ns := by
    rw [hNs, Nat.one_le_ceil_iff]
    have : 0 < Λ := Real.log_pos (by rw [one_lt_div hδ]; nlinarith)
    positivity
  set kn : Knobs := ⟨Ns, Ns, G, ρ / 16, ρ / 4, 1 / 2 + C.middleRate⟩
  have hA : kn.Admissible (σ := σ) (α := α) θ C ρ y := by
    refine ⟨hNs0, le_rfl, hθ, hG0.le, hδ, by linarith, hs1, he0, hμ0, by linarith, hρ0, hρ1,
      le_rfl, show (0 : ℝ) ≤ ρ / 16 by positivity, show ρ / 16 ≤ 1 by linarith,
      show (0 : ℝ) ≤ ρ / 4 by positivity, show ρ / 4 ≤ 1 by linarith,
      show 2 * C.middleRate ≤ 1 / 2 + C.middleRate by linarith,
      show 1 / 2 + C.middleRate ≤ 1 by linarith, ?_, ?_, show 4 * G ≤ _ by linarith,
      by linarith, hps, ?_⟩
    · show ρ / 16 ≤ (1 - (1 / 2 + C.middleRate)) * (ρ / 4)
      nlinarith
    · show S * Ns ≤ C.probeBudget
      have hlt : (Ns : ℝ) < 8 * Λ / C.searchShare + 1 :=
        Nat.ceil_lt_add_one (by positivity)
      have : ((S * Ns : ℕ) : ℝ) ≤ C.probeBudget := by
        push_cast
        calc (S : ℝ) * Ns ≤ S * (8 * Λ / C.searchShare + 1) := by gcongr
          _ = 8 * S / C.searchShare * Λ + S := by ring
          _ ≤ _ := hP
      exact_mod_cast this
    · exact middles_fall_of (by linarith) hμ0 hμ hss.le hpm
  refine (round_le_at M side U θ μ R P C ρ y p₀ kn hA hp₀ (fun u _ => hpre u)).trans
    (ENNReal.ofReal_le_ofReal ?_)
  unfold roundBoundAt
  have hnoise : noiseRisk nQ nα P.L (3 / 2 * θ) p₀ G = δ / 4 := by
    unfold noiseRisk
    have e : -(G - 2 * (3 / 2 * θ) * (P.L + 1) * (nQ + 1)) / p₀ = -Real.log (4 * K / δ) := by
      rw [hG]; field_simp; ring
    push_cast
    rw [← hK, e, Real.exp_neg, Real.exp_log (by positivity)]
    field_simp
  have hsum : ∑ j ∈ Finset.range S,
      stretchRisk C (C.level j) Ns Ns (ρ / 16) (ρ / 4) (1 / 2 + C.middleRate) ≤ 3 * δ / 4 := by
    calc _ ≤ ∑ j ∈ Finset.range S, 6 * C.level j := by
          refine Finset.sum_le_sum fun j hj => ?_
          refine stretch_le C hδ hss hssρ hρ1 hμ0 hμ hΛ0 ?_ (Nat.le_ceil _)
          have hj : (j : ℝ) + 1 ≤ S := by
            have : j + 1 ≤ S := Finset.mem_range.1 hj
            exact_mod_cast this
          have ha := level_pos C hδ j
          apply Real.log_le_log (by positivity)
          unfold Level3Settings.level
          rw [div_div_eq_mul_div, div_le_div_iff_of_pos_right hδ]
          have hj0 : (0 : ℝ) ≤ j := Nat.cast_nonneg j
          nlinarith
      _ = 6 * (δ / 8 * (S / (S + 1))) := by rw [← Finset.mul_sum, sum_level]
      _ ≤ 3 * δ / 4 := by
          have : (S : ℝ) / (S + 1) ≤ 1 := by rw [div_le_one (by positivity)]; linarith
          nlinarith
  show noiseRisk nQ nα P.L (3 / 2 * θ) p₀ G + _ ≤ δ
  rw [hnoise]
  linarith

end Random

end OrthoDFA
