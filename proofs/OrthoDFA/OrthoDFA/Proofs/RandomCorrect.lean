import OrthoDFA.Proofs.RandomRound

/-!
# Level 3's claim: the proofs

`RandomRoundCorrect` is `round_le_at` at the knobs `Ns = N₁ = ⌈28Λ / ε⌉`,
`Λ = log(16 S (S + 1) / δ)`, `θr = ε/28`, `εd' = 3ε/7`, `θpt' = 7/8`, and the claim's `G`. Then
`noiseRisk` is `δ/4`, and under the `j`-th hypothesis each way a stretch outlives `Ns` probes has
chance at most half its tests' level `a_j` (`stretch_le`), as `Λ ≥ log(2 / a_j)`, so a
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

/-- Where `Λ ≥ log(2 / a_j)` and `Ns ≥ 28Λ / ε`, a stretch at the `j`-th hypothesis risks at most
`6 a_j` at the knobs `θr = ε/28`, `εd' = 3ε/7` and `θpt' = 7/8`. -/
theorem stretch_le (C : Level3Settings) (hδ : 0 < C.δ) (hρ0 : 0 < C.disagreementRate)
    (hρ1 : C.disagreementRate < 1) {Λ : ℝ} {Ns j : ℕ} (hΛ0 : 0 ≤ Λ)
    (hΛ : Real.log (2 / C.level j) ≤ Λ) (hNs : 28 * Λ / C.disagreementRate ≤ Ns) :
    stretchRisk C (C.level j) Ns Ns (C.disagreementRate / 28) (3 * C.disagreementRate / 7) (7 / 8)
      ≤ 6 * C.level j := by
  have ha : 0 < C.level j := level_pos C hδ j
  unfold stretchRisk Level3Settings.disagreementAlternative Level3Settings.middleAlternative
    Level3Settings.middleRate Level3Settings.searchShare
  set a := C.level j
  set ρ := C.disagreementRate
  set k := ⌈ρ / 4 * Ns⌉₊
  have hN0 : (0 : ℝ) ≤ Ns := Nat.cast_nonneg _
  have hρN : 28 * Λ ≤ ρ * Ns := by rw [div_le_iff₀ hρ0] at hNs; linarith
  have hE : Real.exp (-Λ) ≤ a / 2 := by
    calc Real.exp (-Λ) ≤ Real.exp (-Real.log (2 / a)) := Real.exp_le_exp.2 (by linarith)
      _ = a / 2 := by rw [Real.exp_neg, Real.exp_log (by positivity), inv_div]
  have hE2 : Real.exp (-(2 * Λ)) ≤ a ^ 2 / 4 := by
    rw [show -(2 * Λ) = -Λ + -Λ by ring, Real.exp_add]
    have := Real.exp_pos (-Λ)
    nlinarith
  -- no clean disagreement
  have h2 : (1 - ρ / 28) ^ Ns ≤ a / 2 := by
    refine (pow_le_exp (c := -(ρ / 28)) (by linarith)
      (by linarith [Real.add_one_le_exp (-(ρ / 28))]) Ns).trans
      (le_trans (Real.exp_le_exp.2 ?_) hE)
    nlinarith
  -- success silent
  have hBD : 3 * ρ / 7 * (ρ / (ρ / 2)) + (1 - 3 * ρ / 7) * ((1 - ρ) / (1 - ρ / 2))
      ≤ 1 - ρ / 14 := by
    have e : ρ / (ρ / 2) = 2 := by field_simp
    rw [e]
    have hd : 0 < 1 - ρ / 2 := by linarith
    have : (1 - 3 * ρ / 7) * ((1 - ρ) / (1 - ρ / 2)) ≤ 1 - 13 * ρ / 14 := by
      rw [← mul_div_assoc, div_le_iff₀ hd]; nlinarith
    linarith
  have hBD0 : 0 ≤ 3 * ρ / 7 * (ρ / (ρ / 2)) + (1 - 3 * ρ / 7) * ((1 - ρ) / (1 - ρ / 2)) := by
    have : 0 ≤ (1 - ρ) / (1 - ρ / 2) := div_nonneg (by linarith) (by linarith)
    have : 0 ≤ 1 - 3 * ρ / 7 := by linarith
    positivity
  have h3 : (3 * ρ / 7 * (ρ / (ρ / 2)) + (1 - 3 * ρ / 7) * ((1 - ρ) / (1 - ρ / 2))) ^ Ns / a
      ≤ a / 2 := by
    rw [div_le_iff₀ ha]
    refine (pow_le_exp (c := -(ρ / 14)) hBD0
      (hBD.trans (by linarith [Real.add_one_le_exp (-(ρ / 14))])) Ns).trans
      (le_trans (Real.exp_le_exp.2 ?_) (hE2.trans (by nlinarith)))
    nlinarith
  -- too few searches
  have hk1 : ((k - 1 : ℕ) : ℝ) ≤ ρ / 4 * Ns := by
    rcases Nat.eq_zero_or_pos k with h | h
    · rw [h]; simp only [Nat.zero_sub, Nat.cast_zero]; positivity
    · rw [Nat.cast_sub h, Nat.cast_one]
      linarith [Nat.ceil_lt_add_one (show 0 ≤ ρ / 4 * Ns by positivity)]
  have hlog2 : Real.log 2 ≤ 7 / 10 := by linarith [Real.log_two_lt_d9]
  have hlog20 : 0 ≤ Real.log 2 := Real.log_nonneg (by norm_num)
  have h4 : (2 : ℝ) ^ (k - 1) * (1 - 3 * ρ / 7 / 2) ^ Ns ≤ a / 2 := by
    have e2 : (2 : ℝ) ^ (k - 1) = Real.exp ((k - 1 : ℕ) * Real.log 2) := by
      rw [Real.exp_nat_mul, Real.exp_log two_pos]
    rw [e2]
    have hp := pow_le_exp (c := -(3 * ρ / 7 / 2)) (show 0 ≤ 1 - 3 * ρ / 7 / 2 by linarith)
      (by linarith [Real.add_one_le_exp (-(3 * ρ / 7 / 2))]) Ns
    have hkl : ((k - 1 : ℕ) : ℝ) * Real.log 2 ≤ ρ / 4 * Ns * (7 / 10) :=
      mul_le_mul hk1 hlog2 hlog20 (by positivity)
    calc _ ≤ Real.exp ((k - 1 : ℕ) * Real.log 2) * Real.exp (Ns * -(3 * ρ / 7 / 2)) :=
          mul_le_mul_of_nonneg_left hp (Real.exp_pos _).le
      _ = Real.exp ((k - 1 : ℕ) * Real.log 2 + Ns * -(3 * ρ / 7 / 2)) := by rw [Real.exp_add]
      _ ≤ Real.exp (-Λ) := Real.exp_le_exp.2 (by nlinarith)
      _ ≤ a / 2 := hE
  -- middles silent
  have hBM : (7 / 8 : ℝ) * (1 / 4 / (2 * (1 / 4)))
      + (1 - 7 / 8) * ((1 - 1 / 4) / (1 - 2 * (1 / 4))) = 5 / 8 := by norm_num
  have h5 : ((7 / 8 : ℝ) * (1 / 4 / (2 * (1 / 4)))
      + (1 - 7 / 8) * ((1 - 1 / 4) / (1 - 2 * (1 / 4)))) ^ k / a ≤ a / 2 := by
    rw [hBM, div_le_iff₀ ha]
    have hk : ρ / 4 * Ns ≤ k := Nat.le_ceil _
    refine (pow_le_exp (c := -(1 / 3)) (by norm_num)
      (by linarith [Real.add_one_le_exp (-(1 / 3 : ℝ))]) k).trans
      (le_trans (Real.exp_le_exp.2 ?_) (hE2.trans (by nlinarith)))
    nlinarith
  linarith

/-- Under level 3's premises at `δ < 1` and a positive `G`, the knobs `RandomRoundCorrect` picks
are admissible and their stretches risk at most `3δ/4` in all. -/
theorem knobs_exist {α : Type*} [Fintype α] {σ : Type*} [Fintype σ] (C : Level3Settings)
    (θ G : ℝ) (S : ℕ)
    (hS : S = Fintype.card σ * ((Fintype.card σ + 2) * Fintype.card α + 1) + 2 * Fintype.card α + 1)
    (hθ : 0 ≤ θ) (hδ : 0 < C.δ) (hδ1 : C.δ < 1) (hG0 : 0 < G) (hGy : 2 * G ≤ C.harvestRate)
    (hyρ : C.harvestRate ≤ C.disagreementRate / 32) (hρ1 : C.disagreementRate < 1)
    (hP : 28 * S / C.disagreementRate * Real.log (16 * S * (S + 1) / C.δ) + S ≤ C.probeBudget) :
    ∃ kn : Knobs, kn.Admissible (σ := σ) (α := α) θ C C.disagreementRate C.harvestRate
      ∧ kn.G = G
      ∧ ∑ j ∈ Finset.range S, stretchRisk C (C.level j) kn.Ns kn.N₁ kn.θr kn.εd' kn.θpt'
        ≤ 3 * C.δ / 4 := by
  set δ := C.δ
  set ρ := C.disagreementRate
  set y := C.harvestRate
  have hy0 : 0 < y := by linarith
  have hρ0 : 0 < ρ := by linarith
  have hpy : max (2 * G) y = y := max_eq_right hGy
  have hS1 : (1 : ℝ) ≤ S := by
    have : 1 ≤ S := by omega
    exact_mod_cast this
  set Λ := Real.log (16 * S * (S + 1) / δ) with hΛ
  have hΛ0 : 0 < Λ := Real.log_pos (by rw [one_lt_div hδ]; nlinarith)
  set Ns := ⌈28 * Λ / ρ⌉₊ with hNs
  have hNs0 : 1 ≤ Ns := by
    rw [hNs, Nat.one_le_ceil_iff]
    positivity
  refine ⟨⟨Ns, Ns, G, ρ / 28, 3 * ρ / 7, 7 / 8⟩, ?_, rfl, ?_⟩
  · refine ⟨hNs0, le_rfl, hθ, hG0.le, hδ, hy0, show y < 1 / 2 by linarith,
      show (0 : ℝ) ≤ 1 / 4 by norm_num, show (0 : ℝ) < 1 / 4 by norm_num,
      show (1 / 4 : ℝ) < 1 / 2 by norm_num, hρ0, hρ1, le_rfl,
      show (0 : ℝ) ≤ ρ / 28 by positivity, show ρ / 28 ≤ 1 by linarith,
      show (0 : ℝ) ≤ 3 * ρ / 7 by positivity, show 3 * ρ / 7 ≤ 1 by linarith,
      show 2 * (1 / 4 : ℝ) ≤ 7 / 8 by norm_num, show (7 / 8 : ℝ) ≤ 1 by norm_num,
      show ρ / 28 ≤ (1 - 7 / 8) * (3 * ρ / 7) by linarith, ?_,
      show 4 * G ≤ 1 / 4 * (8 * y) by linarith, show 2 * y ≤ 1 / 4 * (8 * y) by linarith,
      show max (2 * G) y ≤ y from hpy.le, ?_⟩
    · change _ * Ns ≤ C.probeBudget
      rw [← hS]
      have hlt : (Ns : ℝ) < 28 * Λ / ρ + 1 :=
        Nat.ceil_lt_add_one (by positivity)
      have : ((S * Ns : ℕ) : ℝ) ≤ C.probeBudget := by
        push_cast
        calc (S : ℝ) * Ns ≤ S * (28 * Λ / ρ + 1) := by gcongr
          _ = 28 * S / ρ * Λ + S := by ring
          _ ≤ _ := hP
      exact_mod_cast this
    · exact middles_fall_of (μ := C.middleRate) (s := C.searchShare) (by positivity)
        (show (0 : ℝ) < 1 / 4 by norm_num) (show (1 / 4 : ℝ) ≤ 1 / 4 from le_rfl)
        (show 0 ≤ ρ / 4 by positivity)
        (show 2 * max (2 * G) y ≤ ρ / 4 * (1 / 4) by rw [hpy]; linarith)
  · change ∑ j ∈ Finset.range S, stretchRisk C (C.level j) Ns Ns (ρ / 28) (3 * ρ / 7) (7 / 8)
      ≤ 3 * δ / 4
    calc _ ≤ ∑ j ∈ Finset.range S, 6 * C.level j := by
          refine Finset.sum_le_sum fun j hj => ?_
          refine stretch_le C hδ hρ0 hρ1 hΛ0.le ?_ (Nat.le_ceil _)
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

theorem randomRoundCorrect_holds : RandomRoundCorrect := by
  intro α _ _ σ _ M side U θ Ω _ μ _ R P C p₀
  dsimp only
  intro hpre hθ hδ hGy hyρ hρ1 hP
  set nQ := Fintype.card σ
  set nα := Fintype.card α
  set δ := C.δ
  set K : ℝ := (1 + (nQ + 1) ^ 3 * nα) ^ nQ with hK
  set G := 3 * θ * (P.L + 1) * (nQ + 1) + p₀ * Real.log (4 * K / δ) with hG
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
  obtain ⟨kn, hA, hkG, hsum⟩ := knobs_exist (σ := σ) (α := α) C θ G _ rfl hθ hδ hδ1 hG0 hGy hyρ
    hρ1 (by exact_mod_cast hP)
  refine (round_le_at M side U θ μ R P C _ _ p₀ kn hA hp₀ (fun u _ => hpre u)).trans
    (ENNReal.ofReal_le_ofReal ?_)
  unfold roundBoundAt
  have hnoise : noiseRisk nQ nα P.L (3 / 2 * θ) p₀ G = δ / 4 := by
    unfold noiseRisk
    have e : -(G - 2 * (3 / 2 * θ) * (P.L + 1) * (nQ + 1)) / p₀ = -Real.log (4 * K / δ) := by
      rw [hG]; field_simp; ring
    push_cast
    rw [← hK, e, Real.exp_neg, Real.exp_log (by positivity)]
    field_simp
  rw [hkG, hnoise]
  linarith

end Random

end OrthoDFA
