import OrthoDFA.Proofs.RandomRound

/-!
# Level 3's claim: the proofs

`Level3RoundCorrect` is `round_le_at` at the knobs `Ns = N₁ = ⌈36Λ / ε⌉`,
`Λ = log(16 S (S + 1) / δ)`, `θr = ε/36`, `εd' = 4ε/9`, `θe' = 113ε/720`, `θpt' = 7/8`, `Ge = 0`,
`θgpt = 1/16`, `ηm = 0` and `pM = ε/48`. Then the tails at every state's rate risk `δ/8` at each
site, and under the `j`-th hypothesis each way a
stretch outlives `Ns` probes has chance at most half its tests' level `a_j` (`stretch_le`), as
`Λ ≥ log(2 / a_j)`, so a stretch risks at most `6 a_j`, and these sum to at most `3δ/4`.
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

/-- Where `Λ ≥ log(2 / a_j)`, `Ns ≥ 36Λ / ε` and `y ≤ ε/48`, a stretch at the `j`-th hypothesis
risks at most `6 a_j` at the knobs `θr = ε/36`, `εd' = 4ε/9`, `θe' = 113ε/720` and `θpt' = 7/8`:
its tests `4 a_j`, its silences at most `a_j/2` or `a_j/4` each. -/
theorem stretch_le (C : Level3Settings) (hδ : 0 < C.δ) (hρ0 : 0 < C.disagreementRate)
    (hρ1 : C.disagreementRate < 1) (hy0 : 0 < C.harvestRate)
    (hy : C.harvestRate ≤ C.disagreementRate / 48) {Λ : ℝ} {Ns j : ℕ} (hΛ0 : 0 ≤ Λ)
    (hΛ : Real.log (2 / C.level j) ≤ Λ) (hNs : 36 * Λ / C.disagreementRate ≤ Ns) :
    stretchRisk C (C.level j) Ns Ns (C.disagreementRate / 36) (4 * C.disagreementRate / 9)
      (113 * C.disagreementRate / 720) (7 / 8) ≤ 6 * C.level j := by
  have ha : 0 < C.level j := level_pos C hδ j
  unfold stretchRisk Level3Settings.disagreementAlternative Level3Settings.middleAlternative
    Level3Settings.middleRate Level3Settings.searchShare Level3Settings.endsAlternative
    Level3Settings.endsRate
  set a := C.level j
  set ρ := C.disagreementRate
  set t := C.harvestRate
  set k := ⌈ρ / 6 * Ns⌉₊
  have hN0 : (0 : ℝ) ≤ Ns := Nat.cast_nonneg _
  have hρN : 36 * Λ ≤ ρ * Ns := by rw [div_le_iff₀ hρ0] at hNs; linarith
  have hE : Real.exp (-Λ) ≤ a / 2 := by
    calc Real.exp (-Λ) ≤ Real.exp (-Real.log (2 / a)) := Real.exp_le_exp.2 (by linarith)
      _ = a / 2 := by rw [Real.exp_neg, Real.exp_log (by positivity), inv_div]
  have hE2 : Real.exp (-(2 * Λ)) ≤ a ^ 2 / 4 := by
    rw [show -(2 * Λ) = -Λ + -Λ by ring, Real.exp_add]
    have := Real.exp_pos (-Λ)
    nlinarith
  -- no change of hypothesis
  have h2 : (1 - ρ / 36) ^ Ns ≤ a / 2 := by
    refine (pow_le_exp (c := -(ρ / 36)) (by linarith)
      (by linarith [Real.add_one_le_exp (-(ρ / 36))]) Ns).trans
      (le_trans (Real.exp_le_exp.2 ?_) hE)
    nlinarith
  -- success silent
  have hBD : 4 * ρ / 9 * (ρ / (ρ / 2)) + (1 - 4 * ρ / 9) * ((1 - ρ) / (1 - ρ / 2))
      ≤ 1 - ρ / 18 := by
    have e : ρ / (ρ / 2) = 2 := by field_simp
    rw [e]
    have hd : 0 < 1 - ρ / 2 := by linarith
    have : (1 - 4 * ρ / 9) * ((1 - ρ) / (1 - ρ / 2)) ≤ 1 - 17 * ρ / 18 := by
      rw [← mul_div_assoc, div_le_iff₀ hd]; nlinarith
    linarith
  have hBD0 : 0 ≤ 4 * ρ / 9 * (ρ / (ρ / 2)) + (1 - 4 * ρ / 9) * ((1 - ρ) / (1 - ρ / 2)) := by
    have : 0 ≤ (1 - ρ) / (1 - ρ / 2) := div_nonneg (by linarith) (by linarith)
    have : 0 ≤ 1 - 4 * ρ / 9 := by linarith
    positivity
  have h3 : (4 * ρ / 9 * (ρ / (ρ / 2)) + (1 - 4 * ρ / 9) * ((1 - ρ) / (1 - ρ / 2))) ^ Ns / a
      ≤ a / 4 := by
    rw [div_le_iff₀ ha]
    refine (pow_le_exp (c := -(ρ / 18)) hBD0
      (hBD.trans (by linarith [Real.add_one_le_exp (-(ρ / 18))])) Ns).trans
      (le_trans (Real.exp_le_exp.2 ?_) (hE2.trans (by nlinarith)))
    nlinarith
  -- ends silent
  have ht2 : 0 < 1 - 2 * t := by linarith
  have htv : (1 - t) / (1 - 2 * t) ≤ 1 + ρ / 46 := by
    rw [div_le_iff₀ ht2]; nlinarith
  have hBE : 113 * ρ / 720 * (t / (2 * t)) + (1 - 113 * ρ / 720) * ((1 - t) / (1 - 2 * t))
      ≤ 1 - ρ / 18 := by
    have e : t / (2 * t) = 1 / 2 := by field_simp
    rw [e]
    have : (1 - 113 * ρ / 720) * ((1 - t) / (1 - 2 * t)) ≤ (1 - 113 * ρ / 720) * (1 + ρ / 46) :=
      mul_le_mul_of_nonneg_left htv (by linarith)
    nlinarith
  have hBE0 :
      0 ≤ 113 * ρ / 720 * (t / (2 * t)) + (1 - 113 * ρ / 720) * ((1 - t) / (1 - 2 * t)) := by
    have : 0 ≤ (1 - t) / (1 - 2 * t) := div_nonneg (by linarith) ht2.le
    have : 0 ≤ t / (2 * t) := by positivity
    have : 0 ≤ 1 - 113 * ρ / 720 := by linarith
    positivity
  have h3e : (113 * ρ / 720 * (t / (2 * t)) + (1 - 113 * ρ / 720) * ((1 - t) / (1 - 2 * t))) ^ Ns
      / a ≤ a / 4 := by
    rw [div_le_iff₀ ha]
    refine (pow_le_exp (c := -(ρ / 18)) hBE0
      (hBE.trans (by linarith [Real.add_one_le_exp (-(ρ / 18))])) Ns).trans
      (le_trans (Real.exp_le_exp.2 ?_) (hE2.trans (by nlinarith)))
    nlinarith
  -- too few searches
  have hk1 : ((k - 1 : ℕ) : ℝ) ≤ ρ / 6 * Ns := by
    rcases Nat.eq_zero_or_pos k with h | h
    · rw [h]; simp only [Nat.zero_sub, Nat.cast_zero]; positivity
    · rw [Nat.cast_sub h, Nat.cast_one]
      linarith [Nat.ceil_lt_add_one (show 0 ≤ ρ / 6 * Ns by positivity)]
  have hlog2 : Real.log 2 ≤ 6932 / 10000 := by linarith [Real.log_two_lt_d9]
  have hlog20 : 0 ≤ Real.log 2 := Real.log_nonneg (by norm_num)
  have h4 : (2 : ℝ) ^ (k - 1) * (1 - (4 * ρ / 9 - 113 * ρ / 720) / 2) ^ Ns ≤ a / 2 := by
    have e2 : (2 : ℝ) ^ (k - 1) = Real.exp ((k - 1 : ℕ) * Real.log 2) := by
      rw [Real.exp_nat_mul, Real.exp_log two_pos]
    rw [e2, show 4 * ρ / 9 - 113 * ρ / 720 = 23 * ρ / 80 by ring]
    have hp := pow_le_exp (c := -(23 * ρ / 80 / 2)) (show 0 ≤ 1 - 23 * ρ / 80 / 2 by linarith)
      (by linarith [Real.add_one_le_exp (-(23 * ρ / 80 / 2))]) Ns
    have hkl : ((k - 1 : ℕ) : ℝ) * Real.log 2 ≤ ρ / 6 * Ns * (6932 / 10000) :=
      mul_le_mul hk1 hlog2 hlog20 (by positivity)
    calc _ ≤ Real.exp ((k - 1 : ℕ) * Real.log 2) * Real.exp (Ns * -(23 * ρ / 80 / 2)) :=
          mul_le_mul_of_nonneg_left hp (Real.exp_pos _).le
      _ = Real.exp ((k - 1 : ℕ) * Real.log 2 + Ns * -(23 * ρ / 80 / 2)) := by rw [Real.exp_add]
      _ ≤ Real.exp (-Λ) := Real.exp_le_exp.2 (by nlinarith)
      _ ≤ a / 2 := hE
  -- middles silent
  have hBM : (7 / 8 : ℝ) * (1 / 4 / (2 * (1 / 4)))
      + (1 - 7 / 8) * ((1 - 1 / 4) / (1 - 2 * (1 / 4))) = 5 / 8 := by norm_num
  have h5 : ((7 / 8 : ℝ) * (1 / 4 / (2 * (1 / 4)))
      + (1 - 7 / 8) * ((1 - 1 / 4) / (1 - 2 * (1 / 4)))) ^ k / a ≤ a / 4 := by
    rw [hBM, div_le_iff₀ ha]
    have hk : ρ / 6 * Ns ≤ k := Nat.le_ceil _
    refine (pow_le_exp (c := -(3 / 8)) (by norm_num)
      (by linarith [Real.add_one_le_exp (-(3 / 8 : ℝ))]) k).trans
      (le_trans (Real.exp_le_exp.2 ?_) (hE2.trans (by nlinarith)))
    nlinarith
  linarith

/-- Under level 3's premises at `δ < 1`, the knobs `Level3RoundCorrect` picks are admissible and
their stretches risk at most `3δ/4` in all. -/
theorem knobs_exist {α : Type*} [Fintype α] {σ : Type*} [Fintype σ] (C : Level3Settings) (S : ℕ)
    (hS : S = Fintype.card σ * ((Fintype.card σ + 2) * Fintype.card α + 1) + 2 * Fintype.card α + 1)
    (hδ : 0 < C.δ) (hδ1 : C.δ < 1) (hy0 : 0 < C.harvestRate)
    (hyρ : C.harvestRate ≤ C.disagreementRate / 48) (hρ1 : C.disagreementRate < 1)
    (hP : 36 * S / C.disagreementRate * Real.log (16 * S * (S + 1) / C.δ) + S ≤ C.probeBudget) :
    ∃ kn : Knobs, kn.Admissible (σ := σ) (α := α) C C.disagreementRate C.harvestRate
      ∧ kn.θgpt = 1 / 16 ∧ kn.ηm = 0
      ∧ ∑ j ∈ Finset.range S, stretchRisk C (C.level j) kn.Ns kn.N₁ kn.θr kn.εd' kn.θe' kn.θpt'
        ≤ 3 * C.δ / 4 := by
  set δ := C.δ
  set ρ := C.disagreementRate
  set y := C.harvestRate
  have hρ0 : 0 < ρ := by linarith
  have hS1 : (1 : ℝ) ≤ S := by
    have : 1 ≤ S := by omega
    exact_mod_cast this
  set Λs := Real.log (16 * S * (S + 1) / δ) with hΛs
  have hΛs0 : 0 < Λs := Real.log_pos (by rw [one_lt_div hδ]; nlinarith)
  set Ns := ⌈36 * Λs / ρ⌉₊ with hNs
  have hNs0 : 1 ≤ Ns := by
    rw [hNs, Nat.one_le_ceil_iff]
    positivity
  refine ⟨⟨Ns, Ns, 0, 1 / 16, 0, ρ / 48,
    ρ / 36, 4 * ρ / 9, 113 * ρ / 720, 7 / 8⟩, ?_, rfl, rfl, ?_⟩
  · refine ⟨hNs0, le_rfl, le_rfl, hδ, hy0, show y < 1 / 2 by linarith,
      show (0 : ℝ) < 1 / 4 by norm_num,
      show (1 / 4 : ℝ) < 1 / 2 by norm_num, hρ0, hρ1, le_rfl,
      show (0 : ℝ) ≤ ρ / 36 by positivity, show ρ / 36 ≤ 1 by linarith,
      show (0 : ℝ) ≤ 4 * ρ / 9 by positivity, show 4 * ρ / 9 ≤ 1 by linarith,
      show (0 : ℝ) ≤ 113 * ρ / 720 by positivity, show 113 * ρ / 720 ≤ 4 * ρ / 9 by linarith,
      show 2 * (1 / 4 : ℝ) ≤ 7 / 8 by norm_num, show (7 / 8 : ℝ) ≤ 1 by norm_num,
      show ρ / 36 ≤ (1 - 7 / 8) * (4 * ρ / 9 - 113 * ρ / 720) by linarith, ?_,
      show max (2 * 0) y ≤ y from max_le (by linarith) le_rfl,
      show 4 * (1 / 16 : ℝ) ≤ 1 / 4 by norm_num,
      show (0 : ℝ) ≤ ρ / 48 by positivity, hyρ,
      show 4 * (0 : ℝ) ≤ ρ / 48 by linarith, ?_⟩
    · change _ * Ns ≤ C.probeBudget
      rw [← hS]
      have hlt : (Ns : ℝ) < 36 * Λs / ρ + 1 :=
        Nat.ceil_lt_add_one (by positivity)
      have : ((S * Ns : ℕ) : ℝ) ≤ C.probeBudget := by
        push_cast
        calc (S : ℝ) * Ns ≤ S * (36 * Λs / ρ + 1) := by gcongr
          _ = 36 * S / ρ * Λs + S := by ring
          _ ≤ _ := hP
      exact_mod_cast this
    · exact middles_fall_of (μ := C.middleRate) (s := C.searchShare) (by positivity)
        (show (0 : ℝ) < 1 / 4 by norm_num) (show (1 / 4 : ℝ) ≤ 1 / 4 from le_rfl)
        (show 0 ≤ ρ / 6 by positivity)
        (show 2 * (ρ / 48) ≤ ρ / 6 * (1 / 4) by linarith)
  · change ∑ j ∈ Finset.range S,
        stretchRisk C (C.level j) Ns Ns (ρ / 36) (4 * ρ / 9) (113 * ρ / 720) (7 / 8) ≤ 3 * δ / 4
    calc _ ≤ ∑ j ∈ Finset.range S, 6 * C.level j := by
          refine Finset.sum_le_sum fun j hj => ?_
          refine stretch_le C hδ hρ0 hρ1 hy0 hyρ hΛs0.le ?_ (Nat.le_ceil _)
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

/-- Against `p₀ log(8(|Q| + 1)K/δ)`, `K` the pairs' count, the tails at every state's rate and
at both sites risk `δ/4`. -/
theorem tailRisk_eq (nQ nα : ℕ) {p₀ δ : ℝ} (hp₀ : 0 < p₀) (hδ : 0 < δ) (c : ℝ) (hc : 0 < c) :
    2 * (nQ + 1) * tailRisk nQ nα p₀ (p₀ * Real.log (c * (nQ + 1)
      * ((1 + (nQ + 1) ^ 3 * nα) ^ nQ * (nQ + 3) ^ ((nQ + 2) * nα) : ℕ) / δ))
      = 2 * δ / c := by
  set K : ℝ := (((1 + (nQ + 1) ^ 3 * nα) ^ nQ * (nQ + 3) ^ ((nQ + 2) * nα) : ℕ) : ℝ)
  have hK : 0 < K := by
    have : 0 < (1 + (nQ + 1) ^ 3 * nα) ^ nQ * (nQ + 3) ^ ((nQ + 2) * nα) := by positivity
    exact Nat.cast_pos.2 this
  change 2 * (nQ + 1) * (K * Real.exp (-(p₀ * Real.log (c * (nQ + 1) * K / δ) / p₀)))
    = 2 * δ / c
  rw [mul_div_cancel_left₀ _ hp₀.ne', Real.exp_neg, Real.exp_log (by positivity)]
  field_simp

theorem level3RoundCorrect_holds : Level3RoundCorrect := by
  intro α _ _ σ _ M side U Ω _ μ _ R P C p₀
  dsimp only
  intro hpre hδ hy0 hyρ hρ1 hP
  set nQ := Fintype.card σ
  set nα := Fintype.card α
  set δ := C.δ
  set y := C.harvestRate
  set K : ℕ := (1 + (nQ + 1) ^ 3 * nα) ^ nQ * (nQ + 3) ^ ((nQ + 2) * nα) with hK
  set η := p₀ * Real.log (8 * (nQ + 1) * (K : ℝ) / δ) with hη
  by_cases hδ1 : 1 ≤ δ
  · calc _ ≤ ∫⁻ _, 1 ∂μ := lintegral_mono fun _ => prob_le_one
      _ = 1 := by simp
      _ ≤ _ := by rw [← ENNReal.ofReal_one]; exact ENNReal.ofReal_le_ofReal hδ1
  push Not at hδ1
  have hp₀ : 0 < p₀ :=
    (prefixMax_pos P.D C.walkStart).trans_le (ciSup_le hpre)
  have hK1 : (1 : ℝ) ≤ K := by
    have : 1 ≤ K := Nat.one_le_iff_ne_zero.2 (by positivity)
    exact_mod_cast this
  have hη0 : 0 ≤ η := mul_nonneg hp₀.le (Real.log_nonneg (by
    rw [le_div_iff₀ hδ]; nlinarith [(Nat.cast_nonneg nQ : (0 : ℝ) ≤ nQ)]))
  obtain ⟨kn, hA, hθg, hηm, hsum⟩ := knobs_exist (σ := σ) (α := α) C _ rfl hδ hδ1 hy0 hyρ hρ1
    (by exact_mod_cast hP)
  have hy4 : y ≤ 1 / 4 := by linarith
  have hnQ : (0 : ℝ) ≤ nQ := Nat.cast_nonneg _
  refine (round_le_at M side U μ R P C _ _ p₀ η _ _ kn hA hp₀ hpre hy0 hy4 hη0 ?_ ?_ le_rfl
    hθg.ge (by rw [hηm])).trans (ENNReal.ofReal_le_ofReal ?_)
  · rw [div_le_iff₀ hy0]
    have : 0 ≤ (nQ + 1) * (16 * ((P.L : ℝ) + 1)) * y := by positivity
    have e : (nQ + 1) * (6 / y + 16 * ((P.L : ℝ) + 1)) * y
        = 6 * (nQ + 1) + (nQ + 1) * (16 * ((P.L : ℝ) + 1)) * y := by
      field_simp
    rw [e]; linarith
  · have : 0 ≤ (nQ + 1) * (6 / y) := by positivity
    nlinarith
  · unfold roundBoundAt
    have h1 := tailRisk_eq nQ nα hp₀ hδ 8 (by norm_num)
    rw [show (8 : ℝ) * (nQ + 1) * K / δ = 8 * ((nQ : ℝ) + 1) * K / δ from rfl] at h1
    have : 2 * ((nQ : ℝ) + 1) * tailRisk nQ nα p₀ η = 2 * δ / 8 := by
      rw [hη]; exact_mod_cast h1
    linarith

end Random

end OrthoDFA
