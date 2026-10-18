import OrthoDFA.Proofs.Level5Risk
import Mathlib.Analysis.Complex.ExponentialBounds

/-!
# A closed-form bound on level 5's risk

The knobs are explicit functions of the primitives: each of the three noise tails is at most
`δ / 8`, and success fires at `nEnd` probes with at most `hS` searches while the middles test
fires at `hS + 1` middles in a row.
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

omit [DecidableEq α] in
theorem count_le_exp [Nonempty α] (σ : Type*) [Fintype σ] (S : ℕ) {δ : ℝ} (hδ0 : 0 < δ)
    (hδ1 : δ ≤ 1) :
    8 * (classCount (α := α) σ S * (edgeMapCount (α := α) σ S
      * ((Fintype.card σ + S + 2) * Fintype.card α))) / δ
      ≤ Real.exp (4 * (Fintype.card σ + S + 2) * Fintype.card α
        * Real.log ((Fintype.card σ + S + 3) * Fintype.card α / δ)) := by
  set N := Fintype.card σ + S with hN
  set A := Fintype.card α with hA
  have hA1 : 1 ≤ A := Fintype.card_pos
  have hA1' : (1 : ℝ) ≤ A := by exact_mod_cast hA1
  have hNc : ((N : ℕ) : ℝ) = Fintype.card σ + S := by rw [hN]; push_cast; rfl
  set X : ℝ := (N + 3) * A / δ with hX
  have hX' : ((N : ℝ) + 3) * A ≤ X := by
    rw [hX, le_div_iff₀ hδ0]
    nlinarith [mul_le_mul_of_nonneg_left hδ1 (show (0 : ℝ) ≤ (N + 3) * A by positivity)]
  have hN0 : (0 : ℝ) ≤ N := Nat.cast_nonneg N
  have hX3 : (3 : ℝ) ≤ X := by nlinarith
  have hX1 : (1 : ℝ) ≤ X := by linarith
  have hexp : Real.exp (4 * (Fintype.card σ + S + 2) * Fintype.card α
      * Real.log ((Fintype.card σ + S + 3) * Fintype.card α / δ)) = X ^ (4 * (N + 2) * A) := by
    rw [← Real.exp_log (show 0 < X by linarith), ← Real.exp_nat_mul]
    push_cast
    rw [← hNc]
  rw [hexp]
  have hc : classCount (α := α) σ S ≤ X ^ (3 * N) := by
    unfold classCount
    rw [pow_mul]
    refine pow_le_pow_left₀ (by positivity) ?_ _
    rw [← hNc]
    have : ((N : ℝ) + 3) ^ 3 * A ≤ X ^ 3 := by
      calc ((N : ℝ) + 3) ^ 3 * A ≤ ((N + 3) * A) ^ 3 := by
            rw [mul_pow]; gcongr; exact le_self_pow₀ hA1' (by norm_num)
        _ ≤ X ^ 3 := by gcongr
    rw [← hA]
    have h3 : ((N : ℝ) + 1) ^ 3 + 1 ≤ (N + 3) ^ 3 := by nlinarith
    nlinarith [mul_le_mul_of_nonneg_right h3 (show (0 : ℝ) ≤ A by positivity)]
  have he : edgeMapCount (α := α) σ S ≤ X ^ ((N + 2) * A) := by
    unfold edgeMapCount
    rw [← hNc]
    exact pow_le_pow_left₀ (by positivity) (by nlinarith) _
  have hk : 8 * (((N : ℝ) + 2) * A) / δ ≤ X ^ 3 := by
    calc 8 * (((N : ℝ) + 2) * A) / δ ≤ X ^ 2 * X := by
          rw [mul_div_assoc]
          gcongr
          · nlinarith
          · rw [hX]; gcongr; linarith
      _ = X ^ 3 := by ring
  rw [← hNc]
  calc 8 * (classCount (α := α) σ S * (edgeMapCount (α := α) σ S * (((N : ℝ) + 2) * A))) / δ
      = classCount (α := α) σ S * edgeMapCount (α := α) σ S * (8 * ((N + 2) * A) / δ) := by ring
    _ ≤ X ^ (3 * N) * X ^ ((N + 2) * A) * X ^ 3 := by
        gcongr
        exact le_of_lt (by unfold edgeMapCount; positivity)
    _ = X ^ (3 * N + (N + 2) * A + 3) := by rw [pow_add, pow_add]
    _ ≤ X ^ (4 * (N + 2) * A) := pow_le_pow_right₀ hX1 (by nlinarith)

omit [DecidableEq α] in
theorem count_le_unionLog [Nonempty α] (σ : Type*) [Fintype σ] {δ : ℝ} (hδ0 : 0 < δ)
    (hδ1 : δ ≤ 1) :
    8 * (classCount (α := α) σ 0 * (edgeMapCount (α := α) σ 0
      * ((Fintype.card σ + 2) * Fintype.card α))) / δ ≤ Real.exp (unionLog (α := α) σ δ) := by
  have := count_le_exp (α := α) σ 0 hδ0 hδ1
  simp only [Nat.cast_zero, add_zero] at this
  exact this

theorem log_ge_one_sub_inv {x : ℝ} (hx : 0 < x) : 1 - 1 / x ≤ Real.log x := by
  have := Real.log_le_sub_one_of_pos (inv_pos.2 hx)
  rw [Real.log_inv] at this
  rw [one_div]
  linarith

theorem log_three_halves_ge : 2 / 5 ≤ Real.log (3 / 2) := by
  have h2 := Real.log_two_gt_d9
  have h98 := log_ge_one_sub_inv (show (0 : ℝ) < 9 / 8 by norm_num)
  have e1 : Real.log (9 / 4) = Real.log 2 + Real.log (9 / 8) := by
    rw [← Real.log_mul (by norm_num) (by norm_num)]; norm_num
  have e2 : Real.log (9 / 4) = 2 * Real.log (3 / 2) := by
    rw [show (9 : ℝ) / 4 = (3 / 2) ^ 2 by norm_num, Real.log_pow]; push_cast; ring
  norm_num at h98
  linarith

/-- The termination knobs: success at most `hS` searches in `nEnd` probes fires, so do `hS + 1`
middles in a row, and `hS + 1` is past `91 log (1/a)` and three quarters of `εd nEnd`. -/
theorem exists_end_knobs (εd a : ℝ) (hεd0 : 0 < εd) (hεd1 : εd < 1) (ha0 : 0 < a) (ha1 : a < 1) :
    ∃ nEnd hS : ℕ, (nEnd : ℝ) ≤ 112 * Real.log (1 / a) / εd + 1 ∧ hS ≤ nEnd
      ∧ 91 * Real.log (1 / a) < hS + 1 ∧ seqRate (1 / 2) (3 / 4) a (hS + 1) (hS + 1)
      ∧ seqRate εd (7 / 8 * εd) a nEnd hS ∧ 3 / 4 * εd * nEnd ≤ hS + 1 := by
  set lg := Real.log (1 / a) with hlg
  have hlg0 : 0 < lg := Real.log_pos (by rw [lt_div_iff₀ ha0]; linarith)
  set x := 112 * lg / εd with hx
  have hx0 : 0 ≤ x := by positivity
  set nEnd := ⌈x⌉₊ with hnEnd
  have hN1 : x ≤ nEnd := Nat.le_ceil x
  have hN2 : (nEnd : ℝ) ≤ x + 1 := (Nat.ceil_lt_add_one hx0).le
  have hB : 112 * lg ≤ nEnd * εd := by
    have : 112 * lg / εd ≤ nEnd := hN1
    rwa [div_le_iff₀ hεd0] at this
  have hnE0 : (0 : ℝ) ≤ nEnd := Nat.cast_nonneg _
  set hS := ⌊13 / 16 * εd * (nEnd : ℝ)⌋₊ with hhS
  have hS0 : (0 : ℝ) ≤ hS := Nat.cast_nonneg _
  have hS1 : (hS : ℝ) ≤ 13 / 16 * εd * nEnd := Nat.floor_le (by positivity)
  have hS2 : 13 / 16 * εd * (nEnd : ℝ) < hS + 1 := Nat.lt_floor_add_one _
  have hεn : 0 ≤ εd * nEnd := mul_nonneg hεd0.le hnE0
  have hSn' : (hS : ℝ) ≤ nEnd := by linarith [mul_nonneg (sub_nonneg.2 hεd1.le) hnE0]
  have hSn : hS ≤ nEnd := by exact_mod_cast hSn'
  have h91 : 91 * lg < hS + 1 := by linarith
  have hpt : seqRate (1 / 2) (3 / 4) a (hS + 1) (hS + 1) := by
    rw [seqRate_iff_log (by norm_num) (by norm_num) (by norm_num) (by norm_num) ha0 le_rfl,
      show (3 / 4 : ℝ) / (1 / 2) = 3 / 2 by norm_num]
    rw [← hlg]
    simp only [Nat.cast_add, Nat.cast_one, sub_self, zero_mul, add_zero]
    have := mul_le_mul_of_nonneg_left log_three_halves_ge (show (0 : ℝ) ≤ hS + 1 by positivity)
    linarith
  have hsucc : seqRate εd (7 / 8 * εd) a nEnd hS := by
    rw [seqRate_iff_log hεd0 hεd1 (by positivity) (by linarith) ha0 hSn,
      show 7 / 8 * εd / εd = (8 / 7)⁻¹ by rw [mul_div_assoc, div_self hεd0.ne']; norm_num,
      Real.log_inv]
    have hB2 : Real.log (8 / 7) ≤ 1 / 7 := by
      have := Real.log_le_sub_one_of_pos (show (0 : ℝ) < 8 / 7 by norm_num); linarith
    have hc : 0 < 1 - 7 / 8 * εd := by linarith
    set q := εd / 8 / (1 - 7 / 8 * εd) with hq
    have hq0 : 0 ≤ q := by positivity
    have hqe : q * (1 - 7 / 8 * εd) = εd / 8 := by rw [hq, div_mul_cancel₀ _ hc.ne']
    have hA2 : q ≤ Real.log ((1 - 7 / 8 * εd) / (1 - εd)) := by
      refine le_trans (le_of_eq ?_) (log_ge_one_sub_inv (div_pos hc (by linarith)))
      rw [one_div_div, hq, eq_sub_iff_add_eq, ← add_div, div_eq_one_iff_eq hc.ne']; ring
    have e1 := mul_le_mul_of_nonneg_left hB2 hS0
    have e2 := mul_le_mul_of_nonneg_left hA2 (show (0 : ℝ) ≤ nEnd - hS by linarith)
    have e3 : (nEnd : ℝ) * (1 - 7 / 8 * εd) ≤ nEnd - hS := by linarith
    have e4 := mul_le_mul_of_nonneg_right e3 hq0
    have e5 : (nEnd : ℝ) * (q * (1 - 7 / 8 * εd)) = nEnd * (εd / 8) := by rw [hqe]
    rw [← hlg]
    linarith
  exact ⟨nEnd, hS, by linarith, hSn, h91, hpt, hsucc, by linarith⟩

/-- Each tail united over at most `K₁` events is at most `δ / 8` once its exponent reaches `Λ`. -/
theorem mul_exp_le {K K₁ Λ x δ : ℝ} (hδ : 0 < δ) (hK0 : 0 ≤ K) (hK : K ≤ K₁)
    (hΛ : 8 * K₁ / δ ≤ Real.exp Λ) (hx : Λ ≤ x) : K * Real.exp (-x) ≤ δ / 8 := by
  have h1 : K * Real.exp (-x) ≤ K₁ * Real.exp (-Λ) :=
    mul_le_mul hK (Real.exp_le_exp.2 (by linarith)) (Real.exp_pos _).le (hK0.trans hK)
  have h2 : K₁ * Real.exp (-Λ) ≤ δ / 8 := by
    rw [Real.exp_neg, ← div_eq_mul_inv, div_le_iff₀ (Real.exp_pos _)]
    rw [div_le_iff₀ hδ] at hΛ
    linarith
  linarith

set_option maxHeartbeats 1000000 in -- every field of `Knobs.Admissible` and all five terms
theorem level5_risk_bound_holds : Level5RiskBound := by
  intro α _ _ _ σ _ κ ε θ D _ L C δ
  dsimp only
  intro hδ0 hδ1 hL hLmax hκ hε hθ0 hθe hθpt hθs hφe hφpt hεd1 hθs_alt hθs₁_lt hθpt_eq hθpt₁_eq
    hεd₁_eq hφ_eq hn₀ hn₀7 hexc ha0 ha hbudget hwrong
  generalize hp₀ : prefixMax D C.walkStart = p₀ at hθs hφe hφpt
  generalize hΛ : unionLog (α := α) σ δ = Λ at hθs hφe hφpt
  generalize hlg : Real.log (1 / C.testLevel) = lg at hn₀ hn₀7 hexc hbudget
  have hp0 : 0 < p₀ := hp₀ ▸ prefixMax_pos D C.walkStart
  have hK := count_le_unionLog (α := α) σ hδ0 hδ1
  rw [hΛ] at hK
  have hA1 : (1 : ℝ) ≤ Fintype.card α := by exact_mod_cast Fintype.card_pos
  have hN0 : (0 : ℝ) ≤ Fintype.card σ := Nat.cast_nonneg _
  have hΛ0 : 0 < Λ := by
    rw [← hΛ]
    unfold unionLog
    refine mul_pos (mul_pos (by positivity) (by linarith)) (Real.log_pos ?_)
    rw [lt_div_iff₀ hδ0]
    have h3 : (3 : ℝ) ≤ Fintype.card σ + 3 := by linarith
    nlinarith only [h3, hA1, hδ1]
  have hpΛ : 0 < p₀ * Λ := mul_pos hp0 hΛ0
  have hL1 : (1 : ℝ) ≤ L := by exact_mod_cast hL
  have hb0 : (0 : ℝ) ≤ C.probeBudget := Nat.cast_nonneg _
  have hV1 : (1 : ℝ) ≤ versionBound C (Fintype.card α) := by
    have : 1 ≤ versionBound C (Fintype.card α) := by
      unfold versionBound
      have : 1 ≤ C.maxLeaves := by omega
      exact Nat.one_le_iff_ne_zero.2 (Nat.mul_ne_zero (by omega) (by omega))
    exact_mod_cast this
  have hLA : (0 : ℝ) ≤ 2 * C.maxLeaves * Fintype.card α := by positivity
  have ha1 : C.testLevel < 1 := by
    have h4 : (4 : ℝ) ≤ versionBound C (Fintype.card α) * (2 * C.maxLeaves * Fintype.card α + 4) := by
      nlinarith
    nlinarith
  have hlg0 : 0 < lg := by rw [← hlg]; exact Real.log_pos (by rw [lt_div_iff₀ ha0]; linarith)
  have hθpt_pos : 0 < C.middleUndecidedRate := by rw [hθpt_eq]; norm_num
  have hφpt_pos : 0 < C.searchShare := by
    have : 0 < C.middleUndecidedRate * C.searchShare := by linarith
    exact pos_of_mul_pos_right this hθpt_pos.le
  have hεd0 : 0 < C.disagreementRate := by rw [hφ_eq] at hφpt_pos; linarith
  have hθe0 : 0 ≤ C.edgeUndecidedRate := by linarith
  have hφeθe : 0 < C.edgeTraffic * C.edgeUndecidedRate := by
    have : 0 < (L + 1) * (p₀ * Λ) := mul_pos (by positivity) hpΛ
    linarith
  have hφe0 : 0 ≤ C.edgeTraffic := by
    by_contra h
    push Not at h
    nlinarith only [h, hθe0, hφeθe]
  have hθs0 : 0 < C.startUndecidedRate := by
    have h1 : 0 ≤ 6 * ((Fintype.card σ : ℝ) + 1) * θ := by positivity
    have h2 : 0 < 2 * p₀ * Λ := by positivity
    linarith
  have hSq : C.SeqOK :=
    { a_pos := ha0
      a_lt := ha1
      θs_pos := hθs0
      θs_alt := hθs_alt
      θs_alt_lt := hθs₁_lt
      θpt_pos := hθpt_pos
      θpt_alt := by rw [hθpt_eq, hθpt₁_eq]; norm_num
      θpt_alt_lt := by rw [hθpt₁_eq]; norm_num
      εd_pos := hεd0
      εd_lt := hεd1
      εd_alt_pos := by rw [hεd₁_eq]; positivity
      εd_alt := by rw [hεd₁_eq]; linarith
      φ_pos := hφpt_pos
      φ_lt := by rw [hφ_eq]; linarith }
  obtain ⟨nEnd, hS, hnE, hSn, h91, hpt, hsucc, hφE⟩ :=
    exists_end_knobs C.disagreementRate C.testLevel hεd0 hεd1 ha0 ha1
  rw [hlg] at hnE h91
  obtain ⟨θgs, hθgs⟩ : ∃ x : ℝ, x = 3 * θ * ((Fintype.card σ : ℝ) + 1) + p₀ * Λ := ⟨_, rfl⟩
  obtain ⟨θgpt, hθgpt⟩ : ∃ x : ℝ,
      x = 3 * ((L + 1) * ((Fintype.card σ : ℝ) + 1)) * θ + 8 * p₀ * Λ / C.searchShare :=
    ⟨_, rfl⟩
  have hgφ : θgpt * C.searchShare = 3 * ((L + 1) * ((Fintype.card σ : ℝ) + 1)) * θ
      * C.searchShare + 8 * (p₀ * Λ) := by
    rw [hθgpt]; field_simp
  have hgs0 : 0 ≤ θgs := by rw [hθgs]; positivity
  have h4gpt : 4 * θgpt ≤ C.middleUndecidedRate := by
    have : 8 * p₀ * Λ / C.searchShare * 4 ≤ C.middleUndecidedRate / 2 := by
      rw [div_mul_eq_mul_div, div_le_iff₀ hφpt_pos]; linarith
    rw [hθgpt]; nlinarith
  have hadm : Knobs.Admissible σ κ ε θ D L C ⟨nEnd, hS, 3 * θ, θgs, θgpt, 1 / p₀⟩ :=
    { κ_nonneg := hκ
      ε_nonneg := hε
      θ_nonneg := hθ0
      l_nonneg := by dsimp only; positivity
      θg_ge := by dsimp only; linarith
      θgpt_ge := by
        dsimp only; rw [hθgpt]
        have : 0 ≤ 8 * p₀ * Λ / C.searchShare := by positivity
        linarith
      θgpt_le := h4gpt
      θe_ge := by dsimp only; linarith
      θgs_nonneg := hgs0
      θgs_le := by dsimp only; rw [hθgs]; linarith
      φe_nonneg := hφe0
      Lmax_ge := hLmax
      L_pos := hL
      seq := hSq
      n₀_le_hS := by
        have : (C.firstLook : ℝ) < hS + 1 := by linarith
        exact_mod_cast this.le
      hS_le := hSn
      pt_level := by rw [hθpt_eq, hθpt₁_eq]; exact_mod_cast hpt
      success_level := by rw [hεd₁_eq]; exact hsucc
      φpt_nEnd := by rw [hφ_eq]; exact hφE
      exc_gt := fun j => by rw [hlg]; exact hexc j
      gate_test := by rw [hlg]; exact hn₀
      budget_ge := by
        have h1 : ((versionBound C (Fintype.card α) * nEnd : ℕ) : ℝ) ≤ C.probeBudget := by
          push_cast
          exact (mul_le_mul_of_nonneg_left hnE (by linarith)).trans hbudget
        exact_mod_cast h1 }
  unfold roundRisk
  refine (iInf₂_le _ hadm).trans ?_
  unfold roundBound edgeTail startTail middlesTail wrongTerm testLevels
  dsimp only
  rw [hp₀]
  have hc1 : 1 ≤ classCount (α := α) σ 0 := by
    unfold classCount; exact one_le_pow₀ (le_add_of_nonneg_right (by positivity))
  have he1 : 1 ≤ edgeMapCount (α := α) σ 0 := by
    unfold edgeMapCount; exact one_le_pow₀ (by push_cast; linarith)
  have hm1' : (1 : ℝ) ≤ (Fintype.card σ + 2) * Fintype.card α := by nlinarith only [hA1, hN0]
  have hK0 : classCount (α := α) σ 0 * edgeMapCount (α := α) σ 0 ≤ classCount (α := α) σ 0
      * (edgeMapCount (α := α) σ 0 * ((Fintype.card σ + 2) * Fintype.card α)) := by
    rw [← mul_assoc]; exact le_mul_of_one_le_right (by positivity) hm1'
  have hKc : classCount (α := α) σ 0 ≤ classCount (α := α) σ 0
      * (edgeMapCount (α := α) σ 0 * ((Fintype.card σ + 2) * Fintype.card α)) :=
    le_mul_of_one_le_right (by positivity) (by nlinarith only [he1, hm1'])
  have hθ6 : 6 * θ * ((Fintype.card σ : ℝ) + 1) ≤ 1 := by
    rw [hθpt_eq] at hθpt
    have : 0 ≤ θ * ((Fintype.card σ : ℝ) + 1) := by positivity
    nlinarith
  have hT2 : classCount (α := α) σ 0
      * (edgeMapCount (α := α) σ 0 * ((Fintype.card σ + 2) * Fintype.card α))
      * Real.exp (-(C.edgeTraffic * (C.edgeUndecidedRate - 8 * (3 * θ)) / 4 / (2 * p₀ * (L + 1)
        * (1 + 4 * (3 / 2 * θ) * (Fintype.card σ + 1))))) ≤ δ / 8 := by
    refine mul_exp_le hδ0 (by positivity) le_rfl hK ?_
    rw [le_div_iff₀ (by positivity)]
    have e1 : 1 + 4 * (3 / 2 * θ) * ((Fintype.card σ : ℝ) + 1) ≤ 2 := by linarith
    have e2 := mul_le_mul_of_nonneg_left e1
      (show 0 ≤ 2 * p₀ * (L + 1) * Λ from mul_nonneg (by positivity) hΛ0.le)
    have e3 : C.edgeTraffic * (C.edgeUndecidedRate / 4) / 2
        ≤ C.edgeTraffic * (C.edgeUndecidedRate - 8 * (3 * θ)) / 4 := by
      have := mul_le_mul_of_nonneg_left (show C.edgeUndecidedRate / 2
        ≤ C.edgeUndecidedRate - 8 * (3 * θ) by linarith) hφe0
      linarith
    nlinarith only [e2, e3, hφe]
  have hT3 : classCount (α := α) σ 0
      * Real.exp ((Fintype.card σ + 1) * (3 / 2 * θ)
        * (Real.exp (1 / p₀ * p₀) - 1) / p₀ - 1 / p₀ * θgs) ≤ δ / 8 := by
    rw [one_div_mul_cancel hp0.ne']
    have := mul_exp_le (K := classCount (α := α) σ 0) (x := -((Fintype.card σ + 1)
      * (3 / 2 * θ) * (Real.exp 1 - 1) / p₀ - 1 / p₀ * θgs)) hδ0 (by positivity) hKc hK ?_
    · rwa [neg_neg] at this
    have he : Real.exp 1 - 1 ≤ 2 := by linarith [Real.exp_one_lt_three]
    have h1 : ((Fintype.card σ : ℝ) + 1) * (3 / 2 * θ) * (Real.exp 1 - 1) / p₀
        - 1 / p₀ * θgs = (((Fintype.card σ : ℝ) + 1) * (3 / 2 * θ) * (Real.exp 1 - 1)
          - θgs) / p₀ := by field_simp
    rw [h1, le_neg, div_le_iff₀ hp0, hθgs]
    have := mul_le_mul_of_nonneg_left he
      (show 0 ≤ ((Fintype.card σ : ℝ) + 1) * (3 / 2 * θ) from
        mul_nonneg (by positivity) (by linarith))
    nlinarith only [this]
  have hT4 : classCount (α := α) σ 0 * edgeMapCount (α := α) σ 0
      * Real.exp (-(θgpt * C.searchShare / 2 / (2 * p₀
        * (1 + 4 * ((L + 1) * (Fintype.card σ + 1) * (3 / 2 * θ)))))) ≤ δ / 8 := by
    refine mul_exp_le hδ0 (by positivity) hK0 hK ?_
    rw [le_div_iff₀ (by positivity), hgφ]
    have e1 : 1 + 4 * ((L + 1) * ((Fintype.card σ : ℝ) + 1) * (3 / 2 * θ)) ≤ 2 := by
      rw [hθpt_eq] at hθpt
      nlinarith
    have e2 := mul_le_mul_of_nonneg_left e1 (show 0 ≤ 2 * (p₀ * Λ) by linarith)
    have e3 : 0 ≤ 3 * ((L + 1) * ((Fintype.card σ : ℝ) + 1)) * θ * C.searchShare :=
      mul_nonneg (by positivity) hφpt_pos.le
    nlinarith only [e2, e3]
  have hW : (C.probeBudget : ℝ) * ((L + 1) * ((C.maxLeaves + 1) * Fintype.card α ^ C.maxLeaves))
      * (ε + κ) ≤ δ / 4 := by linarith
  have hTL : (versionBound C (Fintype.card α) : ℝ) * (2 * C.maxLeaves * Fintype.card α + 4)
      * C.testLevel ≤ δ / 4 := by linarith
  have hd : 0 ≤ δ / 8 := by positivity
  have hW0 : 0 ≤ (C.probeBudget : ℝ) * ((L + 1) * ((C.maxLeaves + 1)
      * Fintype.card α ^ C.maxLeaves)) * (ε + κ) := by positivity
  have hTL0 : 0 ≤ (versionBound C (Fintype.card α) : ℝ) * (2 * C.maxLeaves * Fintype.card α + 4)
      * C.testLevel := by positivity
  have hsum : ENNReal.ofReal (δ / 8) + ENNReal.ofReal (δ / 8) + ENNReal.ofReal (δ / 8)
      + ENNReal.ofReal (δ / 4) + ENNReal.ofReal (δ / 4) ≤ ENNReal.ofReal δ := by
    have h4 : 0 ≤ δ / 4 := by positivity
    rw [← ENNReal.ofReal_add hd hd, ← ENNReal.ofReal_add (by positivity) hd,
      ← ENNReal.ofReal_add (by positivity) h4, ← ENNReal.ofReal_add (by positivity) h4]
    exact ENNReal.ofReal_le_ofReal (by linarith)
  exact le_trans (add_le_add (add_le_add (add_le_add (add_le_add (ENNReal.ofReal_le_ofReal hT2)
    (ENNReal.ofReal_le_ofReal hT3)) (ENNReal.ofReal_le_ofReal hT4)) (ENNReal.ofReal_le_ofReal hW))
    (ENNReal.ofReal_le_ofReal hTL)) hsum

end OrthoDFA
