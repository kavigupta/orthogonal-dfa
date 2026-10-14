import OrthoDFA.Proofs.Level5Risk
import OrthoDFA.Proofs.Adaptive
import OrthoDFA.Proofs.BinomLaw
import OrthoDFA.Proofs.SeqRate
import Mathlib.Analysis.Complex.ExponentialBounds

/-!
# A closed-form bound on level 5's risk

The knobs are explicit functions of the primitives, each of the four noise tails and the fake
race is at most `δ / 8`, and every test and termination term is at most `C.testLevel`.
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

theorem binomSfGe_add_reflect (n j : ℕ) (p : ℝ) (hj : j ≤ n) :
    binomSfGe n p (j + 1) + binomSfGe n (1 - p) (n - j) = 1 := by
  refine Eq.trans ?_ (binomSfGe_zero_right n p)
  rw [binomSfGe_eq_range, binomSfGe_eq_range, binomSfGe_eq_range,
    ← Finset.sum_range_reflect _ (n + 1)]
  conv_rhs => rw [← Finset.sum_range_reflect _ (n + 1)]
  rw [← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun i hi => ?_
  simp only [Finset.mem_range] at hi
  have h1 : n + 1 - 1 - i = n - i := by omega
  have h2 : n - (n - i) = i := by omega
  rw [h1, h2, Nat.choose_symm (by omega), sub_sub_cancel]
  by_cases hij : j + 1 ≤ n - i
  · have : ¬ n - j ≤ i := by omega
    simp [hij, this]
  · have : n - j ≤ i := by omega
    simp only [hij, this, if_true, if_false, zero_add, zero_le]
    ring

/-- Hoeffding's bound on the binomial lower tail. -/
theorem one_sub_binomSfGe_le (n j : ℕ) (p τ : ℝ) (hp0 : 0 ≤ p) (hp1 : p ≤ 1) (hτ : 0 ≤ τ)
    (h : (j : ℝ) + n * τ ≤ n * p) :
    1 - binomSfGe n p (j + 1) ≤ Real.exp (-2 * (n : ℝ) * τ ^ 2) := by
  have hjn : j ≤ n := by
    have : (j : ℝ) ≤ n := by nlinarith [(Nat.cast_nonneg n : (0 : ℝ) ≤ n)]
    exact_mod_cast this
  have := binomSfGe_add_reflect n j p hjn
  rw [show 1 - binomSfGe n p (j + 1) = binomSfGe n (1 - p) (n - j) by linarith]
  refine binomSfGe_le n (n - j) (1 - p) τ (by linarith) (by linarith) hτ ?_
  rw [Nat.cast_sub hjn]
  linarith

variable {α : Type*} [Fintype α] [DecidableEq α]

omit [DecidableEq α] in
theorem count_le_unionLog [Nonempty α] (σ : Type*) [Fintype σ] (S : ℕ) {δ : ℝ} (hδ0 : 0 < δ)
    (hδ1 : δ ≤ 1) :
    8 * (classCount (α := α) σ S * (edgeMapCount (α := α) σ S
      * ((Fintype.card σ + S + 2) * Fintype.card α))) / δ
      ≤ Real.exp (unionLog (α := α) σ S δ) := by
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
  have hexp : Real.exp (unionLog (α := α) σ S δ) = X ^ (4 * (N + 2) * A) := by
    rw [← Real.exp_log (show 0 < X by linarith), ← Real.exp_nat_mul]
    unfold unionLog
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

theorem exp_neg_le_of_le {a x : ℝ} (ha0 : 0 < a) (h : Real.log (1 / a) ≤ x) :
    Real.exp (-x) ≤ a := by
  have : Real.log a = -Real.log (1 / a) := by rw [one_div, Real.log_inv, neg_neg]
  rw [← Real.exp_log ha0, this]
  exact Real.exp_le_exp.2 (by linarith)

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

/-- The termination tests' knobs: `nEnd` probes a hypothesis, the success count `hS` and the
middles count `hP`. -/
theorem exists_end_knobs (n₀ : ℕ) (εd a : ℝ) (hεd0 : 0 < εd) (hεd1 : εd < 1) (ha0 : 0 < a)
    (ha1 : a < 1) :
    ∃ nEnd hS hP : ℕ,
      (nEnd : ℝ) ≤ 6 * Real.log (1 / a) / εd ^ 2 + 112 * Real.log (1 / a) / εd + 2 * n₀ / εd + 1
      ∧ hS ≤ nEnd ∧ n₀ ≤ hS + 1 ∧ seqRate (1 / 4) (3 / 8) a (hS + 1) hP
      ∧ seqRate εd (7 / 8 * εd) a nEnd hS
      ∧ 3 / 4 * εd * nEnd ≤ hS + 1 ∧ termLevel nEnd hP hS (1 / 2) (εd / 2) ≤ 2 * a := by
  set lg := Real.log (1 / a) with hlg
  have hlg0 : 0 < lg := Real.log_pos (by rw [lt_div_iff₀ ha0]; linarith)
  set x := 6 * lg / εd ^ 2 + 112 * lg / εd + 2 * n₀ / εd with hx
  have hx1 : 0 ≤ 6 * lg / εd ^ 2 := by positivity
  have hx2 : 0 ≤ 112 * lg / εd := by positivity
  have hx3 : 0 ≤ 2 * (n₀ : ℝ) / εd := by positivity
  have hx0 : 0 ≤ x := by positivity
  set nEnd := ⌈x⌉₊ with hnEnd
  have hN1 : x ≤ nEnd := Nat.le_ceil x
  have hN2 : (nEnd : ℝ) ≤ x + 1 := (Nat.ceil_lt_add_one hx0).le
  have hA : 6 * lg ≤ nEnd * εd ^ 2 := by
    have : 6 * lg / εd ^ 2 ≤ nEnd := by linarith
    rwa [div_le_iff₀ (by positivity)] at this
  have hB : 112 * lg ≤ nEnd * εd := by
    have : 112 * lg / εd ≤ nEnd := by linarith
    rwa [div_le_iff₀ hεd0] at this
  have hD : 2 * (n₀ : ℝ) ≤ nEnd * εd := by
    have : 2 * (n₀ : ℝ) / εd ≤ nEnd := by linarith
    rwa [div_le_iff₀ hεd0] at this
  have hnE0 : (0 : ℝ) ≤ nEnd := Nat.cast_nonneg _
  set hS := ⌊13 / 16 * εd * (nEnd : ℝ)⌋₊ with hhS
  have hS0 : (0 : ℝ) ≤ hS := Nat.cast_nonneg _
  have hS1 : (hS : ℝ) ≤ 13 / 16 * εd * nEnd := Nat.floor_le (by positivity)
  have hS2 : 13 / 16 * εd * (nEnd : ℝ) < hS + 1 := Nat.lt_floor_add_one _
  have hεn : 0 ≤ εd * nEnd := mul_nonneg hεd0.le hnE0
  have hSn' : (hS : ℝ) ≤ nEnd := by linarith [mul_nonneg (sub_nonneg.2 hεd1.le) hnE0]
  have hSn : hS ≤ nEnd := by exact_mod_cast hSn'
  set t := hS + 1 with ht
  have ht' : 13 / 16 * εd * (nEnd : ℝ) < t := by rw [ht]; push_cast; exact hS2
  have ht40 : 40 * lg ≤ t := by linarith
  have ht0 : (0 : ℝ) ≤ t := Nat.cast_nonneg _
  set hP := ⌈3 / 8 * (t : ℝ)⌉₊ with hhP
  have hP1 : 3 / 8 * (t : ℝ) ≤ hP := Nat.le_ceil _
  have hP2 : (hP : ℝ) < 3 / 8 * t + 1 := Nat.ceil_lt_add_one (by positivity)
  have hPt : hP ≤ t := Nat.ceil_le.2 (by linarith)
  have hPt' : (hP : ℝ) ≤ t := by exact_mod_cast hPt
  have hpt : seqRate (1 / 4) (3 / 8) a t hP := by
    rw [seqRate_iff_log (by norm_num) (by norm_num) (by norm_num) (by norm_num) ha0 hPt,
      show (3 / 8 : ℝ) / (1 / 4) = 3 / 2 by norm_num,
      show (1 - 3 / 8 : ℝ) / (1 - 1 / 4) = (6 / 5)⁻¹ by norm_num, Real.log_inv]
    have h65 : Real.log (6 / 5) ≤ 1 / 5 := by
      have := Real.log_le_sub_one_of_pos (show (0 : ℝ) < 6 / 5 by norm_num); linarith
    have e1 := mul_le_mul_of_nonneg_left log_three_halves_ge (show (0 : ℝ) ≤ hP by positivity)
    have e2 := mul_le_mul_of_nonneg_left h65 (show (0 : ℝ) ≤ t - hP by linarith)
    rw [← hlg]
    linarith
  have hsucc : seqRate εd (7 / 8 * εd) a nEnd hS := by
    rw [seqRate_iff_log hεd0 hεd1 (by positivity) (by linarith) ha0 hSn,
      show 7 / 8 * εd / εd = (8 / 7)⁻¹ by rw [mul_div_assoc, div_self hεd0.ne']; norm_num, Real.log_inv]
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
  refine ⟨nEnd, hS, hP, by linarith, hSn, ?_, hpt, hsucc, by linarith, ?_⟩
  · have : (n₀ : ℝ) ≤ hS + 1 := by linarith [Nat.cast_nonneg (α := ℝ) n₀]
    exact_mod_cast this
  · unfold termLevel
    have h1 : 1 - binomSfGe t (1 / 2) hP ≤ a := by
      cases hPc : hP with
      | zero => rw [binomSfGe_zero_right]; linarith
      | succ j =>
        have hj : (j : ℝ) + 1 < 3 / 8 * t + 1 := by
          have := hP2; rw [hPc] at this; push_cast at this; exact this
        refine (one_sub_binomSfGe_le t j (1 / 2) (1 / 8) (by norm_num) (by norm_num)
          (by norm_num) (by linarith)).trans ?_
        rw [show -2 * (t : ℝ) * (1 / 8) ^ 2 = -(t / 32) by ring]
        exact exp_neg_le_of_le ha0 (by linarith)
    have h2 : binomSfGe nEnd (εd / 2) t ≤ a := by
      refine (binomSfGe_le nEnd t (εd / 2) (5 / 16 * εd) (by positivity) (by linarith)
        (by positivity) (by linarith)).trans ?_
      rw [show -2 * (nEnd : ℝ) * (5 / 16 * εd) ^ 2 = -(25 / 128 * (nEnd * εd ^ 2)) by ring]
      exact exp_neg_le_of_le ha0 (by linarith)
    linarith

/-- The probes at hypotheses that record, enough to make `B` records at rate `εd / 4`. -/
theorem exists_rec_knob (B : ℕ) (εd a : ℝ) (hεd0 : 0 < εd) (hεd1 : εd ≤ 1) (ha0 : 0 < a)
    (ha1 : a ≤ 1) :
    ∃ nRec : ℕ, (nRec : ℝ) ≤ 8 * B / εd + 32 * Real.log (1 / a) / εd ^ 2 + 1
      ∧ 1 - binomSfGe nRec (εd / 4) B ≤ a := by
  set lg := Real.log (1 / a) with hlg
  have hlg0 : 0 ≤ lg := Real.log_nonneg (by rw [le_div_iff₀ ha0]; linarith)
  set x := 8 * B / εd + 32 * lg / εd ^ 2 with hx
  have hx0 : 0 ≤ x := by positivity
  refine ⟨⌈x⌉₊, by linarith [Nat.ceil_lt_add_one hx0], ?_⟩
  have hn : x ≤ ⌈x⌉₊ := Nat.le_ceil x
  set nRec := ⌈x⌉₊
  have h1 : 8 * (B : ℝ) ≤ nRec * εd := by
    have : 8 * (B : ℝ) / εd ≤ nRec := by
      have : 0 ≤ 32 * lg / εd ^ 2 := by positivity
      linarith
    rwa [div_le_iff₀ hεd0] at this
  have h2 : 32 * lg ≤ nRec * εd ^ 2 := by
    have : 32 * lg / εd ^ 2 ≤ nRec := by
      have : 0 ≤ 8 * (B : ℝ) / εd := by positivity
      linarith
    rwa [div_le_iff₀ (by positivity)] at this
  cases B with
  | zero => rw [binomSfGe_zero_right]; linarith
  | succ j =>
    push_cast at h1
    refine (one_sub_binomSfGe_le nRec j (εd / 4) (εd / 8) (by positivity) (by linarith)
      (by positivity) (by linarith)).trans ?_
    rw [show -2 * (nRec : ℝ) * (εd / 8) ^ 2 = -(nRec * εd ^ 2 / 32) by ring]
    exact exp_neg_le_of_le ha0 (by linarith)

theorem race_side_of (κs L : ℝ) (hκs : 0 ≤ κs) (hL : 0 ≤ L + 1) (h : 4 * κs * (L + 1) ≤ 1) :
    (Real.exp 1 - 1) * κs * (L + 1) ≤ 1 - Real.exp (-(4 * κs * (L + 1))) := by
  set x := 4 * κs * (L + 1) with hx
  have hx0 : 0 ≤ x := by positivity
  have he : Real.exp 1 - 1 ≤ 2 := by linarith [Real.exp_one_lt_three]
  have h1 : Real.exp (-x) ≤ 1 / (1 + x) := by
    rw [Real.exp_neg, ← one_div]
    exact one_div_le_one_div_of_le (by positivity) (by linarith [Real.add_one_le_exp x])
  have h2 : 1 / (1 + x) ≤ 1 - x / 2 := by
    rw [div_le_iff₀ (by positivity)]; nlinarith
  have h3 : (Real.exp 1 - 1) * κs * (L + 1) ≤ x / 2 := by
    rw [hx]; nlinarith [mul_nonneg hκs hL]
  linarith

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

set_option maxHeartbeats 1000000 in -- every field of `Knobs.Admissible` and all seven terms
theorem level5_risk_bound_holds : Level5RiskBound := by
  intro α _ _ _ σ _ κ ε θ D _ L C κs δ S
  dsimp only
  intro hδ0 hδ1 hL hLmax hκ hκs hκsL hε0 hεL hθ0 hθe hθpt hθs hθs1 hφe hφpt hεd1 hθs_eq hθs₁_eq
    hθpt_eq hθpt₁_eq hεd₁_eq hφ_eq hn₀ hexc_gt hexc_le ha0 ha hbudget hm1 hm2 hm3 hm4
  generalize hp₀ : prefixMax D C.walkStart = p₀ at hθs hφe hφpt hm4
  generalize hΛ : unionLog (α := α) σ S δ = Λ at hθs hφe hφpt hm4
  generalize hlg : Real.log (1 / C.testLevel) = lg at hn₀ hexc_gt hexc_le hbudget hm2
  have hp0 : 0 < p₀ := hp₀ ▸ prefixMax_pos D C.walkStart
  have hK := count_le_unionLog (α := α) σ S hδ0 hδ1
  rw [hΛ] at hK
  have hA1 : (1 : ℝ) ≤ Fintype.card α := by exact_mod_cast Fintype.card_pos
  have hN0 : (0 : ℝ) ≤ Fintype.card σ + S := by positivity
  have hΛ0 : 0 < Λ := by
    rw [← hΛ]
    unfold unionLog
    refine mul_pos (mul_pos (by positivity) (by linarith)) (Real.log_pos ?_)
    rw [lt_div_iff₀ hδ0]
    have h3 : (3 : ℝ) ≤ Fintype.card σ + S + 3 := by linarith
    nlinarith only [h3, hA1, hδ1]
  have hpΛ : 0 < p₀ * Λ := mul_pos hp0 hΛ0
  have hL1 : (1 : ℝ) ≤ L := by exact_mod_cast hL
  have hb0 : (0 : ℝ) ≤ C.probeBudget := Nat.cast_nonneg _
  have hpow : (2 : ℝ) ≤ 2 ^ (C.maxLeaves + 2) := by
    calc (2 : ℝ) = 2 ^ 1 := by norm_num
      _ ≤ 2 ^ (C.maxLeaves + 2) := pow_le_pow_right₀ (by norm_num) (by omega)
  have hpowA : (2 : ℝ) ≤ 2 ^ (C.maxLeaves + 2) * Fintype.card α := by nlinarith only [hpow, hA1]
  have ha8 : 8 * C.testLevel ≤ δ := by
    nlinarith only [mul_nonneg hb0 ha0.le, mul_le_mul_of_nonneg_right hpowA ha0.le, ha]
  have ha1 : C.testLevel ≤ 1 := by linarith
  have ha_lt : C.testLevel < 1 := by linarith
  have hlg0 : 0 < lg := by rw [← hlg]; exact Real.log_pos (by rw [lt_div_iff₀ ha0]; linarith)
  have hθX : 0 ≤ ((L + 1) * ((Fintype.card σ : ℝ) + S + 1)) * θ :=
    mul_nonneg (by positivity) hθ0
  have hθpt0 : 0 ≤ C.middleUndecidedRate := by linarith
  have hθφ : 0 < C.middleUndecidedRate * C.searchShare := by linarith
  have hθpt_pos : 0 < C.middleUndecidedRate := by
    rcases hθpt0.lt_or_eq with h | h
    · exact h
    · rw [← h, zero_mul] at hθφ; exact absurd hθφ (lt_irrefl 0)
  have hφpt_pos : 0 < C.searchShare := pos_of_mul_pos_right hθφ hθpt0
  have hεd0 : 0 < C.disagreementRate := by rw [hφ_eq] at hφpt_pos; linarith
  have hSq : C.SeqOK :=
    { a_pos := ha0
      a_lt := ha_lt
      θs_pos := by rw [hθs_eq]; norm_num
      θs_alt := by rw [hθs_eq, hθs₁_eq]; norm_num
      θs_alt_lt := by rw [hθs₁_eq]; norm_num
      θpt_pos := hθpt_pos
      θpt_alt := by rw [hθpt_eq, hθpt₁_eq]; norm_num
      θpt_alt_lt := by rw [hθpt₁_eq]; norm_num
      εd_pos := hεd0
      εd_lt := hεd1
      εd_alt_pos := by rw [hεd₁_eq]; positivity
      εd_alt := by rw [hεd₁_eq]; linarith
      φ_pos := hφpt_pos
      φ_lt := by rw [hφ_eq]; linarith }
  have hθe0 : 0 ≤ C.edgeUndecidedRate := by linarith
  have hφeθe : 0 < C.edgeTraffic * C.edgeUndecidedRate := by
    have : 0 < (L + 1) * (p₀ * Λ) := mul_pos (by positivity) hpΛ
    linarith
  have hφe0 : 0 ≤ C.edgeTraffic := by
    by_contra h
    push Not at h
    nlinarith only [h, hθe0, hφeθe]
  have hθs0 : 0 ≤ C.startUndecidedRate := by
    have : 0 ≤ ((Fintype.card σ : ℝ) + S + 1) * θ := mul_nonneg (by positivity) hθ0
    linarith
  have hθN : ((Fintype.card σ : ℝ) + S + 1) * θ ≤ ((L + 1) * ((Fintype.card σ : ℝ) + S + 1))
      * θ := by
    have : 0 ≤ ((Fintype.card σ : ℝ) + S + 1) * θ := mul_nonneg (by positivity) hθ0
    nlinarith only [this, hL1]
  have hκs0 : 0 ≤ κs := by linarith
  obtain ⟨ρ, hρ⟩ : ∃ ρ : ℝ, ρ = 2 * (ε * ((L + 1) * ((Fintype.card σ : ℝ) + S + 1)))
      + 2 * p₀ * (1 + 4 * (κ * (L + 1) + ε * ((L + 1) * ((Fintype.card σ : ℝ) + S + 1))))
        * Λ := ⟨_, rfl⟩
  obtain ⟨θgs, hθgs⟩ : ∃ x : ℝ, x = 3 * θ * ((Fintype.card σ : ℝ) + S + 1) + p₀ * Λ :=
    ⟨_, rfl⟩
  obtain ⟨θgpt, hθgpt⟩ : ∃ x : ℝ,
      x = 3 * ((L + 1) * ((Fintype.card σ : ℝ) + S + 1)) * θ
          + 8 * p₀ * Λ / C.searchShare := ⟨_, rfl⟩
  obtain ⟨Xe, hXe⟩ : ∃ x : ℝ, x = 8 * (L + 1) * (1 + 12 * θ * C.maxLeaves) * lg := ⟨_, rfl⟩
  have hκL : 0 ≤ κ * (L + 1) := mul_nonneg hκ.le (by positivity)
  have hεN : 0 ≤ ε * ((L + 1) * ((Fintype.card σ : ℝ) + S + 1)) :=
    mul_nonneg hε0 (by positivity)
  have hρ1 : 0 ≤ 2 * p₀ * (1 + 4 * (κ * (L + 1) + ε * ((L + 1)
      * ((Fintype.card σ : ℝ) + S + 1)))) * Λ := by
    have : 0 ≤ 1 + 4 * (κ * (L + 1) + ε * ((L + 1) * ((Fintype.card σ : ℝ) + S + 1))) := by
      linarith
    exact mul_nonneg (mul_nonneg (by linarith) this) hΛ0.le
  have hgφ : θgpt * C.searchShare = 3 * ((L + 1) * ((Fintype.card σ : ℝ) + S + 1)) * θ
      * C.searchShare
      + 8 * (p₀ * Λ) := by
    rw [hθgpt]; field_simp
  have hgpt0 : 0 ≤ θgpt := by
    rw [hθgpt]; have : 0 ≤ 8 * p₀ * Λ / C.searchShare := by positivity
    linarith
  have h16 : 16 * θgpt ≤ C.middleUndecidedRate := by
    have : 8 * p₀ * Λ / C.searchShare * 16 ≤ C.middleUndecidedRate / 2 := by
      rw [div_mul_eq_mul_div, div_le_iff₀ hφpt_pos]; linarith
    rw [hθgpt]; linarith
  have hgs0 : 0 ≤ θgs := by
    rw [hθgs]; have : 0 ≤ θ * ((Fintype.card σ : ℝ) + S + 1) := mul_nonneg hθ0 (by positivity)
    linarith
  have h8gs : 16 * θgs ≤ C.startUndecidedRate := by rw [hθgs]; linarith
  have hlgle : Real.log (1 / C.testLevel) ≤ lg := hlg.le
  obtain ⟨nEnd, hS, hP, hnE, hSn, hn₀S, hpt, hsucc, hφE, hterm⟩ :=
    exists_end_knobs C.firstLook C.disagreementRate C.testLevel hεd0 hεd1 ha0 ha_lt
  rw [hlg] at hnE
  obtain ⟨nRec, hnR, hrec⟩ := exists_rec_knob
    (Fintype.card σ * Fintype.card α * C.recordsToMove
        + (S + 1) * C.recordsToMove) C.disagreementRate C.testLevel hεd0 hεd1.le ha0 ha1
  rw [hlg] at hnR
  have hlog8 : 0 < Real.log (8 / δ) := Real.log_pos (by rw [lt_div_iff₀ hδ0]; linarith)
  have hm0 : 0 < C.recordsToMove := by
    rcases Nat.eq_zero_or_pos C.recordsToMove with h | h
    · rw [h] at hm1; push_cast at hm1; linarith
    · exact h
  have hW : (((S + Fintype.card σ + 1) * subT C (Fintype.card α) nEnd nRec : ℕ) : ℝ)
      ≤ C.probeBudget := by
    unfold subT subVersions
    push_cast
    have h1 : ((2 * C.maxLeaves * Fintype.card α + 1 : ℝ)) * nEnd
        ≤ (2 * C.maxLeaves * Fintype.card α + 1)
        * (6 * lg / C.disagreementRate ^ 2 + 112 * lg / C.disagreementRate
            + 2 * C.firstLook / C.disagreementRate + 1) :=
      mul_le_mul_of_nonneg_left hnE (by positivity)
    have h2 : (nRec : ℝ) ≤ 8 * (Fintype.card σ * Fintype.card α + S + 1)
        * C.recordsToMove / C.disagreementRate
        + 32 * lg / C.disagreementRate ^ 2 + 1 := by
      refine hnR.trans (le_of_eq ?_)
      push_cast
      ring
    refine le_trans ?_ hbudget
    exact mul_le_mul_of_nonneg_left (by linarith) (by positivity)
  have hbig : ((S : ℝ) + Fintype.card σ + 1) * (2 * C.maxLeaves * Fintype.card α + 1)
      ≤ C.probeBudget := by
    refine le_trans ?_ hbudget
    have hx : 0 ≤ 6 * lg / C.disagreementRate ^ 2 + 112 * lg / C.disagreementRate
        + 2 * C.firstLook / C.disagreementRate := by positivity
    have hy : 0 ≤ 8 * ((Fintype.card σ : ℝ) * Fintype.card α + S + 1)
        * C.recordsToMove / C.disagreementRate
      + 32 * lg / C.disagreementRate ^ 2 + 1 := by positivity
    have hP : (0 : ℝ) ≤ (S + Fintype.card σ + 1) := by positivity
    have hQ : (0 : ℝ) ≤ 2 * C.maxLeaves * Fintype.card α + 1 := by positivity
    nlinarith only [mul_nonneg (mul_nonneg hP hQ) hx, mul_nonneg hP hy]
  have hadm : Knobs.Admissible σ κ ε θ D L C κs ⟨S, nEnd, nRec, hP, hS, ρ, 3 * θ, θgs, θgpt,
      1 / 2, C.disagreementRate / 4, C.disagreementRate / 2, Xe, 2
          * C.startUndecidedRate, C.firstLook, 1, 4 * κs, 1 / p₀⟩ := by
    refine
      { κ_pos := hκ
        ε_nonneg := hε0
        θ_nonneg := hθ0
        records_slack := by dsimp only; rw [hρ]; linarith
        l_nonneg := by dsimp only; positivity
        θg_ge := by dsimp only; linarith
        θgpt_ge := by
          dsimp only; rw [hθgpt]
          have : 0 ≤ 8 * p₀ * Λ / C.searchShare := by positivity
          linarith
        θgpt_le := by dsimp only; linarith
        θe_ge := by dsimp only; linarith
        m_pos := hm0
        Lmax_ge := hLmax
        L_pos := hL
        seq := hSq
        θpt'_nonneg := by dsimp only; linarith
        θpt'_le := by dsimp only; linarith
        θr_nonneg := by dsimp only; linarith
        θr_le_one := by dsimp only; linarith
        εd'_nonneg := by dsimp only; linarith
        εd'_le := by dsimp only; linarith
        θr_le := by dsimp only; linarith
        θgs_nonneg := hgs0
        θgs_le := by dsimp only; linarith
        φe_nonneg := hφe0
        n₀_le_hS := hn₀S
        hS_le := hSn
        pt_level := by rw [hθpt_eq, hθpt₁_eq]; exact hpt
        success_level := by rw [hεd₁_eq]; exact hsucc
        φpt_nEnd := by rw [hφ_eq]; exact hφE
        exc_gt := fun j => by dsimp only; rw [hlg]; linarith [hexc_gt j]
        start_test := fun t h hht hs => by
          rw [hθs_eq, hθs₁_eq, seqRate_iff_log (by norm_num) (by norm_num) (by norm_num)
            (by norm_num) ha0 hht, show (3 / 5 : ℝ) / (1 / 2) = 6 / 5 by norm_num,
            show (1 - 3 / 5 : ℝ) / (1 - 1 / 2) = (5 / 4)⁻¹ by norm_num, Real.log_inv, hlg] at hs
          rw [hθs_eq, hlg]
          have e0 : Real.log (6 / 5) + Real.log (5 / 4) = Real.log (3 / 2) := by
            rw [← Real.log_mul (by norm_num) (by norm_num)]; norm_num
          have e1 : Real.log (3 / 2) ≤ 1 / 2 := by
            have := Real.log_le_sub_one_of_pos (show (0 : ℝ) < 3 / 2 by norm_num); linarith
          have e2 : 1 / 16 ≤ Real.log (5 / 4) := by
            have := log_ge_one_sub_inv (show (0 : ℝ) < 5 / 4 by norm_num); linarith
          have e3 := mul_le_mul_of_nonneg_left (e0.le.trans e1)
            (show (0 : ℝ) ≤ h from Nat.cast_nonneg _)
          have e4 := mul_le_mul_of_nonneg_left e2 (show (0 : ℝ) ≤ t from Nat.cast_nonneg _)
          linarith
        gate_test := by rw [hlg]; exact hn₀
        Xe_nonneg := by
          dsimp only; rw [hXe]
          have : 0 ≤ 1 + 12 * θ * C.maxLeaves := by
            have : 0 ≤ θ * C.maxLeaves := mul_nonneg hθ0 (Nat.cast_nonneg _)
            linarith
          positivity
        exc_le := fun j => by dsimp only; rw [hXe]; exact hexc_le j
        θs'_nonneg := by dsimp only; linarith
        Xs_nonneg := Nat.cast_nonneg _
        start_lin := fun n h hh => by
          dsimp only at hh
          rw [hθs_eq] at hh
          rw [hθs_eq, hθs₁_eq]
          have hf : (1 : ℝ) ≤ C.firstLook := by
            have : (0 : ℝ) < C.firstLook := by linarith
            have : 0 < C.firstLook := by exact_mod_cast this
            exact_mod_cast this
          have hnh : n < h := by
            have : (n : ℝ) < h := by linarith
            exact_mod_cast this
          refine seqRate_of_many (by norm_num) (by norm_num) ha0 hnh ?_
          rw [show (3 / 5 : ℝ) / (1 / 2) = 6 / 5 by norm_num, hlg]
          have e1 : 1 / 6 ≤ Real.log (6 / 5) := by
            have := log_ge_one_sub_inv (show (0 : ℝ) < 6 / 5 by norm_num); linarith
          have e2 := mul_le_mul_of_nonneg_left e1 (show (0 : ℝ) ≤ h from Nat.cast_nonneg _)
          have : (0 : ℝ) ≤ n := Nat.cast_nonneg _
          linarith
        η_nonneg := by dsimp only; norm_num
        ν_nonneg := by dsimp only; linarith
        race_side := race_side_of κs L hκs0 (by positivity) hκsL
        budget_ge := by exact_mod_cast hW }
  unfold roundRisk
  refine (iInf₂_le _ hadm).trans ?_
  unfold roundBound recordsTail edgeTail startTail middlesTail testLevels
  dsimp only
  rw [hp₀]
  have hc1 : 1 ≤ classCount (α := α) σ S := by
    unfold classCount; exact one_le_pow₀ (le_add_of_nonneg_right (by positivity))
  have he1 : 1 ≤ edgeMapCount (α := α) σ S := by
    unfold edgeMapCount; exact one_le_pow₀ (by linarith)
  have hm1' : (1 : ℝ) ≤ (Fintype.card σ + S + 2) * Fintype.card α := by nlinarith only [hA1, hN0]
  have hK0 : classCount (α := α) σ S * edgeMapCount (α := α) σ S ≤ classCount (α := α) σ S
      * (edgeMapCount (α := α) σ S * ((Fintype.card σ + S + 2) * Fintype.card α)) := by
    rw [← mul_assoc]; exact le_mul_of_one_le_right (by positivity) hm1'
  have hKc : classCount (α := α) σ S ≤ classCount (α := α) σ S
      * (edgeMapCount (α := α) σ S * ((Fintype.card σ + S + 2) * Fintype.card α)) :=
    le_mul_of_one_le_right (by positivity) (by nlinarith only [he1, hm1'])
  have hθ6 : 6 * θ * ((Fintype.card σ : ℝ) + S + 1) ≤ 1 := by linarith
  have hT1 : classCount (α := α) σ S * edgeMapCount (α := α) σ S
      * Real.exp (-((ρ - 2 * (ε * ((L + 1) * (Fintype.card σ + S + 1))))
        / (2 * p₀ * (1 + 4 * (κ * (L + 1) + ε * ((L + 1) * (Fintype.card σ + S + 1)))))))
      ≤ δ / 8 := by
    refine mul_exp_le hδ0 (by positivity) hK0 hK ?_
    rw [le_div_iff₀ (by positivity), hρ]
    linarith
  have hT2 : classCount (α := α) σ S
      * (edgeMapCount (α := α) σ S * ((Fintype.card σ + S + 2) * Fintype.card α))
      * Real.exp (-(C.edgeTraffic * (C.edgeUndecidedRate - 8 * (3 * θ)) / 4 / (2 * p₀ * (L + 1)
        * (1 + 4 * (3 / 2 * θ) * (Fintype.card σ + S + 1))))) ≤ δ / 8 := by
    refine mul_exp_le hδ0 (by positivity) le_rfl hK ?_
    rw [le_div_iff₀ (by positivity)]
    have e1 : 1 + 4 * (3 / 2 * θ) * ((Fintype.card σ : ℝ) + S + 1) ≤ 2 := by linarith
    have e2 := mul_le_mul_of_nonneg_left e1
      (show 0 ≤ 2 * p₀ * (L + 1) * Λ from mul_nonneg (by positivity) hΛ0.le)
    have e3 : C.edgeTraffic * (C.edgeUndecidedRate / 4) / 2
        ≤ C.edgeTraffic * (C.edgeUndecidedRate - 8 * (3 * θ)) / 4 := by
      have := mul_le_mul_of_nonneg_left (show C.edgeUndecidedRate / 2
        ≤ C.edgeUndecidedRate - 8 * (3 * θ) by linarith) hφe0
      linarith
    nlinarith only [e2, e3, hφe]
  have hT3 : classCount (α := α) σ S
      * Real.exp ((Fintype.card σ + S + 1) * (3 / 2 * θ)
        * (Real.exp (1 / p₀ * p₀) - 1) / p₀ - 1 / p₀ * θgs) ≤ δ / 8 := by
    rw [one_div_mul_cancel hp0.ne']
    have := mul_exp_le (K := classCount (α := α) σ S) (x := -((Fintype.card σ + S + 1)
      * (3 / 2 * θ) * (Real.exp 1 - 1) / p₀ - 1 / p₀ * θgs)) hδ0 (by positivity) hKc hK ?_
    · rwa [neg_neg] at this
    have he : Real.exp 1 - 1 ≤ 2 := by linarith [Real.exp_one_lt_three]
    have h1 : ((Fintype.card σ : ℝ) + S + 1) * (3 / 2 * θ) * (Real.exp 1 - 1) / p₀
        - 1 / p₀ * θgs = (((Fintype.card σ : ℝ) + S + 1) * (3 / 2 * θ) * (Real.exp 1 - 1)
          - θgs) / p₀ := by field_simp
    rw [h1, le_neg, div_le_iff₀ hp0, hθgs]
    have := mul_le_mul_of_nonneg_left he
      (show 0 ≤ ((Fintype.card σ : ℝ) + S + 1) * (3 / 2 * θ) from
        mul_nonneg (by positivity) (by linarith))
    nlinarith only [this]
  have hT4 : classCount (α := α) σ S * edgeMapCount (α := α) σ S
      * Real.exp (-(θgpt * C.searchShare / 2 / (2 * p₀
        * (1 + 4 * ((L + 1) * (Fintype.card σ + S + 1) * (3 / 2 * θ)))))) ≤ δ / 8 := by
    refine mul_exp_le hδ0 (by positivity) hK0 hK ?_
    rw [le_div_iff₀ (by positivity), hgφ]
    have e1 : 1 + 4 * ((L + 1) * ((Fintype.card σ : ℝ) + S + 1) * (3 / 2 * θ)) ≤ 2 := by
      linarith
    have e2 := mul_le_mul_of_nonneg_left e1 (show 0 ≤ 2 * (p₀ * Λ) by linarith)
    have e3 : 0 ≤ 3 * ((L + 1) * ((Fintype.card σ : ℝ) + S + 1)) * θ * C.searchShare :=
      mul_nonneg (by linarith) hφpt_pos.le
    nlinarith only [e2, e3]
  have hρle : ρ ≤ 2 * (ε * ((L + 1) * ((Fintype.card σ : ℝ) + S + 1))) + 6 * (p₀ * Λ) := by
    rw [hρ]
    have e1 : 1 + 4 * (κ * (L + 1) + ε * ((L + 1) * ((Fintype.card σ : ℝ) + S + 1))) ≤ 3 := by
      nlinarith only [hκs, hκsL, hεL, hL1]
    have := mul_le_mul_of_nonneg_left e1 (show 0 ≤ 2 * (p₀ * Λ) by linarith)
    nlinarith only [this]
  have hρ0 : 0 ≤ ρ := by rw [hρ]; linarith
  have hF : fakeRun C (Fintype.card α) S L ((S + Fintype.card σ + 1)
      * subT C (Fintype.card α) nEnd nRec) ρ Xe (2 * C.startUndecidedRate) C.firstLook 1 (4 * κs)
          ≤ δ / 8 := by
    unfold fakeRun
    set W : ℝ := (((S + Fintype.card σ + 1) * subT C (Fintype.card α) nEnd nRec : ℕ) : ℝ)
    have he : Real.exp 1 - 1 ≤ 2 := by linarith [Real.exp_one_lt_three]
    have hX0 : 0 ≤ C.edgeUndecidedRate * ((L + 1) * C.maxLeaves) + 2 * C.startUndecidedRate
        + 2 * C.maxLeaves * Fintype.card α * C.edgeTraffic := by
      have : 0 ≤ C.edgeUndecidedRate * ((L + 1) * C.maxLeaves) := mul_nonneg hθe0 (by positivity)
      have : 0 ≤ 2 * C.maxLeaves * Fintype.card α * C.edgeTraffic := mul_nonneg (by positivity) hφe0
      linarith
    set X := C.edgeUndecidedRate * ((L + 1) * C.maxLeaves) + 2 * C.startUndecidedRate
        + 2 * C.maxLeaves * Fintype.card α * C.edgeTraffic
    have hc0 : 0 ≤ 4 * κs * X + (Real.exp 1 - 1) * ρ :=
      add_nonneg (mul_nonneg (by linarith) hX0)
        (mul_nonneg (by linarith [Real.add_one_le_exp (1 : ℝ)]) hρ0)
    have hWc := mul_le_mul_of_nonneg_right hW hc0
    have h1 : (Real.exp 1 - 1) * ρ ≤ 2 * ρ := mul_le_mul_of_nonneg_right he hρ0
    have h2 := mul_le_mul_of_nonneg_left h1 hb0
    have h3 := mul_le_mul_of_nonneg_left hρle (show (0 : ℝ) ≤ 2 * C.probeBudget by positivity)
    have hfin : -1 * ((S + 1) * C.recordsToMove) + 4 * κs
        * (2 * C.maxLeaves * Fintype.card α * Xe + C.firstLook)
        + W * (4 * κs * X + (Real.exp 1 - 1) * ρ) ≤ -Real.log (8 / δ) := by
      rw [hXe]
      nlinarith only [hWc, h2, h3, hm1, hm2, hm3, hm4]
    calc Real.exp (-1 * ((S + 1) * C.recordsToMove)
        + 4 * κs * (2 * C.maxLeaves * Fintype.card α * Xe + C.firstLook)
          + W * (4 * κs * X + (Real.exp 1 - 1) * ρ))
        ≤ Real.exp (-Real.log (8 / δ)) := Real.exp_le_exp.2 hfin
      _ = δ / 8 := by rw [Real.exp_neg, Real.exp_log (by positivity)]; field_simp
  have hsub : ((S : ℝ) + Fintype.card σ + 1) * subOpen C (Fintype.card α) (Fintype.card σ) nRec
      ((S + 1) * C.recordsToMove) (C.disagreementRate / 4)
          (termLevel nEnd hP hS (1 / 2) (C.disagreementRate / 2))
      ≤ ((S : ℝ) + Fintype.card σ + 1) * ((2 * C.maxLeaves * Fintype.card α + 1)
          * (2 * C.testLevel) + C.testLevel) := by
    unfold subOpen subVersions
    push_cast
    gcongr
  have hTL : ((S : ℝ) + Fintype.card σ + 1) * ((2 * C.maxLeaves * Fintype.card α + 1)
      * (2 * C.testLevel) + C.testLevel)
      + (2 ^ (C.maxLeaves + 2) * Fintype.card α + 1
          + 3 * C.probeBudget) * C.testLevel
      ≤ δ / 4 := by
    have hP0 : (0 : ℝ) ≤ (S : ℝ) + Fintype.card σ + 1 := by positivity
    have hQ1 : (1 : ℝ) ≤ 2 * C.maxLeaves * Fintype.card α
        + 1 := le_add_of_nonneg_left (by positivity)
    have hb1 : (1 : ℝ) ≤ C.probeBudget := by
      have : (1 : ℝ) ≤ ((S : ℝ) + Fintype.card σ + 1) * (2 * C.maxLeaves * Fintype.card α + 1) :=
        one_le_mul_of_one_le_of_one_le (by linarith) hQ1
      linarith
    have hPa : 0 ≤ ((S : ℝ) + Fintype.card σ + 1) * C.testLevel := mul_nonneg hP0 ha0.le
    have : ((S : ℝ) + Fintype.card σ + 1) * ((2 * C.maxLeaves * Fintype.card α + 1)
        * (2 * C.testLevel) + C.testLevel)
        ≤ 3 * C.probeBudget * C.testLevel := by
      have h1 := mul_le_mul_of_nonneg_left hQ1 hPa
      have h2 := mul_le_mul_of_nonneg_right hbig (show 0 ≤ 3 * C.testLevel by linarith)
      nlinarith only [h1, h2]
    nlinarith only [this, ha, hb1, ha0, mul_nonneg (sub_nonneg.2 hb1) ha0.le]
  obtain ⟨Xs, hXs⟩ : ∃ x : ℝ, x = ((S : ℝ) + Fintype.card σ + 1)
      * ((2 * C.maxLeaves * Fintype.card α + 1) * (2 * C.testLevel) + C.testLevel) := ⟨_, rfl⟩
  obtain ⟨XT, hXT⟩ : ∃ x : ℝ,
      x = (2 ^ (C.maxLeaves + 2) * Fintype.card α + 1
          + 3 * C.probeBudget) * C.testLevel :=
    ⟨_, rfl⟩
  rw [← hXs] at hsub hTL
  rw [← hXT] at hTL
  have hXs0 : 0 ≤ Xs := by rw [hXs]; exact mul_nonneg (by positivity) (by positivity)
  have hXT0 : 0 ≤ XT := by rw [hXT]; exact mul_nonneg (by positivity) ha0.le
  have hd : 0 ≤ δ / 8 := by positivity
  calc _ ≤ ENNReal.ofReal (δ / 8) + ENNReal.ofReal (δ / 8) + ENNReal.ofReal (δ / 8)
        + ENNReal.ofReal (δ / 8) + ENNReal.ofReal (δ / 8) + ENNReal.ofReal Xs
        + ENNReal.ofReal XT := by
        rw [hXT]
        gcongr
    _ = ENNReal.ofReal (δ / 8 + δ / 8 + δ / 8 + δ / 8 + δ / 8 + Xs + XT) := by
        rw [ENNReal.ofReal_add (by linarith) hXT0, ENNReal.ofReal_add (by linarith) hXs0,
          ENNReal.ofReal_add (by linarith) hd, ENNReal.ofReal_add (by linarith) hd,
          ENNReal.ofReal_add (by linarith) hd, ENNReal.ofReal_add hd hd]
    _ ≤ ENNReal.ofReal δ := ENNReal.ofReal_le_ofReal (by linarith)

end OrthoDFA
