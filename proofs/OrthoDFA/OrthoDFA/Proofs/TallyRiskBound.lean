import OrthoDFA.Proofs.TallyRisk
import OrthoDFA.Proofs.Adaptive
import OrthoDFA.Proofs.BinomLaw
import Mathlib.Analysis.Complex.ExponentialBounds

/-!
# A closed-form bound on the tally round's risk

The knobs are explicit functions of the primitives, each of the four noise tails and the fake
race is at most `δ / 8`, and every test and termination term is at most `C.a`.
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

/-- A test at rate `p` that has not fired at `h` lets a rate `q ≤ p / 8` reach `(h + 1) / 2` at
most `a` of the time. -/
theorem binomSfGe_half_le (t h : ℕ) (p q a : ℝ) (hp1 : p ≤ 1) (hq0 : 0 ≤ q) (hpq : 8 * q ≤ p)
    (ha0 : 0 < a) (ha : a ≤ 1 / 2) (ht : 32 * Real.log (1 / a) ≤ t * p ^ 2)
    (hlt : binomSfGe t p h < a) : binomSfGe t q ((h + 1) / 2) ≤ a := by
  have hp0 : 0 ≤ p := by linarith
  have hlg : Real.log 2 ≤ Real.log (1 / a) :=
    Real.log_le_log (by norm_num) (by rw [le_div_iff₀ ha0]; linarith)
  have hl2 : 0 < Real.log 2 := Real.log_pos (by norm_num)
  have hexp : Real.exp (-Real.log (1 / a)) = a := by
    rw [Real.exp_neg, Real.exp_log (by positivity)]; simp
  have hh : (t : ℝ) * p / 2 < h := by
    by_contra hc
    push Not at hc
    cases h with
    | zero => rw [binomSfGe_zero_right] at hlt; linarith
    | succ j =>
      have hj := one_sub_binomSfGe_le t j p (p / 2) hp0 hp1 (by linarith) (by
        push_cast at hc; nlinarith)
      have : Real.exp (-2 * (t : ℝ) * (p / 2) ^ 2) ≤ 1 / 2 := by
        rw [← Real.exp_log (show (0 : ℝ) < 1 / 2 by norm_num), one_div, Real.log_inv]
        exact Real.exp_le_exp.2 (by nlinarith)
      linarith
  have hj : (h : ℝ) / 2 ≤ ((h + 1) / 2 : ℕ) := by
    have : h ≤ 2 * ((h + 1) / 2) := by omega
    have : (h : ℝ) ≤ 2 * ((h + 1) / 2 : ℕ) := by exact_mod_cast this
    linarith
  refine (binomSfGe_le t _ q (p / 8) hq0 (by linarith) (by linarith) (by nlinarith)).trans ?_
  rw [← hexp]
  exact Real.exp_le_exp.2 (by nlinarith)

variable {α : Type*} [Fintype α] [DecidableEq α]

omit [DecidableEq α] in
theorem count_le_tallyLog [Nonempty α] (σ : Type*) [Fintype σ] (S : ℕ) {δ : ℝ} (hδ0 : 0 < δ)
    (hδ1 : δ ≤ 1) :
    8 * (classCount (α := α) σ S * (edgeMapCount (α := α) σ S
      * ((Fintype.card σ + S + 2) * Fintype.card α))) / δ
      ≤ Real.exp (tallyLog (α := α) σ S δ) := by
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
  have hexp : Real.exp (tallyLog (α := α) σ S δ) = X ^ (4 * (N + 2) * A) := by
    rw [← Real.exp_log (show 0 < X by linarith), ← Real.exp_nat_mul]
    unfold tallyLog
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

theorem exp_neg_lt_of_lt {a x : ℝ} (ha0 : 0 < a) (h : Real.log (1 / a) < x) :
    Real.exp (-x) < a := by
  have : Real.log a = -Real.log (1 / a) := by rw [one_div, Real.log_inv, neg_neg]
  rw [← Real.exp_log ha0, this]
  exact Real.exp_lt_exp.2 (by linarith)

theorem exp_neg_le_of_le {a x : ℝ} (ha0 : 0 < a) (h : Real.log (1 / a) ≤ x) :
    Real.exp (-x) ≤ a := by
  have : Real.log a = -Real.log (1 / a) := by rw [one_div, Real.log_inv, neg_neg]
  rw [← Real.exp_log ha0, this]
  exact Real.exp_le_exp.2 (by linarith)

/-- The termination tests' knobs: `nEnd` probes a hypothesis, the success count `hS` and the
middles count `hP`. -/
theorem exists_end_knobs (n₀ : ℕ) (εd θpt φpt a : ℝ) (hεd0 : 0 < εd) (hεd1 : εd ≤ 1)
    (hθ0 : 0 < θpt) (hθ1 : θpt ≤ 1 / 4) (hφ : φpt ≤ 3 / 4 * εd) (ha0 : 0 < a)
    (ha : a ≤ 1 / 2) :
    ∃ nEnd hS hP : ℕ,
      (nEnd : ℝ) ≤ 8 * Real.log (1 / a) / (εd ^ 2 * θpt ^ 2) + 4 * n₀ / (3 * εd) + 2
      ∧ n₀ ≤ nEnd ∧ n₀ ≤ hS + 1 ∧ binomSfGe (hS + 1) θpt hP < a
      ∧ 1 - binomSfGe nEnd εd (hS + 1) < a ∧ a ≤ binomSfGe nEnd εd hS
      ∧ φpt * nEnd ≤ hS + 1 ∧ termLevel nEnd hP hS (2 * θpt) (εd / 2) ≤ 2 * a := by
  set lg := Real.log (1 / a) with hlg
  have hlg0 : 0 < lg := Real.log_pos (by rw [lt_div_iff₀ ha0]; linarith)
  set x := 8 * lg / (εd ^ 2 * θpt ^ 2) + 4 * n₀ / (3 * εd) with hx
  have hx0 : 0 ≤ x := by positivity
  set nEnd := ⌈x⌉₊ + 1 with hnEnd
  have hN1 : x + 1 ≤ nEnd := by rw [hnEnd]; push_cast; linarith [Nat.le_ceil x]
  have hN2 : (nEnd : ℝ) ≤ x + 2 := by
    rw [hnEnd]; push_cast; linarith [Nat.ceil_lt_add_one hx0]
  have hpos : 0 < εd ^ 2 * θpt ^ 2 := by positivity
  have hA : 8 * lg < nEnd * (εd ^ 2 * θpt ^ 2) := by
    have : 8 * lg / (εd ^ 2 * θpt ^ 2) < nEnd := by
      have : 0 ≤ 4 * (n₀ : ℝ) / (3 * εd) := by positivity
      linarith
    rwa [div_lt_iff₀ hpos] at this
  have hθsq : θpt ^ 2 ≤ 1 := by nlinarith
  have hnE0 : (0 : ℝ) ≤ nEnd := Nat.cast_nonneg _
  have hB : 8 * lg < nEnd * εd ^ 2 := by
    have := mul_le_mul_of_nonneg_left hθsq (show 0 ≤ (nEnd : ℝ) * εd ^ 2 by positivity)
    nlinarith
  have hC : 8 * lg < nEnd * εd * θpt ^ 2 := by
    have : (nEnd : ℝ) * (εd ^ 2 * θpt ^ 2) ≤ nEnd * εd * θpt ^ 2 := by
      have : εd ^ 2 ≤ εd := by nlinarith
      nlinarith [mul_le_mul_of_nonneg_left this (show 0 ≤ (nEnd : ℝ) * θpt ^ 2 by positivity)]
    linarith
  have hD : (n₀ : ℝ) ≤ 3 / 4 * nEnd * εd := by
    have : 4 * (n₀ : ℝ) / (3 * εd) ≤ nEnd := by
      have : 0 ≤ 8 * lg / (εd ^ 2 * θpt ^ 2) := by positivity
      linarith
    rw [div_le_iff₀ (by positivity)] at this
    linarith
  set hS := ⌊3 / 4 * (nEnd : ℝ) * εd⌋₊ with hhS
  have hS1 : (hS : ℝ) ≤ 3 / 4 * nEnd * εd := Nat.floor_le (by positivity)
  have hS2 : 3 / 4 * (nEnd : ℝ) * εd < hS + 1 := Nat.lt_floor_add_one _
  have hsucc : 1 - binomSfGe nEnd εd (hS + 1) < a := by
    refine (one_sub_binomSfGe_le nEnd hS εd (εd / 4) hεd0.le hεd1 (by positivity)
      (by linarith)).trans_lt ?_
    rw [show -2 * (nEnd : ℝ) * (εd / 4) ^ 2 = -(nEnd * εd ^ 2 / 8) by ring]
    exact exp_neg_lt_of_lt ha0 (by linarith)
  set t := hS + 1 with ht
  have ht' : 3 / 4 * (nEnd : ℝ) * εd < t := by rw [ht]; push_cast; exact hS2
  have htθ : lg < t * θpt ^ 2 / 2 := by nlinarith
  set hP := ⌈3 / 2 * (t : ℝ) * θpt⌉₊ with hhP
  have hP1 : 3 / 2 * (t : ℝ) * θpt ≤ hP := Nat.le_ceil _
  have hP2 : (hP : ℝ) < 3 / 2 * t * θpt + 1 := Nat.ceil_lt_add_one (by positivity)
  have hpt : binomSfGe t θpt hP < a := by
    refine (binomSfGe_le t hP θpt (θpt / 2) hθ0.le (by linarith) (by positivity)
      (by linarith)).trans_lt ?_
    rw [show -2 * (t : ℝ) * (θpt / 2) ^ 2 = -(t * θpt ^ 2 / 2) by ring]
    exact exp_neg_lt_of_lt ha0 htθ
  refine ⟨nEnd, hS, hP, by linarith, by exact_mod_cast (show (n₀ : ℝ) ≤ nEnd by nlinarith),
    by exact_mod_cast (show (n₀ : ℝ) ≤ hS + 1 by linarith), hpt, hsucc, ?_, by nlinarith, ?_⟩
  · have := binomSfGe_antitone hεd0.le hεd1 (n := nEnd) hS
    linarith
  · unfold termLevel
    have h1 : 1 - binomSfGe t (2 * θpt) hP ≤ a := by
      cases hPc : hP with
      | zero => rw [binomSfGe_zero_right]; linarith
      | succ j =>
        have hj : (j : ℝ) + 1 < 3 / 2 * t * θpt + 1 := by
          have := hP2; rw [hPc] at this; push_cast at this; exact this
        refine (one_sub_binomSfGe_le t j (2 * θpt) (θpt / 2) (by positivity) (by linarith)
          (by positivity) (by linarith)).trans ?_
        rw [show -2 * (t : ℝ) * (θpt / 2) ^ 2 = -(t * θpt ^ 2 / 2) by ring]
        exact (exp_neg_lt_of_lt ha0 htθ).le
    have h2 : binomSfGe nEnd (εd / 2) t ≤ a := by
      refine (binomSfGe_le nEnd t (εd / 2) (εd / 4) (by positivity) (by linarith)
        (by positivity) (by linarith)).trans ?_
      rw [show -2 * (nEnd : ℝ) * (εd / 4) ^ 2 = -(nEnd * εd ^ 2 / 8) by ring]
      exact (exp_neg_lt_of_lt ha0 (by linarith)).le
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
theorem tally_risk_bound_holds : TallyRiskBound := by
  intro α _ _ _ σ _ κ ε θ D _ L C κs δ S
  dsimp only
  intro hδ0 hδ1 hL hLmax hκ hκs hκsL hε0 hεL hθ0 hθe hθpt hθs hθs1 hφe hφpt hεd1 hθpt4 hφεd
    hn₀s hn₀pt hn₀φ hexc_gt hexc_le ha0 ha hbudget hm1 hm2 hm3 hm4
  generalize hp₀ : prefixMax D C.k = p₀ at hθs hφe hφpt hm4
  generalize hΛ : tallyLog (α := α) σ S δ = Λ at hθs hφe hφpt hm4
  generalize hlg : Real.log (1 / C.a) = lg at hn₀s hn₀pt hn₀φ hexc_gt hexc_le hbudget hm2
  have hp0 : 0 < p₀ := hp₀ ▸ prefixMax_pos D C.k
  have hK := count_le_tallyLog (α := α) σ S hδ0 hδ1
  rw [hΛ] at hK
  have hA1 : (1 : ℝ) ≤ Fintype.card α := by exact_mod_cast Fintype.card_pos
  have hN0 : (0 : ℝ) ≤ Fintype.card σ + S := by positivity
  have hΛ0 : 0 < Λ := by
    rw [← hΛ]
    unfold tallyLog
    refine mul_pos (mul_pos (by positivity) (by linarith)) (Real.log_pos ?_)
    rw [lt_div_iff₀ hδ0]
    have h3 : (3 : ℝ) ≤ Fintype.card σ + S + 3 := by linarith
    nlinarith only [h3, hA1, hδ1]
  have hpΛ : 0 < p₀ * Λ := mul_pos hp0 hΛ0
  have hL1 : (1 : ℝ) ≤ L := by exact_mod_cast hL
  have hb0 : (0 : ℝ) ≤ C.budget := Nat.cast_nonneg _
  have hpow : (2 : ℝ) ≤ 2 ^ (C.Lmax + 1) := by
    calc (2 : ℝ) = 2 ^ 1 := by norm_num
      _ ≤ 2 ^ (C.Lmax + 1) := pow_le_pow_right₀ (by norm_num) (by omega)
  have hpowA : (2 : ℝ) ≤ 2 ^ (C.Lmax + 1) * Fintype.card α := by nlinarith only [hpow, hA1]
  have ha8 : 8 * C.a ≤ δ := by nlinarith only [sq_nonneg (C.budget : ℝ), ha, hpowA, ha0]
  have ha2 : C.a ≤ 1 / 2 := by linarith
  have ha1 : C.a ≤ 1 := by linarith
  have hlg0 : 0 < lg := by rw [← hlg]; exact Real.log_pos (by rw [lt_div_iff₀ ha0]; linarith)
  have hθX : 0 ≤ ((L + 1) * ((Fintype.card σ : ℝ) + S + 1)) * θ :=
    mul_nonneg (by positivity) hθ0
  have hθpt0 : 0 ≤ C.θpt := by linarith
  have hθφ : 0 < C.θpt * C.φpt := by linarith
  have hθpt_pos : 0 < C.θpt := by
    rcases hθpt0.lt_or_eq with h | h
    · exact h
    · rw [← h, zero_mul] at hθφ; exact absurd hθφ (lt_irrefl 0)
  have hφpt_pos : 0 < C.φpt := pos_of_mul_pos_right hθφ hθpt0
  have hεd0 : 0 < C.εd := by linarith
  have hθe0 : 0 ≤ C.θe := by linarith
  have hφeθe : 0 < C.φe * C.θe := by
    have : 0 < (L + 1) * (p₀ * Λ) := mul_pos (by positivity) hpΛ
    linarith
  have hφe0 : 0 ≤ C.φe := by
    by_contra h
    push Not at h
    nlinarith only [h, hθe0, hφeθe]
  have hθs0 : 0 ≤ C.θs := by
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
      x = 3 * ((L + 1) * ((Fintype.card σ : ℝ) + S + 1)) * θ + 8 * p₀ * Λ / C.φpt := ⟨_, rfl⟩
  obtain ⟨Xe, hXe⟩ : ∃ x : ℝ, x = 8 * (L + 1) * (1 + 12 * θ * C.Lmax) * lg := ⟨_, rfl⟩
  have hκL : 0 ≤ κ * (L + 1) := mul_nonneg hκ.le (by positivity)
  have hεN : 0 ≤ ε * ((L + 1) * ((Fintype.card σ : ℝ) + S + 1)) :=
    mul_nonneg hε0 (by positivity)
  have hρ1 : 0 ≤ 2 * p₀ * (1 + 4 * (κ * (L + 1) + ε * ((L + 1)
      * ((Fintype.card σ : ℝ) + S + 1)))) * Λ := by
    have : 0 ≤ 1 + 4 * (κ * (L + 1) + ε * ((L + 1) * ((Fintype.card σ : ℝ) + S + 1))) := by
      linarith
    exact mul_nonneg (mul_nonneg (by linarith) this) hΛ0.le
  have hgφ : θgpt * C.φpt = 3 * ((L + 1) * ((Fintype.card σ : ℝ) + S + 1)) * θ * C.φpt
      + 8 * (p₀ * Λ) := by
    rw [hθgpt]; field_simp
  have hgpt0 : 0 ≤ θgpt := by
    rw [hθgpt]; have : 0 ≤ 8 * p₀ * Λ / C.φpt := by positivity
    linarith
  have h16 : 16 * θgpt ≤ C.θpt := by
    have : 8 * p₀ * Λ / C.φpt * 16 ≤ C.θpt / 2 := by
      rw [div_mul_eq_mul_div, div_le_iff₀ hφpt_pos]; linarith
    rw [hθgpt]; linarith
  have hgs0 : 0 ≤ θgs := by
    rw [hθgs]; have : 0 ≤ θ * ((Fintype.card σ : ℝ) + S + 1) := mul_nonneg hθ0 (by positivity)
    linarith
  have h8gs : 8 * θgs ≤ C.θs := by rw [hθgs]; linarith
  have hlgle : Real.log (1 / C.a) ≤ lg := hlg.le
  obtain ⟨nEnd, hS, hP, hnE, hn₀E, hn₀S, hpt, hsucc, hreach, hφE, hterm⟩ :=
    exists_end_knobs C.n₀ C.εd C.θpt C.φpt C.a hεd0 hεd1 hθpt_pos hθpt4 hφεd ha0 ha2
  rw [hlg] at hnE
  obtain ⟨nRec, hnR, hrec⟩ := exists_rec_knob
    (Fintype.card σ * Fintype.card α * C.m + (S + 1) * C.m) C.εd C.a hεd0 hεd1 ha0 ha1
  rw [hlg] at hnR
  have hlog8 : 0 < Real.log (8 / δ) := Real.log_pos (by rw [lt_div_iff₀ hδ0]; linarith)
  have hm0 : 0 < C.m := by
    rcases Nat.eq_zero_or_pos C.m with h | h
    · rw [h] at hm1; push_cast at hm1; linarith
    · exact h
  have hW : (((S + Fintype.card σ + 1) * subT C (Fintype.card α) nEnd nRec : ℕ) : ℝ)
      ≤ C.budget := by
    unfold subT subVersions
    push_cast
    have h1 : ((2 * C.Lmax * Fintype.card α + 1 : ℝ)) * nEnd ≤ (2 * C.Lmax * Fintype.card α + 1)
        * (8 * lg / (C.εd ^ 2 * C.θpt ^ 2) + 4 * C.n₀ / (3 * C.εd) + 2) :=
      mul_le_mul_of_nonneg_left hnE (by positivity)
    have h2 : (nRec : ℝ) ≤ 8 * (Fintype.card σ * Fintype.card α + S + 1) * C.m / C.εd
        + 32 * lg / C.εd ^ 2 + 1 := by
      refine hnR.trans (le_of_eq ?_)
      push_cast
      ring
    refine le_trans ?_ hbudget
    exact mul_le_mul_of_nonneg_left (by linarith) (by positivity)
  have hbig : 2 * (((S : ℝ) + Fintype.card σ + 1) * (2 * C.Lmax * Fintype.card α + 1))
      ≤ C.budget := by
    refine le_trans ?_ hbudget
    have hx : 0 ≤ 8 * lg / (C.εd ^ 2 * C.θpt ^ 2) + 4 * C.n₀ / (3 * C.εd) := by positivity
    have hy : 0 ≤ 8 * ((Fintype.card σ : ℝ) * Fintype.card α + S + 1) * C.m / C.εd
      + 32 * lg / C.εd ^ 2 + 1 := by positivity
    have hP : (0 : ℝ) ≤ (S + Fintype.card σ + 1) := by positivity
    have hQ : (0 : ℝ) ≤ 2 * C.Lmax * Fintype.card α + 1 := by positivity
    nlinarith only [mul_nonneg (mul_nonneg hP hQ) hx, mul_nonneg hP hy]
  have hadm : Knobs.Admissible σ κ ε θ D L C κs ⟨S, nEnd, nRec, hP, hS, ρ, 3 * θ, θgs, θgpt,
      2 * C.θpt, C.εd / 4, C.εd / 2, Xe, 2 * C.θs, C.n₀, 1, 4 * κs, 1 / p₀⟩ := by
    refine
      { κ_pos := hκ
        ε_nonneg := hε0
        θ_nonneg := hθ0
        records_slack := by dsimp only; rw [hρ]; linarith
        l_nonneg := by dsimp only; positivity
        θg_ge := by dsimp only; linarith
        θgpt_ge := by
          dsimp only; rw [hθgpt]
          have : 0 ≤ 8 * p₀ * Λ / C.φpt := by positivity
          linarith
        θgpt_le := by dsimp only; linarith
        θe_ge := by dsimp only; linarith
        m_pos := hm0
        Lmax_ge := hLmax
        L_pos := hL
        a_pos := ha0
        a_le := ha1
        θpt_nonneg := hθpt0
        θpt_le := by linarith
        εd_nonneg := hεd0.le
        εd_le := hεd1
        θpt'_nonneg := by dsimp only; linarith
        θpt'_le := by dsimp only; linarith
        θr_nonneg := by dsimp only; linarith
        θr_le_one := by dsimp only; linarith
        εd'_nonneg := by dsimp only; linarith
        εd'_le := by dsimp only; linarith
        θr_le := by dsimp only; nlinarith only [hθpt4, hεd0]
        θgs_nonneg := hgs0
        θgs_le := by dsimp only; linarith
        φe_nonneg := hφe0
        φpt_nonneg := hφpt_pos.le
        φpt_le := by linarith
        n₀_le_nEnd := hn₀E
        n₀_le_hS := hn₀S
        pt_level := hpt
        success_level := hsucc
        success_reach := hreach
        φpt_nEnd := hφE
        exc_gt := fun j => by dsimp only; rw [hlg]; linarith [hexc_gt j]
        start_test := fun t h ht hlt => by
          dsimp only
          have : (C.n₀ : ℝ) ≤ t := by exact_mod_cast ht
          refine binomSfGe_half_le t h C.θs θgs C.a hθs1 hgs0 h8gs ha0 ha2 ?_ hlt
          rw [hlg]; nlinarith only [hn₀s, this, sq_nonneg C.θs]
        pt_test := fun t h ht hlt => by
          dsimp only
          have : (C.n₀ : ℝ) ≤ t := by exact_mod_cast ht
          refine binomSfGe_half_le t h C.θpt (2 * θgpt) C.a (by linarith) (by linarith)
            (by linarith) ha0 ha2 ?_ hlt
          rw [hlg]; nlinarith only [hn₀pt, this, sq_nonneg C.θpt]
        φpt_test := fun t ht => by
          have : (C.n₀ : ℝ) ≤ t := by exact_mod_cast ht
          refine (binomSfGe_le t _ (C.φpt / 2) (C.φpt / 2) (by positivity) (by linarith)
            (by positivity) (by linarith [Nat.le_ceil (C.φpt * t)])).trans ?_
          rw [show -2 * (t : ℝ) * (C.φpt / 2) ^ 2 = -(t * C.φpt ^ 2 / 2) by ring]
          exact exp_neg_le_of_le ha0 (by rw [hlg]; nlinarith only [hn₀φ, this, sq_nonneg C.φpt])
        Xe_nonneg := by
          dsimp only; rw [hXe]
          have : 0 ≤ 1 + 12 * θ * C.Lmax := by
            have : 0 ≤ θ * C.Lmax := mul_nonneg hθ0 (Nat.cast_nonneg _)
            linarith
          positivity
        exc_le := fun j => by dsimp only; rw [hXe]; exact hexc_le j
        θs'_nonneg := by dsimp only; linarith
        n₀_le_Xs := le_rfl
        start_lin := fun t h ht hh => by
          dsimp only at hh
          have : (C.n₀ : ℝ) ≤ t := by exact_mod_cast ht
          refine (binomSfGe_le t h C.θs C.θs hθs0 hθs1 hθs0 (by
            have : (0 : ℝ) ≤ C.n₀ := Nat.cast_nonneg _
            linarith)).trans_lt ?_
          rw [show -2 * (t : ℝ) * C.θs ^ 2 = -(2 * t * C.θs ^ 2) by ring]
          exact exp_neg_lt_of_lt ha0
            (by rw [hlg]; nlinarith only [hn₀s, this, sq_nonneg C.θs, hlg0])
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
      * Real.exp (-(C.φe * (C.θe / 2 - 2 * (3 * θ)) / 2 / (2 * p₀ * (L + 1)
        * (1 + 4 * (3 / 2 * θ) * (Fintype.card σ + S + 1))))) ≤ δ / 8 := by
    refine mul_exp_le hδ0 (by positivity) le_rfl hK ?_
    rw [le_div_iff₀ (by positivity)]
    have e1 : 1 + 4 * (3 / 2 * θ) * ((Fintype.card σ : ℝ) + S + 1) ≤ 2 := by linarith
    have e2 := mul_le_mul_of_nonneg_left e1
      (show 0 ≤ 2 * p₀ * (L + 1) * Λ from mul_nonneg (by positivity) hΛ0.le)
    have e3 : C.φe * (C.θe / 4) ≤ C.φe * (C.θe / 2 - 2 * (3 * θ)) :=
      mul_le_mul_of_nonneg_left (by linarith) hφe0
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
      * Real.exp (-(θgpt * C.φpt / 2 / (2 * p₀
        * (1 + 4 * ((L + 1) * (Fintype.card σ + S + 1) * (3 / 2 * θ)))))) ≤ δ / 8 := by
    refine mul_exp_le hδ0 (by positivity) hK0 hK ?_
    rw [le_div_iff₀ (by positivity), hgφ]
    have e1 : 1 + 4 * ((L + 1) * ((Fintype.card σ : ℝ) + S + 1) * (3 / 2 * θ)) ≤ 2 := by
      linarith
    have e2 := mul_le_mul_of_nonneg_left e1 (show 0 ≤ 2 * (p₀ * Λ) by linarith)
    have e3 : 0 ≤ 3 * ((L + 1) * ((Fintype.card σ : ℝ) + S + 1)) * θ * C.φpt :=
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
      * subT C (Fintype.card α) nEnd nRec) ρ Xe (2 * C.θs) C.n₀ 1 (4 * κs) ≤ δ / 8 := by
    unfold fakeRun
    set W : ℝ := (((S + Fintype.card σ + 1) * subT C (Fintype.card α) nEnd nRec : ℕ) : ℝ)
    have he : Real.exp 1 - 1 ≤ 2 := by linarith [Real.exp_one_lt_three]
    have hX0 : 0 ≤ C.θe * ((L + 1) * C.Lmax) + 2 * C.θs + 2 * C.Lmax * Fintype.card α * C.φe := by
      have : 0 ≤ C.θe * ((L + 1) * C.Lmax) := mul_nonneg hθe0 (by positivity)
      have : 0 ≤ 2 * C.Lmax * Fintype.card α * C.φe := mul_nonneg (by positivity) hφe0
      linarith
    set X := C.θe * ((L + 1) * C.Lmax) + 2 * C.θs + 2 * C.Lmax * Fintype.card α * C.φe
    have hc0 : 0 ≤ 4 * κs * X + (Real.exp 1 - 1) * ρ :=
      add_nonneg (mul_nonneg (by linarith) hX0)
        (mul_nonneg (by linarith [Real.add_one_le_exp (1 : ℝ)]) hρ0)
    have hWc := mul_le_mul_of_nonneg_right hW hc0
    have h1 : (Real.exp 1 - 1) * ρ ≤ 2 * ρ := mul_le_mul_of_nonneg_right he hρ0
    have h2 := mul_le_mul_of_nonneg_left h1 hb0
    have h3 := mul_le_mul_of_nonneg_left hρle (show (0 : ℝ) ≤ 2 * C.budget by positivity)
    have hfin : -1 * ((S + 1) * C.m) + 4 * κs * (2 * C.Lmax * Fintype.card α * Xe + C.n₀)
        + W * (4 * κs * X + (Real.exp 1 - 1) * ρ) ≤ -Real.log (8 / δ) := by
      rw [hXe]
      nlinarith only [hWc, h2, h3, hm1, hm2, hm3, hm4]
    calc Real.exp (-1 * ((S + 1) * C.m) + 4 * κs * (2 * C.Lmax * Fintype.card α * Xe + C.n₀)
          + W * (4 * κs * X + (Real.exp 1 - 1) * ρ))
        ≤ Real.exp (-Real.log (8 / δ)) := Real.exp_le_exp.2 hfin
      _ = δ / 8 := by rw [Real.exp_neg, Real.exp_log (by positivity)]; field_simp
  have hsub : ((S : ℝ) + Fintype.card σ + 1) * subOpen C (Fintype.card α) (Fintype.card σ) nRec
      ((S + 1) * C.m) (C.εd / 4) (termLevel nEnd hP hS (2 * C.θpt) (C.εd / 2))
      ≤ ((S : ℝ) + Fintype.card σ + 1) * ((2 * C.Lmax * Fintype.card α + 1) * (2 * C.a) + C.a) := by
    unfold subOpen subVersions
    push_cast
    gcongr
  have hTL : ((S : ℝ) + Fintype.card σ + 1) * ((2 * C.Lmax * Fintype.card α + 1) * (2 * C.a) + C.a)
      + (2 ^ (C.Lmax + 1) * Fintype.card α + C.budget + 3 * (C.budget * C.budget)) * C.a
      ≤ δ / 4 := by
    have hP0 : (0 : ℝ) ≤ (S : ℝ) + Fintype.card σ + 1 := by positivity
    have hQ1 : (1 : ℝ) ≤ 2 * C.Lmax * Fintype.card α + 1 := le_add_of_nonneg_left (by positivity)
    have hb1 : (1 : ℝ) ≤ C.budget := by
      have : (1 : ℝ) ≤ ((S : ℝ) + Fintype.card σ + 1) * (2 * C.Lmax * Fintype.card α + 1) :=
        one_le_mul_of_one_le_of_one_le (by linarith) hQ1
      linarith
    have hPa : 0 ≤ ((S : ℝ) + Fintype.card σ + 1) * C.a := mul_nonneg hP0 ha0.le
    have : ((S : ℝ) + Fintype.card σ + 1) * ((2 * C.Lmax * Fintype.card α + 1) * (2 * C.a) + C.a)
        ≤ 3 / 2 * C.budget * C.a := by
      have h1 := mul_le_mul_of_nonneg_left hQ1 hPa
      have h2 := mul_le_mul_of_nonneg_right hbig (show 0 ≤ 3 / 4 * C.a by linarith)
      nlinarith only [h1, h2]
    nlinarith only [this, ha, hb1, ha0, mul_nonneg (sub_nonneg.2 hb1) (mul_nonneg hb0 ha0.le)]
  obtain ⟨Xs, hXs⟩ : ∃ x : ℝ, x = ((S : ℝ) + Fintype.card σ + 1)
      * ((2 * C.Lmax * Fintype.card α + 1) * (2 * C.a) + C.a) := ⟨_, rfl⟩
  obtain ⟨XT, hXT⟩ : ∃ x : ℝ,
      x = (2 ^ (C.Lmax + 1) * Fintype.card α + C.budget + 3 * (C.budget * C.budget)) * C.a :=
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
