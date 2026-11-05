import OrthoDFA.Proofs.ClaimLoopArith
import OrthoDFA.Proofs.LoopReadsArith

/-!
# The loop's calls, asymptotically: the arithmetic

In `ClaimLoopArith`'s inverted coordinates every count and cap is at most `2^c X^k` for
`X = (L + |Q|) Ae iδ R iξ ip is` (`pb_add`, `pb_mul`, and `pb_log`, which takes a log down to
`X¹`), so the loop's calls are at most `2^200 X^50` (`rel_reads_le`).
-/

noncomputable section

namespace OrthoDFA

open Real Asymptotics

section Pow

variable {X a b y : ℝ}

lemma pb_mono {c k c' k' : ℕ} (hX : 1 ≤ X) (h : a ≤ 2 ^ c * X ^ k) (hc : c ≤ c') (hk : k ≤ k') :
    a ≤ 2 ^ c' * X ^ k' :=
  h.trans (mul_le_mul (pow_le_pow_right₀ (by norm_num) hc) (pow_le_pow_right₀ hX hk)
    (pow_nonneg (by linarith) _) (by positivity))

lemma pb_add {c₁ k₁ c₂ k₂ : ℕ} (hX : 1 ≤ X) (ha : a ≤ 2 ^ c₁ * X ^ k₁) (hb : b ≤ 2 ^ c₂ * X ^ k₂) :
    a + b ≤ 2 ^ (max c₁ c₂ + 1) * X ^ (max k₁ k₂) := by
  have h1 := pb_mono hX ha (le_max_left c₁ c₂) (le_max_left k₁ k₂)
  have h2 := pb_mono hX hb (le_max_right c₁ c₂) (le_max_right k₁ k₂)
  rw [pow_succ]
  linarith

lemma pb_mul {c₁ k₁ c₂ k₂ : ℕ} (hX : 0 ≤ X) (hb0 : 0 ≤ b)
    (ha : a ≤ 2 ^ c₁ * X ^ k₁) (hb : b ≤ 2 ^ c₂ * X ^ k₂) :
    a * b ≤ 2 ^ (c₁ + c₂) * X ^ (k₁ + k₂) := by
  have h := mul_le_mul ha hb hb0 (mul_nonneg (by positivity) (pow_nonneg hX k₁))
  rw [pow_add, pow_add]
  linarith

lemma pb_X (h : a ≤ X) : a ≤ 2 ^ 0 * X ^ 1 := by simpa using h

lemma pb_const {c : ℕ} (h : a ≤ 2 ^ c) : a ≤ 2 ^ c * X ^ 0 := by simpa using h

lemma pb_log {c k c' : ℕ} (hX : 1 ≤ X) (hy : 0 < y) (h : y ≤ 2 ^ c * X ^ k)
    (hc : c + k ≤ 2 ^ c') : log y ≤ 2 ^ c' * X ^ 1 := by
  have h1 : log y ≤ c * log 2 + k * log X := by
    calc log y ≤ log (2 ^ c * X ^ k) := log_le_log hy h
      _ = _ := by
        rw [log_mul (by positivity) (pow_pos (by linarith) k).ne', log_pow, log_pow]
  have hl2 : log 2 ≤ 1 := by have := Real.log_two_lt_d9; linarith
  have hlX : log X ≤ X := (Real.log_le_sub_one_of_pos (by linarith)).trans (by linarith)
  have hck : ((c + k : ℕ) : ℝ) ≤ ((2 ^ c' : ℕ) : ℝ) := by exact_mod_cast hc
  have hc0 : (0 : ℝ) ≤ c := Nat.cast_nonneg _
  have hk0 : (0 : ℝ) ≤ k := Nat.cast_nonneg _
  have e1 : (c : ℝ) * log 2 ≤ c * X := mul_le_mul_of_nonneg_left (by linarith) hc0
  have e2 : (k : ℝ) * log X ≤ k * X := mul_le_mul_of_nonneg_left hlX hk0
  have e3 : ((c + k : ℕ) : ℝ) * X ≤ ((2 ^ c' : ℕ) : ℝ) * X :=
    mul_le_mul_of_nonneg_right hck (by linarith)
  push_cast at e3
  rw [pow_one]
  linarith

end Pow

lemma loopSC_inv (ℓ Q A Ae iδ c ξ pAP η₀ : ℝ) :
    loopSC ℓ Q A Ae (1 / iδ) c ξ pAP η₀
      = 8 * (Q + 4) * loopCount ℓ Q A Ae (1 / iδ) c ξ pAP η₀
        + 16 * log (16 * (Q + 2) * (8 * (Q + 1) * iδ)) + 2 := by
  unfold loopSC
  rw [show 16 * (Q + 2) / (1 / iδ / (8 * (Q + 1))) = 16 * (Q + 2) * (8 * (Q + 1) * iδ) by
    field_simp]

lemma loopHN_inv (ℓ Q A Ae iδ ic ξ pAP η₀ : ℝ) :
    loopHN ℓ Q A Ae (1 / iδ) (1 / ic) ξ pAP η₀
      = ℓ + loopBudget ℓ Q A Ae (1 / iδ) (1 / ic) + 3
        + 192 * (4 * (Ae + 1) * ℓ * ic)
          * log (4 * (Q + 1) * loopSC ℓ Q A Ae (1 / iδ) (1 / ic) ξ pAP η₀
            * (8 * (Q + 1) * iδ)) := by
  unfold loopHN
  rw [div_inv_coord, show 4 * (Q + 1) * loopSC ℓ Q A Ae (1 / iδ) (1 / ic) ξ pAP η₀
      / (1 / iδ / (8 * (Q + 1))) = 4 * (Q + 1) * loopSC ℓ Q A Ae (1 / iδ) (1 / ic) ξ pAP η₀
        * (8 * (Q + 1) * iδ) by field_simp]


section Pow

variable {X : ℝ}

lemma pb_D {Q iδ : ℝ} (hX : 1 ≤ X) (hQ : 0 ≤ Q) (hiδ : 0 ≤ iδ) (hQX : Q ≤ X) (hδX : iδ ≤ X) :
    0 ≤ 8 * (Q + 1) * iδ ∧ 8 * (Q + 1) * iδ ≤ 2 ^ 4 * X ^ 2 := by
  have bQ1 : Q + 1 ≤ 2 ^ 1 * X ^ 1 :=
    pb_mono hX (pb_add hX (pb_X hQX) (pb_const (X := X) (by norm_num : (1 : ℝ) ≤ 2 ^ 0)))
      le_rfl le_rfl
  exact ⟨by positivity, pb_mul (by linarith) hiδ
    (pb_mul (by linarith) (by linarith) (pb_const (X := X) (by norm_num : (8 : ℝ) ≤ 2 ^ 3)) bQ1)
    (pb_X hδX)⟩

/-- `loopCount_inv`'s right side, at `P ≤ 2^34 X^16` and `Θι, Θ ≤ 2^19 X^4`. -/
lemma count_pow {Q ℓ iδ ip is P Θ Θι : ℝ} (hX : 1 ≤ X) (hQ : 1 ≤ Q) (hℓ : 1 ≤ ℓ)
    (hiδ : 1 ≤ iδ) (hip : 1 ≤ ip) (his : 0 ≤ is) (hP : 1 ≤ P) (hΘ : 1 ≤ Θ) (hΘι : 0 ≤ Θι)
    (hQX : Q ≤ X) (hℓX : ℓ ≤ X) (hδX : iδ ≤ X) (hpX : ip ≤ X) (hsX : is ≤ X)
    (bP : P ≤ 2 ^ 34 * X ^ 16) (bΘ : Θ ≤ 2 ^ 19 * X ^ 4) (bΘι : Θι ≤ 2 ^ 19 * X ^ 4) :
    0 ≤ 2 ^ 20 * (Q + 2) * Θι ^ 2 * ℓ ^ 2 * log (128 * (P + 1) * (8 * (Q + 1) * iδ)) * is ^ 4
        + 2 ^ 27 * Θ ^ 2 * ℓ ^ 2 * log (16 * (Q + 4) * P * Θ * ℓ * (8 * (Q + 1) * iδ) ^ 2 * ip)
      ∧ 2 ^ 20 * (Q + 2) * Θι ^ 2 * ℓ ^ 2 * log (128 * (P + 1) * (8 * (Q + 1) * iδ)) * is ^ 4
        + 2 ^ 27 * Θ ^ 2 * ℓ ^ 2 * log (16 * (Q + 4) * P * Θ * ℓ * (8 * (Q + 1) * iδ) ^ 2 * ip)
        ≤ 2 ^ 73 * X ^ 16 := by
  have hX0 : 0 ≤ X := by linarith
  obtain ⟨nD, bD⟩ := pb_D hX (by linarith) (by linarith) hQX hδX
  have hD1 : 1 ≤ 8 * (Q + 1) * iδ := by nlinarith
  have b1 : (1 : ℝ) ≤ 2 ^ 0 * X ^ 0 := pb_const (by norm_num)
  have bQ2 : Q + 2 ≤ 2 ^ 2 * X ^ 1 :=
    pb_mono hX (pb_add hX (pb_X hQX) (pb_const (X := X) (by norm_num : (2 : ℝ) ≤ 2 ^ 1)))
      (by norm_num) (by norm_num)
  have bQ4 : Q + 4 ≤ 2 ^ 3 * X ^ 1 :=
    pb_mono hX (pb_add hX (pb_X hQX) (pb_const (X := X) (by norm_num : (4 : ℝ) ≤ 2 ^ 2)))
      (by norm_num) (by norm_num)
  have bP1 : P + 1 ≤ 2 ^ 35 * X ^ 16 := pb_mono hX (pb_add hX bP b1) (by norm_num) (by norm_num)
  have y1 : 1 ≤ 128 * (P + 1) * (8 * (Q + 1) * iδ) := by nlinarith
  have bl1 := pb_log hX (by linarith) (pb_mul hX0 nD
    (pb_mul hX0 (by linarith) (pb_const (X := X) (by norm_num : (128 : ℝ) ≤ 2 ^ 7)) bP1) bD)
    (c' := 6) (by norm_num)
  have hl1 := Real.log_nonneg y1
  have y2 : 1 ≤ 16 * (Q + 4) * P * Θ * ℓ * (8 * (Q + 1) * iδ) ^ 2 * ip := by
    have m := @one_le_mul_of_one_le_of_one_le ℝ _
    exact m (m (m (m (m (by linarith) hP) hΘ) hℓ) (one_le_pow₀ hD1)) hip
  have bD2 : (8 * (Q + 1) * iδ) ^ 2 ≤ 2 ^ (4 + 4) * X ^ (2 + 2) := by
    rw [sq]; exact pb_mul hX0 nD bD bD
  have bl2 := pb_log hX (by linarith)
    (pb_mul hX0 (by linarith) (pb_mul hX0 (sq_nonneg _) (pb_mul hX0 (by linarith)
      (pb_mul hX0 (by linarith) (pb_mul hX0 (by linarith) (pb_mul hX0 (by linarith)
        (pb_const (X := X) (by norm_num : (16 : ℝ) ≤ 2 ^ 4)) bQ4) bP) bΘ) (pb_X hℓX)) bD2)
      (pb_X hpX)) (c' := 7) (by norm_num)
  have hl2 := Real.log_nonneg y2
  have bΘι2 : Θι ^ 2 ≤ 2 ^ (19 + 19) * X ^ (4 + 4) := by
    rw [sq]; exact pb_mul hX0 hΘι bΘι bΘι
  have bΘ2 : Θ ^ 2 ≤ 2 ^ (19 + 19) * X ^ (4 + 4) := by
    rw [sq]; exact pb_mul hX0 (by linarith) bΘ bΘ
  have bℓ2 : ℓ ^ 2 ≤ 2 ^ (0 + 0) * X ^ (1 + 1) := by
    rw [sq]; exact pb_mul hX0 (by linarith) (pb_X hℓX) (pb_X hℓX)
  have bs4 : is ^ 4 ≤ 2 ^ (0 + 0 + 0 + 0) * X ^ (1 + 1 + 1 + 1) := by
    rw [show is ^ 4 = is * is * is * is by ring]
    exact pb_mul hX0 his (pb_mul hX0 his (pb_mul hX0 his (pb_X hsX) (pb_X hsX)) (pb_X hsX))
      (pb_X hsX)
  have bC1 := pb_mul hX0 (pow_nonneg his 4) (pb_mul hX0 hl1 (pb_mul hX0 (sq_nonneg _)
    (pb_mul hX0 (sq_nonneg _) (pb_mul hX0 (by linarith)
      (pb_const (X := X) (le_refl ((2 : ℝ) ^ 20))) bQ2) bΘι2) bℓ2) bl1) bs4
  have bC2 := pb_mul hX0 hl2 (pb_mul hX0 (sq_nonneg _) (pb_mul hX0 (sq_nonneg _)
    (pb_const (X := X) (le_refl ((2 : ℝ) ^ 27))) bΘ2) bℓ2) bl2
  refine ⟨add_nonneg (mul_nonneg (mul_nonneg (mul_nonneg (mul_nonneg (mul_nonneg
      (by norm_num) (by linarith)) (sq_nonneg _)) (sq_nonneg _)) hl1) (pow_nonneg his 4))
    (mul_nonneg (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg _)) (sq_nonneg _)) hl2), ?_⟩
  exact pb_mono hX (pb_add hX bC1 bC2) (by norm_num) (by norm_num)

/-- `loopBudget` at `|Q|, |Σ|, Ae, ℓ, iδ, ic ≤ X`. -/
lemma budget_pow {ℓ Q A Ae iδ ic : ℝ} (hX : 1 ≤ X) (hQ : 1 ≤ Q) (hA : 0 ≤ A) (hAe : 0 ≤ Ae)
    (hℓ : 0 ≤ ℓ) (hiδ : 1 ≤ iδ) (hic : 0 ≤ ic) (hℓX : ℓ ≤ X) (hQX : Q ≤ X) (hAX : A ≤ X)
    (hAeX : Ae ≤ X) (hδX : iδ ≤ X) (hcX : ic ≤ X) :
    0 ≤ loopBudget ℓ Q A Ae (1 / iδ) (1 / ic)
      ∧ loopBudget ℓ Q A Ae (1 / iδ) (1 / ic) ≤ 2 ^ 21 * X ^ 7 := by
  have hX0 : 0 ≤ X := by linarith
  obtain ⟨nD, bD⟩ := pb_D hX (by linarith) (by linarith) hQX hδX
  have b1 : (1 : ℝ) ≤ 2 ^ 0 * X ^ 0 := pb_const (by norm_num)
  have bQ2 : Q + 2 ≤ 2 ^ 2 * X ^ 1 :=
    pb_mono hX (pb_add hX (pb_X hQX) (pb_const (X := X) (by norm_num : (2 : ℝ) ≤ 2 ^ 1)))
      (by norm_num) (by norm_num)
  have bAe : Ae + 1 ≤ 2 ^ 1 * X ^ 1 := pb_mono hX (pb_add hX (pb_X hAeX) b1) le_rfl le_rfl
  have nS' : 0 ≤ (Q + 2) * A + 1 := by
    have := mul_nonneg (by linarith : 0 ≤ Q + 2) hA
    linarith
  have bS : Q * ((Q + 2) * A + 1) + 2 * A + 1 ≤ 2 ^ 5 * X ^ 3 :=
    pb_mono hX (pb_add hX (pb_add hX
      (pb_mul hX0 nS' (pb_X hQX) (pb_add hX (pb_mul hX0 hA bQ2 (pb_X hAX)) b1))
      (pb_mul hX0 hA (pb_const (X := X) (by norm_num : (2 : ℝ) ≤ 2 ^ 1)) (pb_X hAX))) b1)
      (by norm_num) (by norm_num)
  have hS1 : 1 ≤ Q * ((Q + 2) * A + 1) + 2 * A + 1 := by
    have := mul_nonneg (by linarith : 0 ≤ Q) nS'
    linarith
  rw [loopBudget_inv]
  generalize Q * ((Q + 2) * A + 1) + 2 * A + 1 = S at bS hS1 ⊢
  have bS1 : S + 1 ≤ 2 ^ 6 * X ^ 3 := pb_mono hX (pb_add hX bS b1) (by norm_num) (by norm_num)
  have hD1 : 1 ≤ 8 * (Q + 1) * iδ := by nlinarith
  have hz : 1 ≤ 16 * S * (S + 1) * (8 * (Q + 1) * iδ) := by
    have m := @one_le_mul_of_one_le_of_one_le ℝ _
    exact m (m (m (by norm_num) hS1) (by linarith)) hD1
  have blz := pb_log hX (by linarith) (pb_mul hX0 nD (pb_mul hX0 (by linarith)
    (pb_mul hX0 (by linarith) (pb_const (X := X) (by norm_num : (16 : ℝ) ≤ 2 ^ 4)) bS) bS1) bD)
    (c' := 5) (by norm_num)
  have hlz := Real.log_nonneg hz
  have nE : 0 ≤ 4 * (Ae + 1) * ic := mul_nonneg (by linarith) hic
  have h0 : 0 ≤ 36 * S * (4 * (Ae + 1) * ic) * log (16 * S * (S + 1) * (8 * (Q + 1) * iδ))
      + S + 1 := by
    have : 0 ≤ 36 * S * (4 * (Ae + 1) * ic) * log (16 * S * (S + 1) * (8 * (Q + 1) * iδ)) :=
      mul_nonneg (mul_nonneg (mul_nonneg (by norm_num) (by linarith)) nE) hlz
    linarith
  refine ⟨mul_nonneg h0 hℓ, ?_⟩
  have h4 := pb_mul hX0 hlz (pb_mul hX0 nE
    (pb_mul hX0 (by linarith) (pb_const (X := X) (by norm_num : (36 : ℝ) ≤ 2 ^ 6)) bS)
    (pb_mul hX0 hic (pb_mul hX0 (by linarith)
      (pb_const (X := X) (by norm_num : (4 : ℝ) ≤ 2 ^ 2)) bAe) (pb_X hcX))) blz
  exact pb_mono hX (pb_mul hX0 hℓ (pb_add hX (pb_add hX h4 bS) b1) (pb_X hℓX))
    (by norm_num) (by norm_num)

/-- `loopSC_inv`'s right side at a prefix count `C ≤ 2^73 X^16`. -/
lemma sc_pow {Q iδ C : ℝ} (hX : 1 ≤ X) (hQ : 1 ≤ Q) (hiδ : 1 ≤ iδ) (hQX : Q ≤ X)
    (hδX : iδ ≤ X) (hC : 0 ≤ C) (bC : C ≤ 2 ^ 73 * X ^ 16) :
    1 ≤ 8 * (Q + 4) * C + 16 * log (16 * (Q + 2) * (8 * (Q + 1) * iδ)) + 2
      ∧ 8 * (Q + 4) * C + 16 * log (16 * (Q + 2) * (8 * (Q + 1) * iδ)) + 2 ≤ 2 ^ 81 * X ^ 17 := by
  have hX0 : 0 ≤ X := by linarith
  obtain ⟨nD, bD⟩ := pb_D hX (by linarith) (by linarith) hQX hδX
  have bQ2 : Q + 2 ≤ 2 ^ 2 * X ^ 1 :=
    pb_mono hX (pb_add hX (pb_X hQX) (pb_const (X := X) (by norm_num : (2 : ℝ) ≤ 2 ^ 1)))
      (by norm_num) (by norm_num)
  have bQ4 : Q + 4 ≤ 2 ^ 3 * X ^ 1 :=
    pb_mono hX (pb_add hX (pb_X hQX) (pb_const (X := X) (by norm_num : (4 : ℝ) ≤ 2 ^ 2)))
      (by norm_num) (by norm_num)
  have hy : 1 ≤ 16 * (Q + 2) * (8 * (Q + 1) * iδ) := by nlinarith
  have hl := Real.log_nonneg hy
  have h8 : 0 ≤ 8 * (Q + 4) * C := mul_nonneg (by linarith) hC
  refine ⟨by linarith, ?_⟩
  have h1 := pb_mul hX0 hC
    (pb_mul hX0 (by linarith) (pb_const (X := X) (by norm_num : (8 : ℝ) ≤ 2 ^ 3)) bQ4) bC
  have h2 := pb_mul hX0 hl (pb_const (X := X) (by norm_num : (16 : ℝ) ≤ 2 ^ 4))
    (pb_log hX (by linarith) (pb_mul hX0 nD (pb_mul hX0 (by linarith)
      (pb_const (X := X) (by norm_num : (16 : ℝ) ≤ 2 ^ 4)) bQ2) bD) (c' := 4) (by norm_num))
  exact pb_mono hX (pb_add hX (pb_add hX h1 h2)
    (pb_const (X := X) (by norm_num : (2 : ℝ) ≤ 2 ^ 1))) (by norm_num) (by norm_num)

/-- `loopHN_inv`'s right side at a probe budget `T ≤ 2^21 X^7` and a scan cap
`M ≤ 2^81 X^17`. -/
lemma hn_pow {Q Ae ℓ iδ ic T M : ℝ} (hX : 1 ≤ X) (hQ : 1 ≤ Q) (hAe : 0 ≤ Ae) (hℓ : 0 ≤ ℓ)
    (hiδ : 1 ≤ iδ) (hic : 0 ≤ ic) (hQX : Q ≤ X) (hAeX : Ae ≤ X) (hℓX : ℓ ≤ X) (hδX : iδ ≤ X)
    (hcX : ic ≤ X) (hT : 0 ≤ T) (hM : 1 ≤ M) (bT : T ≤ 2 ^ 21 * X ^ 7)
    (bM : M ≤ 2 ^ 81 * X ^ 17) :
    0 ≤ ℓ + T + 3 + 192 * (4 * (Ae + 1) * ℓ * ic) * log (4 * (Q + 1) * M * (8 * (Q + 1) * iδ))
      ∧ ℓ + T + 3 + 192 * (4 * (Ae + 1) * ℓ * ic) * log (4 * (Q + 1) * M * (8 * (Q + 1) * iδ))
        ≤ 2 ^ 24 * X ^ 7 := by
  have hX0 : 0 ≤ X := by linarith
  obtain ⟨nD, bD⟩ := pb_D hX (by linarith) (by linarith) hQX hδX
  have b1 : (1 : ℝ) ≤ 2 ^ 0 * X ^ 0 := pb_const (by norm_num)
  have bQ1 : Q + 1 ≤ 2 ^ 1 * X ^ 1 := pb_mono hX (pb_add hX (pb_X hQX) b1) le_rfl le_rfl
  have bAe : Ae + 1 ≤ 2 ^ 1 * X ^ 1 := pb_mono hX (pb_add hX (pb_X hAeX) b1) le_rfl le_rfl
  have hy : 1 ≤ 4 * (Q + 1) * M * (8 * (Q + 1) * iδ) := by
    have := one_le_mul_of_one_le_of_one_le (by linarith : (1 : ℝ) ≤ 4 * (Q + 1)) hM
    nlinarith
  have hl := Real.log_nonneg hy
  have nF : 0 ≤ 4 * (Ae + 1) * ℓ * ic := mul_nonneg (mul_nonneg (by linarith) hℓ) hic
  have hF : 0 ≤ 192 * (4 * (Ae + 1) * ℓ * ic) * log (4 * (Q + 1) * M * (8 * (Q + 1) * iδ)) :=
    mul_nonneg (mul_nonneg (by norm_num) nF) hl
  refine ⟨by linarith, ?_⟩
  have h1 := pb_mul hX0 hl (pb_mul hX0 nF
    (pb_const (X := X) (by norm_num : (192 : ℝ) ≤ 2 ^ 8))
    (pb_mul hX0 hic (pb_mul hX0 hℓ (pb_mul hX0 (by linarith)
      (pb_const (X := X) (by norm_num : (4 : ℝ) ≤ 2 ^ 2)) bAe) (pb_X hℓX)) (pb_X hcX)))
    (pb_log hX (by linarith) (pb_mul hX0 nD (pb_mul hX0 (by linarith)
      (pb_mul hX0 (by linarith) (pb_const (X := X) (by norm_num : (4 : ℝ) ≤ 2 ^ 2)) bQ1) bM) bD)
      (c' := 7) (by norm_num))
  exact pb_mono hX (pb_add hX (pb_add hX (pb_add hX (pb_X hℓX) bT)
    (pb_const (X := X) (by norm_num : (3 : ℝ) ≤ 2 ^ 2))) h1) (by norm_num) (by norm_num)

/-- `loopReads` at its arguments' power bounds. -/
lemma reads_pow {K a b T M N ℓ A : ℝ} (hX : 1 ≤ X) (hK : 0 ≤ K) (ha : 0 ≤ a) (hb : 0 ≤ b)
    (hT : 0 ≤ T) (hM : 0 ≤ M) (hN : 0 ≤ N) (hℓ : 0 ≤ ℓ) (hA : 0 ≤ A)
    (bK : K ≤ 2 ^ 1 * X ^ 1) (ba : a ≤ 2 ^ 78 * X ^ 17) (bb : b ≤ 2 ^ 35 * X ^ 16)
    (bT : T ≤ 2 ^ 21 * X ^ 7) (bM : M ≤ 2 ^ 81 * X ^ 17) (bN : N ≤ 2 ^ 24 * X ^ 7)
    (hℓX : ℓ ≤ X) (hAX : A ≤ X) :
    0 ≤ loopReads K a b T M N ℓ A ∧ loopReads K a b T M N ℓ A ≤ 2 ^ 200 * X ^ 50 := by
  have hX0 : 0 ≤ X := by linarith
  have b1 : (1 : ℝ) ≤ 2 ^ 0 * X ^ 0 := pb_const (by norm_num)
  have bℓ3 : ℓ + 3 ≤ 2 ^ 3 * X ^ 1 :=
    pb_mono hX (pb_add hX (pb_X hℓX) (pb_const (X := X) (by norm_num : (3 : ℝ) ≤ 2 ^ 2)))
      (by norm_num) (by norm_num)
  have bℓ1 : ℓ + 1 ≤ 2 ^ 1 * X ^ 1 := pb_mono hX (pb_add hX (pb_X hℓX) b1) le_rfl le_rfl
  have bT1 : T + 1 ≤ 2 ^ 22 * X ^ 7 := pb_mono hX (pb_add hX bT b1) (by norm_num) (by norm_num)
  have bT2 : T + 2 ≤ 2 ^ 22 * X ^ 7 :=
    pb_mono hX (pb_add hX bT (pb_const (X := X) (by norm_num : (2 : ℝ) ≤ 2 ^ 1)))
      (by norm_num) (by norm_num)
  have bb1 : b + 1 ≤ 2 ^ 36 * X ^ 16 := pb_mono hX (pb_add hX bb b1) (by norm_num) (by norm_num)
  have n1 : 0 ≤ (ℓ + 3) * (T + 1) := mul_nonneg (by linarith) (by linarith)
  have n2 : 0 ≤ N * ((ℓ + 3) * (T + 1)) := mul_nonneg hN n1
  have n3 : 0 ≤ M * (N * ((ℓ + 3) * (T + 1))) := mul_nonneg hM n2
  have n4 : 0 ≤ b * (M * (N * ((ℓ + 3) * (T + 1)))) := mul_nonneg hb n3
  have m1 : 0 ≤ (ℓ + 1) * (T + 1) + (T + 2) * A * (T + 2) :=
    add_nonneg (mul_nonneg (by linarith) (by linarith))
      (mul_nonneg (mul_nonneg (by linarith) hA) (by linarith))
  have m2 : 0 ≤ T * ((ℓ + 1) * (T + 1) + (T + 2) * A * (T + 2)) := mul_nonneg hT m1
  have hin : 0 ≤ K * (b * (M * (N * ((ℓ + 3) * (T + 1))))) + a * (b + 1)
      + b * (T * ((ℓ + 1) * (T + 1) + (T + 2) * A * (T + 2))) :=
    add_nonneg (add_nonneg (mul_nonneg hK n4) (mul_nonneg ha (by linarith))) (mul_nonneg hb m2)
  unfold loopReads
  refine ⟨mul_nonneg hK hin, ?_⟩
  have t1 := pb_mul hX0 n4 bK (pb_mul hX0 n3 bb (pb_mul hX0 n2 bM
    (pb_mul hX0 n1 bN (pb_mul hX0 (by linarith) bℓ3 bT1))))
  have t2 := pb_mul hX0 (by linarith) ba bb1
  have t3 := pb_mul hX0 m2 bb (pb_mul hX0 m1 bT (pb_add hX
    (pb_mul hX0 (by linarith) bℓ1 bT1) (pb_mul hX0 (by linarith) (pb_mul hX0 hA bT2 (pb_X hAX))
      bT2)))
  exact pb_mono hX (pb_mul hX0 hin bK (pb_add hX (pb_add hX t1 t2) t3)) (by norm_num)
    (by norm_num)

end Pow

/-- The loop's calls at `ℓ, |Q|, |Σ|, Ae, iδ, ic, iξ, ip, is ≤ X`. -/
lemma reads_coords {ℓ Q A Ae iδ ic iξ ip is X : ℝ} (hℓ : 1 ≤ ℓ) (hQ : 1 ≤ Q) (hA : 1 ≤ A)
    (hAe : 1 ≤ Ae) (hiδ : 1 ≤ iδ) (hic : 1 ≤ ic) (hiξ : 1 ≤ iξ) (hip : 1 ≤ ip) (his : 1 ≤ is)
    (hℓX : ℓ ≤ X) (hQX : Q ≤ X) (hAX : A ≤ X) (hAeX : Ae ≤ X) (hδX : iδ ≤ X) (hcX : ic ≤ X)
    (hξX : iξ ≤ X) (hpX : ip ≤ X) (hsX : is ≤ X) :
    0 ≤ loopReads (Q + 1)
        (6 * (Q + 2) * loopCount ℓ Q A Ae (1 / iδ) (1 / ic) (1 / iξ) (1 / ip) (1 / 2 - 1 / is))
        (loopPool ℓ Q A Ae (1 / iδ) (1 / ic) (1 / iξ) (1 / ip) (1 / 2 - 1 / is) + 1)
        (loopBudget ℓ Q A Ae (1 / iδ) (1 / ic))
        (loopSC ℓ Q A Ae (1 / iδ) (1 / ic) (1 / iξ) (1 / ip) (1 / 2 - 1 / is))
        (loopHN ℓ Q A Ae (1 / iδ) (1 / ic) (1 / iξ) (1 / ip) (1 / 2 - 1 / is)) ℓ A
      ∧ loopReads (Q + 1)
        (6 * (Q + 2) * loopCount ℓ Q A Ae (1 / iδ) (1 / ic) (1 / iξ) (1 / ip) (1 / 2 - 1 / is))
        (loopPool ℓ Q A Ae (1 / iδ) (1 / ic) (1 / iξ) (1 / ip) (1 / 2 - 1 / is) + 1)
        (loopBudget ℓ Q A Ae (1 / iδ) (1 / ic))
        (loopSC ℓ Q A Ae (1 / iδ) (1 / ic) (1 / iξ) (1 / ip) (1 / 2 - 1 / is))
        (loopHN ℓ Q A Ae (1 / iδ) (1 / ic) (1 / iξ) (1 / ip) (1 / 2 - 1 / is)) ℓ A
        ≤ 2 ^ 200 * X ^ 50 := by
  have hX : 1 ≤ X := hQ.trans hQX
  have hX0 : 0 ≤ X := by linarith
  have hP1 := loopPool_ge (A := A) hℓ hQ hAe hiδ hic hiξ hip his
  have bP := loopPool_le_pow hℓ hQ hA hAe hiδ hic hiξ hip his hℓX hQX hAX hAeX hδX hcX hξX hpX
    hsX
  have hΘ := (loopΘ_le hQ hAe hic hiξ his).1
  have bΘ := loopΘ_le_pow hQ hAe hic hiξ his hQX hAeX hcX hξX hsX
  have hΘι0 : 0 ≤ loopΘι Q Ae (1 / ic) (1 / iξ) := by
    rw [loopΘι_inv]
    exact add_nonneg (by linarith) (mul_nonneg (mul_nonneg (mul_nonneg (by norm_num)
      (by linarith)) (by linarith)) (mul_nonneg (by linarith) (by linarith)))
  have hΘιΘ : loopΘι Q Ae (1 / ic) (1 / iξ) ≤ loopΘ Q Ae (1 / ic) (1 / iξ) (1 / 2 - 1 / is) := by
    rw [loopΘ_inv, loopΘι_inv]
    have : 0 ≤ (10 * ic + 8 * iξ) * is ^ 2 / 2 :=
      div_nonneg (mul_nonneg (by linarith) (sq_nonneg _)) (by norm_num)
    linarith
  have hC : 0 ≤ loopCount ℓ Q A Ae (1 / iδ) (1 / ic) (1 / iξ) (1 / ip) (1 / 2 - 1 / is)
      ∧ loopCount ℓ Q A Ae (1 / iδ) (1 / ic) (1 / iξ) (1 / ip) (1 / 2 - 1 / is)
        ≤ 2 ^ 73 * X ^ 16 := by
    rw [loopCount_inv]
    exact count_pow hX hQ hℓ hiδ hip (by linarith) hP1 hΘ hΘι0 hQX hℓX hδX hpX hsX bP bΘ
      (hΘιΘ.trans bΘ)
  have hT := budget_pow hX hQ (by linarith) (by linarith) (by linarith) hiδ (by linarith) hℓX
    hQX hAX hAeX hδX hcX
  have hM : 1 ≤ loopSC ℓ Q A Ae (1 / iδ) (1 / ic) (1 / iξ) (1 / ip) (1 / 2 - 1 / is)
      ∧ loopSC ℓ Q A Ae (1 / iδ) (1 / ic) (1 / iξ) (1 / ip) (1 / 2 - 1 / is)
        ≤ 2 ^ 81 * X ^ 17 := by
    rw [loopSC_inv]
    exact sc_pow hX hQ hiδ hQX hδX hC.1 hC.2
  have hN : 0 ≤ loopHN ℓ Q A Ae (1 / iδ) (1 / ic) (1 / iξ) (1 / ip) (1 / 2 - 1 / is)
      ∧ loopHN ℓ Q A Ae (1 / iδ) (1 / ic) (1 / iξ) (1 / ip) (1 / 2 - 1 / is)
        ≤ 2 ^ 24 * X ^ 7 := by
    rw [loopHN_inv]
    exact hn_pow hX hQ (by linarith) (by linarith) hiδ (by linarith) hQX hAeX hℓX hδX hcX hT.1
      hM.1 hT.2 hM.2
  have b1 : (1 : ℝ) ≤ 2 ^ 0 * X ^ 0 := pb_const (by norm_num)
  have bQ2 : Q + 2 ≤ 2 ^ 2 * X ^ 1 :=
    pb_mono hX (pb_add hX (pb_X hQX) (pb_const (X := X) (by norm_num : (2 : ℝ) ≤ 2 ^ 1)))
      (by norm_num) (by norm_num)
  exact reads_pow hX (by linarith) (mul_nonneg (by linarith) hC.1) (by linarith) hT.1
    (by linarith) hN.1 (by linarith) (by linarith)
    (pb_mono hX (pb_add hX (pb_X hQX) b1) le_rfl le_rfl)
    (pb_mono hX (pb_mul hX0 hC.1 (pb_mul hX0 (by linarith)
      (pb_const (X := X) (by norm_num : (6 : ℝ) ≤ 2 ^ 3)) bQ2) hC.2) (by norm_num) (by norm_num))
    (pb_mono hX (pb_add hX bP b1) (by norm_num) (by norm_num)) hT.2 hM.2 hN.2 hℓX hAX

/-- The loop's calls at the summed rates. -/
lemma rel_reads_le {L Q A Ae iδ R iξ ip is : ℝ} (hL : 0 ≤ L) (hQ : 1 ≤ Q) (hA : 2 ≤ A)
    (hAAe : A ≤ Ae) (hiδ : 2 ≤ iδ) (hR : 1 ≤ R) (hiξ : 1 ≤ iξ) (hip : 1 ≤ ip) (his : 2 ≤ is) :
    0 ≤ loopReads (Q + 1)
        (6 * (Q + 2) * loopCount (L + Q + 4) Q A (Ae + Q) (1 / iδ) (1 / (2 * R * iξ))
          (1 / (4 * iξ)) (1 / ip) (1 / 2 - 1 / is))
        (loopPool (L + Q + 4) Q A (Ae + Q) (1 / iδ) (1 / (2 * R * iξ)) (1 / (4 * iξ)) (1 / ip)
          (1 / 2 - 1 / is) + 1)
        (loopBudget (L + Q + 4) Q A (Ae + Q) (1 / iδ) (1 / (2 * R * iξ)))
        (loopSC (L + Q + 4) Q A (Ae + Q) (1 / iδ) (1 / (2 * R * iξ)) (1 / (4 * iξ)) (1 / ip)
          (1 / 2 - 1 / is))
        (loopHN (L + Q + 4) Q A (Ae + Q) (1 / iδ) (1 / (2 * R * iξ)) (1 / (4 * iξ)) (1 / ip)
          (1 / 2 - 1 / is))
        (L + Q + 4) A
      ∧ loopReads (Q + 1)
        (6 * (Q + 2) * loopCount (L + Q + 4) Q A (Ae + Q) (1 / iδ) (1 / (2 * R * iξ))
          (1 / (4 * iξ)) (1 / ip) (1 / 2 - 1 / is))
        (loopPool (L + Q + 4) Q A (Ae + Q) (1 / iδ) (1 / (2 * R * iξ)) (1 / (4 * iξ)) (1 / ip)
          (1 / 2 - 1 / is) + 1)
        (loopBudget (L + Q + 4) Q A (Ae + Q) (1 / iδ) (1 / (2 * R * iξ)))
        (loopSC (L + Q + 4) Q A (Ae + Q) (1 / iδ) (1 / (2 * R * iξ)) (1 / (4 * iξ)) (1 / ip)
          (1 / 2 - 1 / is))
        (loopHN (L + Q + 4) Q A (Ae + Q) (1 / iδ) (1 / (2 * R * iξ)) (1 / (4 * iξ)) (1 / ip)
          (1 / 2 - 1 / is))
        (L + Q + 4) A
        ≤ 2 ^ 200 * ((L + Q) * Ae * iδ * R * iξ * ip * is) ^ 50 := by
  have m := @one_le_mul_of_one_le_of_one_le ℝ _
  have hP : 1 ≤ L + Q := by linarith
  have hP0 : 0 ≤ L + Q := by linarith
  have hAe : 2 ≤ Ae := hA.trans hAAe
  have hRξ : 1 ≤ R * iξ := m hR hiξ
  have hV : 1 ≤ R * iξ * ip := m hRξ hip
  have hδs : 4 ≤ iδ * is := by nlinarith
  have hPAe : L + Q ≤ (L + Q) * Ae := by nlinarith
  have hAeP : Ae ≤ (L + Q) * Ae := by nlinarith
  -- `X = (L + |Q|) Ae iδ R iξ ip is` against each coordinate
  have hXP : 8 * (L + Q) ≤ (L + Q) * Ae * iδ * R * iξ * ip * is := by
    rw [show (L + Q) * Ae * iδ * R * iξ * ip * is = (L + Q) * (Ae * (iδ * is) * (R * iξ * ip)) by
      ring]
    have h1 : 8 ≤ Ae * (iδ * is) := by nlinarith
    have h2 : 8 ≤ Ae * (iδ * is) * (R * iξ * ip) := by nlinarith
    nlinarith [mul_le_mul_of_nonneg_left h2 hP0]
  have hXA : 2 * ((L + Q) * Ae) ≤ (L + Q) * Ae * iδ * R * iξ * ip * is := by
    rw [show (L + Q) * Ae * iδ * R * iξ * ip * is = (L + Q) * Ae * (iδ * is * (R * iξ * ip)) by
      ring]
    have h2 : 2 ≤ iδ * is * (R * iξ * ip) := by nlinarith
    nlinarith [mul_le_mul_of_nonneg_left h2 (by linarith : 0 ≤ (L + Q) * Ae)]
  have hXδ : iδ ≤ (L + Q) * Ae * iδ * R * iξ * ip * is := by
    rw [show (L + Q) * Ae * iδ * R * iξ * ip * is = iδ * ((L + Q) * Ae * R * iξ * ip * is) by
      ring]
    exact le_mul_of_one_le_right (by linarith)
      (m (m (m (m (m hP (by linarith)) hR) hiξ) hip) (by linarith))
  have hXp : ip ≤ (L + Q) * Ae * iδ * R * iξ * ip * is := by
    rw [show (L + Q) * Ae * iδ * R * iξ * ip * is = ip * ((L + Q) * Ae * iδ * R * iξ * is) by
      ring]
    exact le_mul_of_one_le_right (by linarith)
      (m (m (m (m (m hP (by linarith)) (by linarith)) hR) hiξ) (by linarith))
  have hXs : is ≤ (L + Q) * Ae * iδ * R * iξ * ip * is :=
    le_mul_of_one_le_left (by linarith) (m (m (m (m (m hP (by linarith)) (by linarith)) hR) hiξ)
      hip)
  have hXc : 8 * (R * iξ) ≤ (L + Q) * Ae * iδ * R * iξ * ip * is := by
    rw [show (L + Q) * Ae * iδ * R * iξ * ip * is = R * iξ * ((L + Q) * Ae * (iδ * is) * ip) by
      ring]
    have h1 : 2 ≤ (L + Q) * Ae := by linarith
    have h2 : 8 ≤ (L + Q) * Ae * (iδ * is) := by nlinarith
    have h3 : 8 ≤ (L + Q) * Ae * (iδ * is) * ip := by nlinarith
    nlinarith [mul_le_mul_of_nonneg_left h3 (by linarith : 0 ≤ R * iξ)]
  have e2 : 2 * R * iξ = 2 * (R * iξ) := by ring
  have hξR : iξ ≤ R * iξ := le_mul_of_one_le_left (by linarith) hR
  exact reads_coords (ℓ := L + Q + 4) (Ae := Ae + Q) (ic := 2 * R * iξ) (iξ := 4 * iξ)
    (X := (L + Q) * Ae * iδ * R * iξ * ip * is) (by linarith) hQ (by linarith) (by linarith)
    (by linarith) (by linarith) (by linarith) hip (by linarith) (by linarith) (by linarith)
    (by linarith) (by linarith) hXδ (by linarith) (by linarith) hXp hXs

/-- The calls `learner_loop_dfa_dist_tol` bounds, at the claim's rates. -/
def distReads (r : LoopDistRun) : ℝ :=
  loopReads (r.Q + 1)
    (6 * (r.Q + 2) * loopCount (r.L + r.Q + 4) r.Q r.A (1 / r.β + r.Q) r.δ (distC r) (r.ξ / 4)
      r.pAP (1 / 2 - r.s₀))
    (loopPool (r.L + r.Q + 4) r.Q r.A (1 / r.β + r.Q) r.δ (distC r) (r.ξ / 4) r.pAP
      (1 / 2 - r.s₀) + 1)
    (loopBudget (r.L + r.Q + 4) r.Q r.A (1 / r.β + r.Q) r.δ (distC r))
    (loopSC (r.L + r.Q + 4) r.Q r.A (1 / r.β + r.Q) r.δ (distC r) (r.ξ / 4) r.pAP (1 / 2 - r.s₀))
    (loopHN (r.L + r.Q + 4) r.Q r.A (1 / r.β + r.Q) r.δ (distC r) (r.ξ / 4) r.pAP (1 / 2 - r.s₀))
    (r.L + r.Q + 4) r.A

lemma distReads_isBigO : distReads =O[Filter.principal loopDistRunRegion] (fun r =>
      ((r.L + r.Q) * r.R / (r.β * r.δ * r.ξ * r.pAP * r.s₀)) ^ 50) := by
  refine bigO_dist (2 ^ 200) (by norm_num) fun r hr => ?_
  obtain ⟨hL, hQ, hA, hAAe, hiδ, hR, hiξ, hip, his, -, -⟩ := dist_coords hr
  have h := rel_reads_le hL hQ hA hAAe hiδ hR hiξ hip his
  simp only [one_div_one_div, summed_inv, quarter_inv] at h
  exact ⟨h.1, h.2.trans_eq (by ring)⟩

end OrthoDFA

end
