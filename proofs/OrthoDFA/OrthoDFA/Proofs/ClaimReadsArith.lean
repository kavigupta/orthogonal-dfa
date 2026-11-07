import OrthoDFA.Proofs.ClaimLoopArith
import OrthoDFA.Proofs.LoopReadsArith

/-!
# The strings the loop asks, asymptotically: the arithmetic

Each factor of `learnerReads` is within a constant of a shape in the claim's coordinates: the
family size of `|V̂|` (`rel_fam_le`), `K` times the scan cap of `totalPref` (`rel_sc_le`), the
probes a round's harvested draws walk of `1/ρ` times that (`rel_hb_le`), the sifts of
`log (L + |Q| + 1)` (`lg_le`), and a sift's depth `|Q| + 1` and leaf bound `|Q| + 2` of `|Q|`. Each
source's reads is the product of its factors' bounds (`pw_mul`), so each is within its shape on its
own.
-/

noncomputable section

namespace OrthoDFA

open Real Asymptotics Filter

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

lemma loopHB_inv (ℓ Q A Ae iδ ic ξ pAP η₀ : ℝ) :
    loopHB ℓ Q A Ae (1 / iδ) (1 / ic) ξ pAP η₀
      = 384 * (4 * (Ae + 1) * ℓ * ic)
          * ((Q + 1) * loopSC ℓ Q A Ae (1 / iδ) (1 / ic) ξ pAP η₀ * log 2
            + log (8 * (Q + 1) * iδ)) + 1 := by
  unfold loopHB
  rw [div_inv_coord, one_div_div, div_inv_coord]


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

end Pow

private lemma pmul {a b x y : ℝ} (ha : 0 ≤ a) (hb : 0 ≤ b) (hx : a ≤ x) (hy : b ≤ y) :
    a * b ≤ x * y :=
  mul_le_mul hx hy hb (ha.trans hx)

private lemma four_le_X {L Q Ae iδ R iξ ip is : ℝ} (hL : 0 ≤ L) (hQ : 1 ≤ Q) (hAe : 2 ≤ Ae)
    (hiδ : 2 ≤ iδ) (hR : 1 ≤ R) (hiξ : 1 ≤ iξ) (hip : 1 ≤ ip) (his : 2 ≤ is) :
    4 * (Q * iδ) ≤ (L + Q) * Ae * iδ * R * iξ * ip * is := by
  have m := @one_le_mul_of_one_le_of_one_le ℝ _
  have k1 : 1 ≤ R * iξ * ip := m (m hR hiξ) hip
  have k2 : 2 * 2 ≤ Ae * is := pmul zero_le_two zero_le_two hAe his
  have n0 : 0 ≤ Q * iδ := mul_nonneg (by linarith) (by linarith)
  have k3 : Q * iδ ≤ (L + Q) * iδ := mul_le_mul_of_nonneg_right (by linarith) (by linarith)
  have k4 : Q * iδ * (2 * 2) ≤ (L + Q) * iδ * (Ae * is) := pmul n0 (by norm_num) k3 k2
  have n4 : 0 ≤ (L + Q) * iδ * (Ae * is) := (mul_nonneg n0 (by norm_num)).trans k4
  calc 4 * (Q * iδ) = Q * iδ * (2 * 2) := by ring
    _ ≤ (L + Q) * iδ * (Ae * is) := k4
    _ ≤ (L + Q) * iδ * (Ae * is) * (R * iξ * ip) := le_mul_of_one_le_right n4 k1
    _ = (L + Q) * Ae * iδ * R * iξ * ip * is := by ring

/-- The scan cap at `ℓ, |Q|, |Σ|, Ae, iδ, ic, iξ, ip, is ≤ X`. -/
lemma sc_coords {ℓ Q A Ae iδ ic iξ ip is X : ℝ} (hℓ : 1 ≤ ℓ) (hQ : 1 ≤ Q) (hA : 1 ≤ A)
    (hAe : 1 ≤ Ae) (hiδ : 1 ≤ iδ) (hic : 1 ≤ ic) (hiξ : 1 ≤ iξ) (hip : 1 ≤ ip) (his : 1 ≤ is)
    (hℓX : ℓ ≤ X) (hQX : Q ≤ X) (hAX : A ≤ X) (hAeX : Ae ≤ X) (hδX : iδ ≤ X) (hcX : ic ≤ X)
    (hξX : iξ ≤ X) (hpX : ip ≤ X) (hsX : is ≤ X) :
    1 ≤ loopSC ℓ Q A Ae (1 / iδ) (1 / ic) (1 / iξ) (1 / ip) (1 / 2 - 1 / is)
      ∧ loopSC ℓ Q A Ae (1 / iδ) (1 / ic) (1 / iξ) (1 / ip) (1 / 2 - 1 / is)
        ≤ 2 ^ 81 * X ^ 17 := by
  have hX : 1 ≤ X := hQ.trans hQX
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
  rw [loopSC_inv]
  exact sc_pow hX hQ hiδ hQX hδX hC.1 hC.2

/-- `sc_coords` at the summed rates, `X = (L + |Q|) Ae iδ R iξ ip is`. -/
lemma rel_sc_pow {L Q A Ae iδ R iξ ip is : ℝ} (hL : 0 ≤ L) (hQ : 1 ≤ Q) (hA : 2 ≤ A)
    (hAAe : A ≤ Ae) (hiδ : 2 ≤ iδ) (hR : 1 ≤ R) (hiξ : 1 ≤ iξ) (hip : 1 ≤ ip) (his : 2 ≤ is) :
    1 ≤ loopSC (L + Q + 4) Q A (Ae + Q) (1 / iδ) (1 / (2 * R * iξ)) (1 / (4 * iξ)) (1 / ip)
        (1 / 2 - 1 / is)
      ∧ loopSC (L + Q + 4) Q A (Ae + Q) (1 / iδ) (1 / (2 * R * iξ)) (1 / (4 * iξ)) (1 / ip)
        (1 / 2 - 1 / is) ≤ 2 ^ 81 * ((L + Q) * Ae * iδ * R * iξ * ip * is) ^ 17 := by
  have m := @one_le_mul_of_one_le_of_one_le ℝ _
  have hP : 1 ≤ L + Q := by linarith
  have hP0 : 0 ≤ L + Q := by linarith
  have hAe : 2 ≤ Ae := hA.trans hAAe
  have hRξ : 1 ≤ R * iξ := m hR hiξ
  have hV : 1 ≤ R * iξ * ip := m hRξ hip
  have hδs : 4 ≤ iδ * is := by nlinarith
  have hPAe : L + Q ≤ (L + Q) * Ae := by nlinarith
  have hAeP : Ae ≤ (L + Q) * Ae := by nlinarith
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
  have hξR : iξ ≤ R * iξ := le_mul_of_one_le_left (by linarith) hR
  exact sc_coords (ℓ := L + Q + 4) (Ae := Ae + Q) (ic := 2 * R * iξ) (iξ := 4 * iξ)
    (X := (L + Q) * Ae * iδ * R * iξ * ip * is) (by linarith) hQ (by linarith) (by linarith)
    (by linarith) (by linarith) (by linarith) hip (by linarith) (by linarith) (by linarith)
    (by linarith) (by linarith) hXδ (by linarith) (by linarith) hXp hXs

lemma loopFam_inv (ℓ Q A Ae iδ ic iξ is : ℝ) :
    loopFam ℓ Q A Ae (1 / iδ) (1 / ic) (1 / iξ) (1 / 2 - 1 / is)
      = (50 * (max (log (20 * loopΘ Q Ae (1 / ic) (1 / iξ) (1 / 2 - 1 / is)))
          (log (8 * loopW₀ Q A (1 / iδ) * (8 * (Q + 1) * iδ))) + 2 * log ℓ) * is ^ 2
          + 20 * is + 28) / 4 := by
  unfold loopFam
  dsimp only
  rw [sub_sub_cancel, show 8 * loopW₀ Q A (1 / iδ) / (1 / iδ / (8 * (Q + 1)))
    = 8 * loopW₀ Q A (1 / iδ) * (8 * (Q + 1) * iδ) by field_simp]
  field_simp

/-- The family size at the summed rates, a fourth of the suffix pool's first part. -/
lemma rel_fam_le {L Q A Ae iδ R iξ is : ℝ} (hL : 0 ≤ L) (hQ : 1 ≤ Q) (hA : 2 ≤ A)
    (hAAe : A ≤ Ae) (hiδ : 2 ≤ iδ) (hR : 1 ≤ R) (hiξ : 1 ≤ iξ) (his : 2 ≤ is) :
    0 ≤ loopFam (L + Q + 4) Q A (Ae + Q) (1 / iδ) (1 / (2 * R * iξ)) (1 / (4 * iξ))
        (1 / 2 - 1 / is)
      ∧ loopFam (L + Q + 4) Q A (Ae + Q) (1 / iδ) (1 / (2 * R * iξ)) (1 / (4 * iξ))
        (1 / 2 - 1 / is) ≤ 2 ^ 14 * (log ((L + Q) * Ae * iδ * R * iξ * is) * is ^ 2) := by
  have h := (rel_suff_le (ip := 1) hL hQ hA hAAe hiδ hR hiξ le_rfl his).2
  rw [loopPool_inv] at h
  rw [loopFam_inv]
  have hΘ := (loopΘ_le (A := Ae + Q) hQ (by linarith) (by nlinarith : 1 ≤ 2 * R * iξ)
    (by linarith : 1 ≤ 4 * iξ) (by linarith : 1 ≤ is)).1
  set mx := max (log (20 * loopΘ Q (Ae + Q) (1 / (2 * R * iξ)) (1 / (4 * iξ)) (1 / 2 - 1 / is)))
    (log (8 * loopW₀ Q A (1 / iδ) * (8 * (Q + 1) * iδ))) with hmx
  have hmx0 : 0 ≤ mx := le_max_of_le_left (Real.log_nonneg (by linarith))
  have hlℓ : 0 ≤ log (L + Q + 4) := Real.log_nonneg (by linarith)
  set F := 50 * (mx + 2 * log (L + Q + 4)) * is ^ 2 + 20 * is + 28 with hF
  have hF0 : 0 ≤ F := by
    have : 0 ≤ 50 * (mx + 2 * log (L + Q + 4)) * is ^ 2 :=
      mul_nonneg (mul_nonneg (by norm_num) (by linarith)) (sq_nonneg _)
    linarith
  refine ⟨by linarith, ?_⟩
  have hl48 : 0 ≤ log (48 * (Q + 1) * iδ) := Real.log_nonneg (by nlinarith)
  set lin := log ((L + Q) * Ae * iδ * R * iξ * is) with hlin
  have hQi : 0 < Q * iδ := by nlinarith
  have hQX : Q * iδ ≤ (L + Q) * Ae * iδ * R * iξ * is := by
    have m := @one_le_mul_of_one_le_of_one_le ℝ _
    have h1 : 1 ≤ Ae * R * iξ * is := m (m (m (by linarith) hR) hiξ) (by linarith)
    have h2 : Q * iδ ≤ (L + Q) * iδ := by nlinarith
    calc Q * iδ ≤ (L + Q) * iδ := h2
      _ ≤ (L + Q) * iδ * (Ae * R * iξ * is) := le_mul_of_one_le_right (by nlinarith) h1
      _ = (L + Q) * Ae * iδ * R * iξ * is := by ring
  have hlQ : log (Q * iδ) ≤ lin := Real.log_le_log hQi hQX
  have hlQ0 : 0 ≤ log (Q * iδ) := Real.log_nonneg (by nlinarith)
  have hs2 : 1 ≤ is ^ 2 := one_le_pow₀ (by linarith)
  have hlin : lin ≤ lin * is ^ 2 := le_mul_of_one_le_right (by linarith) hs2
  have hp : 0 ≤ (Q + 1) * (120 * log (48 * (Q + 1) * iδ) * 1 + 1) :=
    mul_nonneg (by linarith) (by linarith)
  have hQF : Q * F ≤ Q * (2 ^ 15 * (lin * is ^ 2 + log (Q * iδ))) := by
    nlinarith [h, hp, hF0]
  have hF' : F ≤ 2 ^ 15 * (lin * is ^ 2 + log (Q * iδ)) :=
    le_of_mul_le_mul_left hQF (by linarith)
  linarith

/-- The populations' prefix count times the scan cap, against `totalPref`'s shape. -/
lemma rel_sc_le {L Q A Ae iδ R iξ ip is : ℝ} (hL : 0 ≤ L) (hQ : 1 ≤ Q) (hA : 2 ≤ A)
    (hAAe : A ≤ Ae) (hiδ : 2 ≤ iδ) (hR : 1 ≤ R) (hiξ : 1 ≤ iξ) (hip : 1 ≤ ip) (his : 2 ≤ is) :
    0 ≤ (Q + 1) * loopSC (L + Q + 4) Q A (Ae + Q) (1 / iδ) (1 / (2 * R * iξ)) (1 / (4 * iξ))
        (1 / ip) (1 / 2 - 1 / is)
      ∧ (Q + 1) * loopSC (L + Q + 4) Q A (Ae + Q) (1 / iδ) (1 / (2 * R * iξ)) (1 / (4 * iξ))
        (1 / ip) (1 / 2 - 1 / is)
        ≤ 2 ^ 92 * (Q ^ 7 * (Ae + Q) ^ 2 * R ^ 2 * iξ ^ 2 * (L + Q) ^ 2 * is ^ 4
            * log (Q * iδ * ip * is * log ((L + Q) * Ae * iδ * R * iξ * is))
          + Q ^ 2 * (L + Q) ^ 2 * R ^ 2 * iξ ^ 2 * (is ^ 4 + Q ^ 4 * (Ae + Q) ^ 2)
            * log ((L + Q) * Ae * iδ * R * iξ * ip * is)) := by
  obtain ⟨hP0, hP⟩ := rel_pref_le hL hQ hA hAAe hiδ hR hiξ hip his
  have hM1 := (rel_sc_pow hL hQ hA hAAe hiδ hR hiξ hip his).1
  rw [loopSC_inv] at hM1 ⊢
  set C := loopCount (L + Q + 4) Q A (Ae + Q) (1 / iδ) (1 / (2 * R * iξ)) (1 / (4 * iξ)) (1 / ip)
    (1 / 2 - 1 / is)
  set lin := log ((L + Q) * Ae * iδ * R * iξ * is) with hlin
  set lY := log ((L + Q) * Ae * iδ * R * iξ * ip * is) with hlY
  set T1 := Q ^ 7 * (Ae + Q) ^ 2 * R ^ 2 * iξ ^ 2 * (L + Q) ^ 2 * is ^ 4
    * log (Q * iδ * ip * is * lin) with hT1
  set T2 := Q ^ 2 * (L + Q) ^ 2 * R ^ 2 * iξ ^ 2 * (is ^ 4 + Q ^ 4 * (Ae + Q) ^ 2) * lY with hT2
  have m := @one_le_mul_of_one_le_of_one_le ℝ _
  have hQ0 : 0 ≤ Q := by linarith
  have hC0 : 0 ≤ C := by
    have e : (Q + 1) * (6 * (Q + 2) * C) = (6 * (Q + 1) * (Q + 2)) * C := by ring
    exact nonneg_of_mul_nonneg_right (e ▸ hP0) (by positivity)
  have hP1 : 1 ≤ L + Q := by linarith
  have hAe : 2 ≤ Ae := hA.trans hAAe
  have z : (0 : ℝ) ≤ 1 := zero_le_one
  have t : (0 : ℝ) ≤ 2 := zero_le_two
  have hX4 := four_le_X hL hQ hAe hiδ hR hiξ hip his
  have hQδ : 2 ≤ Q * iδ := (show (2 : ℝ) = 1 * 2 by norm_num).trans_le (pmul z t hQ hiδ)
  have hX8 : 8 ≤ (L + Q) * Ae * iδ * R * iξ * ip * is := by linarith
  have hlY1 : 1 ≤ lY := by
    rw [hlY, Real.le_log_iff_exp_le (by linarith)]
    have := Real.exp_one_lt_d9
    linarith
  have hlin1 : 1 ≤ lin := by
    have h : 1 * 2 * 2 * 1 * 1 * 2 ≤ (L + Q) * Ae * iδ * R * iξ * is :=
      pmul (by norm_num) t (pmul (by norm_num) z (pmul (by norm_num) z
        (pmul (by norm_num) t (pmul z t hP1 hAe) hiδ) hR) hiξ) his
    rw [hlin, Real.le_log_iff_exp_le (by linarith)]
    have := Real.exp_one_lt_d9
    linarith
  have hT1n : 0 ≤ T1 := by
    have hlog : 0 ≤ log (Q * iδ * ip * is * lin) :=
      Real.log_nonneg (m (m (m (by linarith) hip) (by linarith)) hlin1)
    have h0 : 0 ≤ Q ^ 7 * (Ae + Q) ^ 2 * R ^ 2 * iξ ^ 2 * (L + Q) ^ 2 * is ^ 4 :=
      mul_nonneg (mul_nonneg (mul_nonneg (mul_nonneg (mul_nonneg (pow_nonneg hQ0 7)
        (sq_nonneg _)) (sq_nonneg _)) (sq_nonneg _)) (sq_nonneg _)) (by positivity)
    rw [hT1]; exact mul_nonneg h0 hlog
  -- the scan's log against `lY`
  have hlog : log (16 * (Q + 2) * (8 * (Q + 1) * iδ)) ≤ 4 * lY := by
    have hb : 16 * (Q + 2) * (8 * (Q + 1) * iδ) ≤ ((L + Q) * Ae * iδ * R * iξ * ip * is) ^ 4 := by
      have a : (Q + 2) * (Q + 1) ≤ 6 * Q ^ 2 := by
        linarith [mul_nonneg (by linarith : (0 : ℝ) ≤ 5 * Q + 2) (by linarith : (0 : ℝ) ≤ Q - 1)]
      have b : iδ ≤ iδ ^ 2 := by
        linarith [mul_nonneg (by linarith : (0 : ℝ) ≤ iδ) (by linarith : (0 : ℝ) ≤ iδ - 1)]
      have h1 : 16 * (Q + 2) * (8 * (Q + 1) * iδ) ≤ 768 * (Q * iδ) ^ 2 :=
        calc 16 * (Q + 2) * (8 * (Q + 1) * iδ) = 128 * ((Q + 2) * (Q + 1)) * iδ := by ring
          _ ≤ 128 * (6 * Q ^ 2) * iδ ^ 2 :=
            mul_le_mul (mul_le_mul_of_nonneg_left a (by norm_num)) b (by linarith) (by positivity)
          _ = 768 * (Q * iδ) ^ 2 := by ring
      have x2 : (2 : ℝ) ^ 2 ≤ (Q * iδ) ^ 2 := pow_le_pow_left₀ (by norm_num) hQδ 2
      have h2 : 768 * (Q * iδ) ^ 2 ≤ (4 * (Q * iδ)) ^ 4 :=
        calc 768 * (Q * iδ) ^ 2 ≤ 256 * ((Q * iδ) ^ 2 * (Q * iδ) ^ 2) := by
              linarith [mul_le_mul_of_nonneg_right x2 (sq_nonneg (Q * iδ)), sq_nonneg (Q * iδ)]
          _ = (4 * (Q * iδ)) ^ 4 := by ring
      have h3 : (4 * (Q * iδ)) ^ 4 ≤ ((L + Q) * Ae * iδ * R * iξ * ip * is) ^ 4 :=
        pow_le_pow_left₀ (by linarith) hX4 4
      linarith
    have hpos : 0 < 16 * (Q + 2) * (8 * (Q + 1) * iδ) :=
      mul_pos (mul_pos (by norm_num) (by linarith)) (mul_pos (mul_pos (by norm_num) (by linarith))
        (by linarith))
    calc log (16 * (Q + 2) * (8 * (Q + 1) * iδ))
        ≤ log (((L + Q) * Ae * iδ * R * iξ * ip * is) ^ 4) := Real.log_le_log hpos hb
      _ = 4 * lY := by rw [Real.log_pow]; push_cast; rfl
  have hQlY : Q * lY ≤ T2 := by
    have h1 : 1 ≤ Q * (L + Q) ^ 2 * R ^ 2 * iξ ^ 2 * (is ^ 4 + Q ^ 4 * (Ae + Q) ^ 2) := by
      have a1 : 1 ≤ (L + Q) ^ 2 := one_le_pow₀ hP1
      have a2 : 1 ≤ R ^ 2 := one_le_pow₀ hR
      have a3 : 1 ≤ iξ ^ 2 := one_le_pow₀ hiξ
      have a4 : 1 ≤ is ^ 4 + Q ^ 4 * (Ae + Q) ^ 2 := by
        have : 1 ≤ is ^ 4 := one_le_pow₀ (by linarith)
        have : 0 ≤ Q ^ 4 * (Ae + Q) ^ 2 := by positivity
        linarith
      exact m (m (m (m hQ a1) a2) a3) a4
    have e : T2 = Q * lY * (Q * (L + Q) ^ 2 * R ^ 2 * iξ ^ 2 * (is ^ 4 + Q ^ 4 * (Ae + Q) ^ 2)) :=
      by rw [hT2]; ring
    rw [e]
    exact le_mul_of_one_le_right (mul_nonneg hQ0 (by linarith)) h1
  have hC8 : (Q + 1) * (8 * (Q + 4) * C) ≤ 8 / 3 * ((Q + 1) * (6 * (Q + 2) * C)) := by
    have : 0 ≤ (Q + 1) * C := mul_nonneg (by linarith) hC0
    linarith [mul_nonneg hQ0 this]
  refine ⟨mul_nonneg (by linarith) (by linarith), ?_⟩
  have hsum : (Q + 1) * (8 * (Q + 4) * C + 16 * log (16 * (Q + 2) * (8 * (Q + 1) * iδ)) + 2)
      = (Q + 1) * (8 * (Q + 4) * C)
        + (Q + 1) * (16 * log (16 * (Q + 2) * (8 * (Q + 1) * iδ)) + 2) := by ring
  have hrest : (Q + 1) * (16 * log (16 * (Q + 2) * (8 * (Q + 1) * iδ)) + 2) ≤ 132 * T2 := by
    have h1 : 16 * log (16 * (Q + 2) * (8 * (Q + 1) * iδ)) + 2 ≤ 66 * lY := by linarith
    have h2 : (Q + 1) * (16 * log (16 * (Q + 2) * (8 * (Q + 1) * iδ)) + 2)
        ≤ (Q + 1) * (66 * lY) := mul_le_mul_of_nonneg_left h1 (by linarith)
    have h3 : (Q + 1) * (66 * lY) ≤ 132 * (Q * lY) := by
      linarith [mul_nonneg (by linarith : (0 : ℝ) ≤ Q - 1) (by linarith : (0 : ℝ) ≤ lY)]
    linarith
  rw [hsum]
  have hT2n : 0 ≤ T2 := le_trans (mul_nonneg hQ0 (by linarith)) hQlY
  linarith

/-- The probes a round's harvested draws walk at the summed rates, the scan caps' total at most
`2^92 P`: `1/ε` times `P`. -/
lemma rel_hb_le {L Q A Ae iδ R iξ ip is P : ℝ} (hL : 0 ≤ L) (hQ : 1 ≤ Q) (hA : 2 ≤ A)
    (hAAe : A ≤ Ae) (hiδ : 2 ≤ iδ) (hR : 1 ≤ R) (hiξ : 1 ≤ iξ) (hip : 1 ≤ ip) (his : 2 ≤ is)
    (hP : (Q + 1) * loopSC (L + Q + 4) Q A (Ae + Q) (1 / iδ) (1 / (2 * R * iξ)) (1 / (4 * iξ))
        (1 / ip) (1 / 2 - 1 / is) ≤ 2 ^ 92 * P) :
    1 ≤ loopHB (L + Q + 4) Q A (Ae + Q) (1 / iδ) (1 / (2 * R * iξ)) (1 / (4 * iξ)) (1 / ip)
        (1 / 2 - 1 / is)
      ∧ loopHB (L + Q + 4) Q A (Ae + Q) (1 / iδ) (1 / (2 * R * iξ)) (1 / (4 * iξ)) (1 / ip)
        (1 / 2 - 1 / is) ≤ 2 ^ 108 * (R * (Ae + Q) * (L + Q) * iξ * P) := by
  obtain ⟨hM1, -⟩ := rel_sc_pow hL hQ hA hAAe hiδ hR hiξ hip his
  obtain ⟨hP0, -⟩ := rel_pref_le hL hQ hA hAAe hiδ hR hiξ hip his
  have hMe := loopSC_inv (L + Q + 4) Q A (Ae + Q) iδ (1 / (2 * R * iξ)) (1 / (4 * iξ)) (1 / ip)
    (1 / 2 - 1 / is)
  rw [loopHB_inv]
  set M := loopSC (L + Q + 4) Q A (Ae + Q) (1 / iδ) (1 / (2 * R * iξ)) (1 / (4 * iξ)) (1 / ip)
    (1 / 2 - 1 / is)
  set C := loopCount (L + Q + 4) Q A (Ae + Q) (1 / iδ) (1 / (2 * R * iξ)) (1 / (4 * iξ)) (1 / ip)
    (1 / 2 - 1 / is)
  have m := @one_le_mul_of_one_le_of_one_le ℝ _
  have hQ0 : 0 ≤ Q := by linarith
  have hC0 : 0 ≤ C := by
    have e : (Q + 1) * (6 * (Q + 2) * C) = (6 * (Q + 1) * (Q + 2)) * C := by ring
    exact nonneg_of_mul_nonneg_right (e ▸ hP0) (by positivity)
  have hy : 1 ≤ 8 * (Q + 1) * iδ := m (by linarith) (by linarith)
  have hlD : 0 ≤ log (8 * (Q + 1) * iδ) := Real.log_nonneg hy
  -- the walk tail's log is within the scan cap's
  have hlDM : log (8 * (Q + 1) * iδ) ≤ M := by
    have h1 : log (8 * (Q + 1) * iδ) ≤ log (16 * (Q + 2) * (8 * (Q + 1) * iδ)) :=
      Real.log_le_log (by linarith) (le_mul_of_one_le_left (by linarith) (by linarith))
    have h2 : 0 ≤ 8 * (Q + 4) * C := mul_nonneg (by linarith) hC0
    rw [hMe]
    linarith
  have hl2 : log 2 ≤ 1 := by have := Real.log_two_lt_d9; linarith
  have hl20 : 0 ≤ log 2 := Real.log_nonneg (by norm_num)
  set Y := (Q + 1) * M with hYd
  have hMY : M ≤ Y := le_mul_of_one_le_left (by linarith) (by linarith)
  have hY1 : 1 ≤ Y := m (by linarith) hM1
  set S := Y * log 2 + log (8 * (Q + 1) * iδ) with hSd
  have hS : S ≤ 2 * Y := by
    have := mul_le_mul_of_nonneg_left hl2 (by linarith : 0 ≤ Y)
    linarith
  have hS0 : 0 ≤ S := add_nonneg (mul_nonneg (by linarith) hl20) hlD
  have hRi : 0 ≤ R * iξ := mul_nonneg (by linarith) (by linarith)
  set ρc := 4 * (Ae + Q + 1) * (L + Q + 4) * (2 * R * iξ) with hρcd
  set ρ := R * (Ae + Q) * (L + Q) * iξ with hρd
  have hρ : ρc ≤ 80 * ρ := by
    have h1 : Ae + Q + 1 ≤ 2 * (Ae + Q) := by linarith
    have h2 : L + Q + 4 ≤ 5 * (L + Q) := by linarith
    have h3 : (Ae + Q + 1) * (L + Q + 4) ≤ 2 * (Ae + Q) * (5 * (L + Q)) :=
      mul_le_mul h1 h2 (by linarith) (by linarith)
    have h4 := mul_le_mul_of_nonneg_right h3 hRi
    rw [hρcd, hρd]
    linarith
  have hρ0 : 0 ≤ ρc := by
    have := mul_nonneg (mul_nonneg (by linarith : (0 : ℝ) ≤ Ae + Q + 1)
      (by linarith : (0 : ℝ) ≤ L + Q + 4)) hRi
    rw [hρcd]
    linarith
  have hρ1 : 1 ≤ ρ := m (m (m hR (by linarith)) (by linarith)) hiξ
  have h1 : 384 * ρc * S ≤ 384 * (80 * ρ) * (2 * Y) :=
    mul_le_mul (mul_le_mul_of_nonneg_left hρ (by norm_num)) hS hS0 (by linarith)
  have hρY : 1 ≤ ρ * Y := m hρ1 hY1
  have h2 : ρ * Y ≤ ρ * (2 ^ 92 * P) := mul_le_mul_of_nonneg_left hP (by linarith)
  have h0 := mul_nonneg (mul_nonneg (by norm_num : (0 : ℝ) ≤ 384) hρ0) hS0
  refine ⟨by linarith, ?_⟩
  linarith

/-- `3 + ⌈log₂ (L + |Q| + 3)⌉` against `log (L + |Q| + 1)`. -/
lemma lg_le {L Q : ℕ} (hQ : 1 ≤ Q) :
    ((3 + Nat.clog 2 (L + Q + 3) : ℕ) : ℝ) ≤ 14 * log ((L : ℝ) + Q + 1) := by
  have hm : 1 < L + Q + 3 := by omega
  have h1 := Nat.pow_pred_clog_lt_self (by norm_num : 1 < 2) hm
  set c := Nat.clog 2 (L + Q + 3)
  have hc : c ≤ (c - 1) + 1 := by omega
  have hc' : (c : ℝ) ≤ ((c - 1 : ℕ) : ℝ) + 1 := by exact_mod_cast hc
  have h2 : (2 : ℝ) ^ (c - 1) < ((L + Q + 3 : ℕ) : ℝ) := by
    rw [Nat.pred_eq_sub_one] at h1
    exact_mod_cast h1
  have h3 : ((c - 1 : ℕ) : ℝ) * log 2 < log ((L + Q + 3 : ℕ) : ℝ) := by
    rw [← Real.log_pow]
    exact Real.log_lt_log (by positivity) h2
  have hQ' : (1 : ℝ) ≤ Q := by exact_mod_cast hQ
  have hL' : (0 : ℝ) ≤ L := Nat.cast_nonneg L
  set lg := log ((L : ℝ) + Q + 1) with hlgd
  have hlg2 : log 2 ≤ lg := Real.log_le_log (by norm_num) (by linarith)
  have hm' : log ((L + Q + 3 : ℕ) : ℝ) ≤ 2 * log 2 + lg := by
    have e : ((L + Q + 3 : ℕ) : ℝ) = (L : ℝ) + Q + 3 := by push_cast; ring
    rw [e]
    have : (L : ℝ) + Q + 3 ≤ 2 ^ 2 * ((L : ℝ) + Q + 1) := by linarith
    calc log ((L : ℝ) + Q + 3) ≤ log (2 ^ 2 * ((L : ℝ) + Q + 1)) :=
        Real.log_le_log (by linarith) this
      _ = 2 * log 2 + lg := by
        rw [Real.log_mul (by norm_num) (by linarith), Real.log_pow]; push_cast; rfl
  have hl2 := Real.log_two_gt_d9
  have hc0 : (0 : ℝ) ≤ c := Nat.cast_nonneg _
  have h4 : (c : ℝ) * log 2 ≤ (((c - 1 : ℕ) : ℝ) + 1) * log 2 :=
    mul_le_mul_of_nonneg_right hc' (by linarith)
  have h5 : (c : ℝ) * 0.6931471803 ≤ c * log 2 := mul_le_mul_of_nonneg_left hl2.le hc0
  push_cast
  linarith

/-- `f ≤ K g`, `f` nonnegative: the form each source's bound takes. -/
lemma pw_mul {f₁ g₁ f₂ g₂ K₁ K₂ : ℝ} (h₁ : 0 ≤ f₁ ∧ f₁ ≤ K₁ * g₁) (h₂ : 0 ≤ f₂ ∧ f₂ ≤ K₂ * g₂) :
    0 ≤ f₁ * f₂ ∧ f₁ * f₂ ≤ K₁ * K₂ * (g₁ * g₂) :=
  ⟨mul_nonneg h₁.1 h₂.1, (mul_le_mul h₁.2 h₂.2 h₂.1 (h₁.1.trans h₁.2)).trans_eq (by ring)⟩

lemma pw_add {f₁ g₁ f₂ g₂ K₁ K₂ : ℝ} (hK₁ : 0 < K₁) (hK₂ : 0 < K₂)
    (h₁ : 0 ≤ f₁ ∧ f₁ ≤ K₁ * g₁) (h₂ : 0 ≤ f₂ ∧ f₂ ≤ K₂ * g₂) :
    0 ≤ f₁ + f₂ ∧ f₁ + f₂ ≤ (K₁ + K₂) * (g₁ + g₂) := by
  have hg₁ := nonneg_of_mul_nonneg_right (h₁.1.trans h₁.2) hK₁
  have hg₂ := nonneg_of_mul_nonneg_right (h₂.1.trans h₂.2) hK₂
  refine ⟨add_nonneg h₁.1 h₂.1, ?_⟩
  nlinarith [h₁.2, h₂.2, mul_nonneg hK₂.le hg₁, mul_nonneg hK₁.le hg₂]

/-! ## The claim's shapes -/

def shPref (r : LoopDistRun) : ℝ :=
  r.Q ^ 7 * (1 / r.β + r.Q) ^ 2 * (r.L + r.Q) ^ 2 * r.R ^ 2 / (r.ξ ^ 2 * r.s₀ ^ 4)
      * log (r.Q * log ((r.L + r.Q) * r.R / (r.β * r.δ * r.ξ * r.s₀)) / (r.δ * r.pAP * r.s₀))
    + r.Q ^ 2 * (r.L + r.Q) ^ 2 * r.R ^ 2 / r.ξ ^ 2
      * (1 / r.s₀ ^ 4 + r.Q ^ 4 * (1 / r.β + r.Q) ^ 2)
      * log ((r.L + r.Q) * r.R / (r.β * r.δ * r.ξ * r.pAP * r.s₀))

def shSuff (r : LoopDistRun) : ℝ :=
  r.Q * (log ((r.L + r.Q) * r.R / (r.β * r.δ * r.ξ * r.s₀)) / (r.pAP * r.s₀ ^ 2)
    + log (r.Q / r.δ) / r.pAP)

def shProbe (r : LoopDistRun) : ℝ :=
  r.Q ^ 3 * r.A * (1 / r.β + r.Q) * (r.L + r.Q) * r.R / r.ξ * log (r.Q * r.A / r.δ)

def shHeld (r : LoopDistRun) : ℝ := log (r.Q / r.δ) / r.ξ ^ 2

/-- The family size `|V̂|`. -/
def shFam (r : LoopDistRun) : ℝ := log ((r.L + r.Q) * r.R / (r.β * r.δ * r.ξ * r.s₀)) / r.s₀ ^ 2

/-- `1/ρ`, the disagreement rate's inverse. -/
def shRho (r : LoopDistRun) : ℝ := r.R * (1 / r.β + r.Q) * (r.L + r.Q) / r.ξ

def shLg (r : LoopDistRun) : ℝ := log (r.L + r.Q + 1)

def shCluster (r : LoopDistRun) : ℝ := shPref r * shSuff r / r.Q

def shSearch (r : LoopDistRun) : ℝ := shProbe r * shFam r * r.Q * shLg r

def shResift (r : LoopDistRun) : ℝ := r.Q ^ 4 * r.A * shFam r

def shHarvest (r : LoopDistRun) : ℝ := r.Q ^ 2 * shPref r * shFam r * shLg r * shRho r

def shStart (r : LoopDistRun) : ℝ := shHeld r * shFam r

/-! ## The learner's reads at the claim's rates -/

def distCount (r : LoopDistRun) : ℝ :=
  loopCount (r.L + r.Q + 4) r.Q r.A (1 / r.β + r.Q) r.δ (distC r) (r.ξ / 4) r.pAP (1 / 2 - r.s₀)

def distPool (r : LoopDistRun) : ℝ :=
  loopPool (r.L + r.Q + 4) r.Q r.A (1 / r.β + r.Q) r.δ (distC r) (r.ξ / 4) r.pAP (1 / 2 - r.s₀)

def distFam (r : LoopDistRun) : ℝ :=
  loopFam (r.L + r.Q + 4) r.Q r.A (1 / r.β + r.Q) r.δ (distC r) (r.ξ / 4) (1 / 2 - r.s₀)

def distBudget (r : LoopDistRun) : ℝ :=
  loopBudget (r.L + r.Q + 4) r.Q r.A (1 / r.β + r.Q) r.δ (distC r)

def distSC (r : LoopDistRun) : ℝ :=
  loopSC (r.L + r.Q + 4) r.Q r.A (1 / r.β + r.Q) r.δ (distC r) (r.ξ / 4) r.pAP (1 / 2 - r.s₀)

/-- The probes a round's harvested draws walk, at most, at the claim's rates. -/
def distHB (r : LoopDistRun) : ℝ :=
  loopHB (r.L + r.Q + 4) r.Q r.A (1 / r.β + r.Q) r.δ (distC r) (r.ξ / 4) r.pAP (1 / 2 - r.s₀)

def distLg (r : LoopDistRun) : ℝ := ((3 + Nat.clog 2 (r.L + r.Q + 3) : ℕ) : ℝ)

/-- The strings `learner_loop_dfa_dist_tol` bounds the learner's asking by, at the claim's rates. -/
def distReads (r : LoopDistRun) : ℝ :=
  learnerReads (r.Q + 1) (6 * (r.Q + 2) * distCount r) (distPool r + 1) (distFam r)
    (distBudget r) r.Q (⌈distHB r⌉₊ : ℝ) (distLg r) r.A (distHeldOut r.toLoopDistInputs)

section Pointwise

variable {r : LoopDistRun} (hr : r ∈ loopDistRunRegion)
include hr

lemma pref_pw : 0 ≤ distPref r ∧ distPref r ≤ 2 ^ 90 * shPref r := by
  obtain ⟨hL, hQ, hA, hAAe, hiδ, hR, hiξ, hip, his, -, -⟩ := dist_coords hr
  have h := rel_pref_le hL hQ hA hAAe hiδ hR hiξ hip his
  simp only [one_div_one_div, summed_inv, quarter_inv] at h
  exact ⟨h.1, h.2.trans_eq (by unfold shPref; ring_nf)⟩

lemma suff_pw : 0 ≤ distSuff r ∧ distSuff r ≤ 2 ^ 15 * shSuff r := by
  obtain ⟨hL, hQ, hA, hAAe, hiδ, hR, hiξ, hip, his, -, -⟩ := dist_coords hr
  have h := rel_suff_le hL hQ hA hAAe hiδ hR hiξ hip his
  simp only [one_div_one_div, summed_inv, quarter_inv] at h
  exact ⟨h.1, h.2.trans_eq (by unfold shSuff; ring_nf)⟩

lemma probes_pw : 0 ≤ distProbes r ∧ distProbes r ≤ 2 ^ 20 * shProbe r := by
  obtain ⟨hL, hQ, hA, hAAe, hiδ, hR, hiξ, -, -, -, -⟩ := dist_coords hr
  have h := rel_probes_le hL hQ hA hAAe hiδ hR hiξ
  simp only [one_div_one_div, summed_inv] at h
  exact ⟨h.1, h.2.trans_eq (by unfold shProbe; ring_nf)⟩

lemma held_pw :
    0 ≤ (distHeldOut r.toLoopDistInputs : ℝ) ∧ (distHeldOut r.toLoopDistInputs : ℝ)
      ≤ 2 ^ 13 * shHeld r := by
  obtain ⟨⟨hQ, -, hδ, hδ2, -, hξ, hξ1, -, -, -, -⟩, -, -⟩ := hr
  have hQ' : (1 : ℝ) ≤ r.Q := by exact_mod_cast hQ
  have hiδ : 2 ≤ 1 / r.δ := (le_div_iff₀ hδ).2 (by linarith)
  have hiξ : 1 ≤ 1 / r.ξ := (le_div_iff₀ hξ).2 (by linarith)
  have h := claim_heldOut_le hQ' hiδ (by linarith : 1 ≤ 4 * (1 / r.ξ))
  rw [one_div_one_div, quarter_inv] at h
  refine ⟨Nat.cast_nonneg _, h.trans_eq ?_⟩
  unfold shHeld
  field_simp
  ring

lemma fam_pw : 0 ≤ distFam r ∧ distFam r ≤ 2 ^ 14 * shFam r := by
  obtain ⟨hL, hQ, hA, hAAe, hiδ, hR, hiξ, -, his, -, -⟩ := dist_coords hr
  have h := rel_fam_le hL hQ hA hAAe hiδ hR hiξ his
  simp only [one_div_one_div, summed_inv, quarter_inv] at h
  exact ⟨h.1, h.2.trans_eq (by unfold shFam; ring_nf)⟩

lemma sc_pw : 0 ≤ (r.Q + 1) * distSC r ∧ (r.Q + 1) * distSC r ≤ 2 ^ 92 * shPref r := by
  obtain ⟨hL, hQ, hA, hAAe, hiδ, hR, hiξ, hip, his, -, -⟩ := dist_coords hr
  have h := rel_sc_le hL hQ hA hAAe hiδ hR hiξ hip his
  simp only [one_div_one_div, summed_inv, quarter_inv] at h
  exact ⟨h.1, h.2.trans_eq (by unfold shPref; ring_nf)⟩

lemma hb_pw : 1 ≤ distHB r ∧ distHB r ≤ 2 ^ 108 * (shRho r * shPref r) := by
  obtain ⟨hL, hQ, hA, hAAe, hiδ, hR, hiξ, hip, his, -, -⟩ := dist_coords hr
  have h := rel_hb_le hL hQ hA hAAe hiδ hR hiξ hip his
    (rel_sc_le hL hQ hA hAAe hiδ hR hiξ hip his).2
  simp only [one_div_one_div, summed_inv, quarter_inv] at h
  exact ⟨h.1, h.2.trans_eq (by unfold shRho shPref; ring_nf)⟩

lemma hbc_pw :
    0 ≤ (⌈distHB r⌉₊ : ℝ) ∧ (⌈distHB r⌉₊ : ℝ) ≤ 2 ^ 109 * (shRho r * shPref r) := by
  obtain ⟨h1, h2⟩ := hb_pw hr
  have := Nat.ceil_lt_add_one (show 0 ≤ distHB r by linarith)
  refine ⟨Nat.cast_nonneg _, ?_⟩
  linarith

lemma lg_pw : 0 ≤ distLg r ∧ distLg r ≤ 14 * shLg r :=
  ⟨Nat.cast_nonneg _, lg_le hr.1.1⟩

lemma k_pw : 0 ≤ (r.Q : ℝ) + 1 ∧ (r.Q : ℝ) + 1 ≤ 2 * r.Q := by
  have hQ : (1 : ℝ) ≤ r.Q := by exact_mod_cast hr.1.1
  constructor <;> linarith

lemma leaves_pw : 0 ≤ (r.Q : ℝ) + 2 ∧ (r.Q : ℝ) + 2 ≤ 3 * r.Q := by
  have hQ : (1 : ℝ) ≤ r.Q := by exact_mod_cast hr.1.1
  constructor <;> linarith

lemma pool_pw : 0 ≤ distPool r + 2 ∧ distPool r + 2 ≤ 2 ^ 16 * (shSuff r / r.Q) := by
  obtain ⟨hL, hQ, hA, hAAe, hiδ, hR, hiξ, hip, his, -, -⟩ := dist_coords hr
  have hRξ : 1 * 1 ≤ r.R * (1 / r.ξ) := mul_le_mul hR hiξ zero_le_one (by linarith)
  have hP1 := loopPool_ge (A := r.A) (ℓ := r.L + r.Q + 4) (Ae := 1 / r.β + r.Q) (iδ := 1 / r.δ)
    (ic := 2 * r.R * (1 / r.ξ)) (iξ := 4 * (1 / r.ξ)) (is := 1 / r.s₀) (by linarith) hQ
    (by linarith) (by linarith) (by linarith) (by linarith) hip (by linarith)
  simp only [one_div_one_div, summed_inv, quarter_inv] at hP1
  obtain ⟨h0, h⟩ := suff_pw hr
  have hP1' : 1 ≤ distPool r := hP1
  have hQ0 : (0 : ℝ) < r.Q := by linarith
  have e : distSuff r = (r.Q + 1) * (distPool r + 1) := rfl
  refine ⟨by linarith, ?_⟩
  rw [mul_div_assoc', le_div_iff₀ hQ0]
  rw [e] at h
  nlinarith [mul_nonneg (by linarith : (0 : ℝ) ≤ distPool r) hQ0.le]

end Pointwise

lemma cluster_pw {r : LoopDistRun} (hr : r ∈ loopDistRunRegion) :
    0 ≤ readsCluster (r.Q + 1) (6 * (r.Q + 2) * distCount r) (distPool r + 1)
      ∧ readsCluster (r.Q + 1) (6 * (r.Q + 2) * distCount r) (distPool r + 1)
        ≤ 2 ^ 90 * 2 ^ 16 * shCluster r := by
  have e : readsCluster (r.Q + 1) (6 * (r.Q + 2) * distCount r) (distPool r + 1)
      = distPref r * (distPool r + 2) := by
    unfold readsCluster distPref distCount distPool; ring
  rw [e, show shCluster r = shPref r * (shSuff r / r.Q) by unfold shCluster; ring]
  exact pw_mul (pref_pw hr) (pool_pw hr)

lemma search_pw {r : LoopDistRun} (hr : r ∈ loopDistRunRegion) :
    0 ≤ readsSearch (r.Q + 1) (distFam r) (distBudget r) r.Q (distLg r)
      ∧ readsSearch (r.Q + 1) (distFam r) (distBudget r) r.Q (distLg r)
        ≤ 2 ^ 20 * (2 ^ 14 * (14 * 2)) * shSearch r := by
  have e : readsSearch (r.Q + 1) (distFam r) (distBudget r) r.Q (distLg r)
      = distProbes r * (distFam r * (distLg r * (r.Q + 1))) := by
    unfold readsSearch distProbes distBudget; ring
  rw [e, show shSearch r = shProbe r * (shFam r * (shLg r * r.Q)) by unfold shSearch; ring]
  exact pw_mul (probes_pw hr) (pw_mul (fam_pw hr) (pw_mul (lg_pw hr) (k_pw hr)))

lemma resift_pw {r : LoopDistRun} (hr : r ∈ loopDistRunRegion) :
    0 ≤ readsResift (r.Q + 1) (distFam r) r.Q r.A
      ∧ readsResift (r.Q + 1) (distFam r) r.Q r.A
        ≤ 2 * (2 ^ 14 * (1 * (3 * 1 * 3))) * shResift r := by
  have hA : 0 ≤ (r.A : ℝ) ∧ (r.A : ℝ) ≤ 1 * r.A := ⟨Nat.cast_nonneg _, by linarith⟩
  have hQ : 0 ≤ (r.Q : ℝ) ∧ (r.Q : ℝ) ≤ 1 * r.Q := ⟨Nat.cast_nonneg _, by linarith⟩
  rw [show shResift r = r.Q * (shFam r * (r.Q * (r.Q * r.A * r.Q))) by unfold shResift; ring]
  exact pw_mul (k_pw hr) (pw_mul (fam_pw hr) (pw_mul hQ
    (pw_mul (pw_mul (leaves_pw hr) hA) (leaves_pw hr))))

lemma harvest_pw {r : LoopDistRun} (hr : r ∈ loopDistRunRegion) :
    0 ≤ readsHarvest (r.Q + 1) (distFam r) r.Q (⌈distHB r⌉₊ : ℝ) (distLg r)
      ∧ readsHarvest (r.Q + 1) (distFam r) r.Q (⌈distHB r⌉₊ : ℝ) (distLg r)
        ≤ 2 * (2 ^ 14 * (2 ^ 109 * (14 * 2))) * shHarvest r := by
  rw [show shHarvest r = r.Q * (shFam r * ((shRho r * shPref r) * (shLg r * r.Q))) by
    unfold shHarvest; ring]
  exact pw_mul (k_pw hr) (pw_mul (fam_pw hr) (pw_mul (hbc_pw hr) (pw_mul (lg_pw hr) (k_pw hr))))

lemma start_pw {r : LoopDistRun} (hr : r ∈ loopDistRunRegion) :
    0 ≤ readsStart (distHeldOut r.toLoopDistInputs) (distFam r)
      ∧ readsStart (distHeldOut r.toLoopDistInputs) (distFam r) ≤ 2 ^ 13 * 2 ^ 14 * shStart r :=
  pw_mul (held_pw hr) (fam_pw hr)

lemma shapes_nonneg {r : LoopDistRun} (hr : r ∈ loopDistRunRegion) :
    0 ≤ shCluster r ∧ 0 ≤ shSearch r ∧ 0 ≤ shResift r ∧ 0 ≤ shHarvest r ∧ 0 ≤ shStart r :=
  ⟨nonneg_of_mul_nonneg_right ((cluster_pw hr).1.trans (cluster_pw hr).2) (by norm_num),
    nonneg_of_mul_nonneg_right ((search_pw hr).1.trans (search_pw hr).2) (by norm_num),
    nonneg_of_mul_nonneg_right ((resift_pw hr).1.trans (resift_pw hr).2) (by norm_num),
    nonneg_of_mul_nonneg_right ((harvest_pw hr).1.trans (harvest_pw hr).2) (by norm_num),
    nonneg_of_mul_nonneg_right ((start_pw hr).1.trans (start_pw hr).2) (by norm_num)⟩

lemma distHB_isBigO : distHB =O[𝓟 loopDistRunRegion] (fun r => shRho r * shPref r) :=
  isBigO_on (2 ^ 108) fun r hr => by
    obtain ⟨h1, h2⟩ := hb_pw hr
    have h0 : 0 ≤ distHB r := by linarith
    exact ⟨h0, nonneg_of_mul_nonneg_right (h0.trans h2) (by norm_num), h2⟩

/-- Each source of reads within its own shape. -/
lemma distReads_isBigO : distReads =O[𝓟 loopDistRunRegion]
    (fun r => shCluster r + shSearch r + shResift r + shHarvest r + shStart r) := by
  have hg : ∀ f : LoopDistRun → ℝ, (∀ r ∈ loopDistRunRegion, 0 ≤ f r ∧
      f r ≤ shCluster r + shSearch r + shResift r + shHarvest r + shStart r) →
      f =O[𝓟 loopDistRunRegion]
        (fun r => shCluster r + shSearch r + shResift r + shHarvest r + shStart r) :=
    fun f h => isBigO_on 1 fun r hr => ⟨(h r hr).1, (h r hr).1.trans (h r hr).2,
      (h r hr).2.trans_eq (one_mul _).symm⟩
  have t1 := (bigO_dist _ (by norm_num) fun _ hr => cluster_pw hr).trans
    (hg shCluster fun r hr => by
      obtain ⟨h1, h2, h3, h4, h5⟩ := shapes_nonneg hr; exact ⟨h1, by linarith⟩)
  have t2 := (bigO_dist _ (by norm_num) fun _ hr => search_pw hr).trans
    (hg shSearch fun r hr => by
      obtain ⟨h1, h2, h3, h4, h5⟩ := shapes_nonneg hr; exact ⟨h2, by linarith⟩)
  have t3 := (bigO_dist _ (by norm_num) fun _ hr => resift_pw hr).trans
    (hg shResift fun r hr => by
      obtain ⟨h1, h2, h3, h4, h5⟩ := shapes_nonneg hr; exact ⟨h3, by linarith⟩)
  have t4 := (bigO_dist _ (by norm_num) fun _ hr => harvest_pw hr).trans
    (hg shHarvest fun r hr => by
      obtain ⟨h1, h2, h3, h4, h5⟩ := shapes_nonneg hr; exact ⟨h4, by linarith⟩)
  have t5 := (bigO_dist _ (by norm_num) fun _ hr => start_pw hr).trans
    (hg shStart fun r hr => by
      obtain ⟨h1, h2, h3, h4, h5⟩ := shapes_nonneg hr; exact ⟨h5, by linarith⟩)
  exact (((t1.add t2).add t3).add t4).add t5

end OrthoDFA

end

