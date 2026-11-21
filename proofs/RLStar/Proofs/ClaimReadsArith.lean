import RLStar.Proofs.ClaimRLStarArith

/-!
# The trie loop's tolerance and reads against the claim's shapes

`ClaimRLStarArith` continued on `TC`: `trieTolInv` within `2¹³⁸` of `shTol` (`tc_tol`), the reads
within `2¹¹²` of `shReads` (`tc_reads`), `shTol` below `2⁹ (|Σ| V)¹⁸`, `V = X log(1/δ)/(ξps)`,
for the uniform claims' `k₀` (`tc_tolPow`), and the simple claim's shapes (`tc_simple`).
-/

noncomputable section

namespace OrthoDFA

open Real


namespace TC

variable {Q A X δ ξ p s : ℝ} (h : TC Q A X δ ξ p s)
include h

/-- `Z = X|Σ|/(δξps)` against the atoms under it. -/
lemma Z_facts : 8 ≤ X * A / (δ * ξ * p * s) ∧ Q ≤ X * A / (δ * ξ * p * s)
    ∧ 1 / δ ≤ X * A / (δ * ξ * p * s) ∧ 1 / s ≤ X * A / (δ * ξ * p * s)
    ∧ 1 / p ≤ X * A / (δ * ξ * p * s) ∧ shι Q X ξ ≤ 3 * (X * A / (δ * ξ * p * s)) ^ 3
    ∧ shZ A X δ ξ p s ≤ X * A / (δ * ξ * p * s) := by
  obtain ⟨-, -, hXY, -, hδY, -, hsY, hY8, -⟩ := h.Y_ge
  obtain ⟨hYZ, hpZ⟩ := h.Z_ge
  have hY0 : 0 ≤ X * A / (δ * ξ * s) := by linarith
  refine ⟨hY8.trans hYZ, (h.hQX.trans hXY).trans hYZ, hδY.trans hYZ, hsY.trans hYZ, hpZ, ?_,
    h.lZ_Z⟩
  have := pow_le_pow_left₀ hY0 hYZ 3
  linarith [h.ι_Y]

lemma QA_le : Q * A ≤ Q * A / δ :=
  le_div_self (by have := h.hQ; have := h.hA; positivity) h.hδ (by linarith [h.hδ2])

lemma l_sh : 0 ≤ log (16 * (Q + 2) / (δ / (8 * (Q + 1))))
    ∧ log (16 * (Q + 2) / (δ / (8 * (Q + 1)))) ≤ 7 * log (Q * A / δ) := by
  obtain ⟨hq4, -, hQq, -, -⟩ := h.Yq_ge
  have hQ := h.hQ; have hδ := h.hδ
  have hQd := h.Q_div
  refine ⟨h.log_ge_of (by linarith), ?_⟩
  have hδ' : 0 < δ / (8 * (Q + 1)) := div_pos hδ (by linarith)
  have e : 16 * (Q + 2) / (δ / (8 * (Q + 1))) = 128 * (Q + 1) * (Q + 2) * (1 / δ) := by
    rw [div_div_eq_mul_div]; ring
  have hb : 16 * (Q + 2) / (δ / (8 * (Q + 1))) ≤ (Q * A / δ) ^ 7 := by
    rw [e]
    have k1 : 128 * (Q + 1) * (Q + 2) * (1 / δ) ≤ 768 * (Q * (Q / δ)) := by
      have : 128 * (Q + 1) * (Q + 2) ≤ 768 * Q ^ 2 := by nlinarith
      have : 0 ≤ 1 / δ := by positivity
      calc 128 * (Q + 1) * (Q + 2) * (1 / δ) ≤ 768 * Q ^ 2 * (1 / δ) := by gcongr
        _ = 768 * (Q * (Q / δ)) := by ring
    have k2 : Q * (Q / δ) ≤ (Q * A / δ) * (Q * A / δ) :=
      mul_le_mul hQq hQd (by positivity) (by linarith)
    have k3 : (768 : ℝ) ≤ (Q * A / δ) ^ 5 :=
      (show (768 : ℝ) ≤ 4 ^ 5 by norm_num).trans (pow_le_pow_left₀ (by norm_num) hq4 5)
    calc 128 * (Q + 1) * (Q + 2) * (1 / δ) ≤ 768 * (Q * (Q / δ)) := k1
      _ ≤ (Q * A / δ) ^ 5 * ((Q * A / δ) * (Q * A / δ)) :=
          mul_le_mul k3 k2 (by positivity) (by positivity)
      _ = (Q * A / δ) ^ 7 := by ring
  have := log_le_of_le_pow 7 (div_pos (by linarith) hδ') hb
  push_cast at this; exact this

lemma l_two : 0 ≤ log (2 / (δ / (8 * (Q + 1))))
    ∧ log (2 / (δ / (8 * (Q + 1)))) ≤ 4 * log (Q * A / δ) := by
  obtain ⟨hq4, -, -, -, -⟩ := h.Yq_ge
  have hQ := h.hQ; have hδ := h.hδ
  have hQd := h.Q_div
  refine ⟨h.log_ge_of (by norm_num), ?_⟩
  have hδ' : 0 < δ / (8 * (Q + 1)) := div_pos hδ (by linarith)
  have e : 2 / (δ / (8 * (Q + 1))) = 16 * (Q + 1) * (1 / δ) := by
    rw [div_div_eq_mul_div]; ring
  have hb : 2 / (δ / (8 * (Q + 1))) ≤ (Q * A / δ) ^ 4 := by
    rw [e]
    have k1 : 16 * (Q + 1) * (1 / δ) ≤ 32 * (Q / δ) := by
      have : 0 ≤ 1 / δ := by positivity
      calc 16 * (Q + 1) * (1 / δ) ≤ 32 * Q * (1 / δ) :=
            mul_le_mul_of_nonneg_right (by linarith) this
        _ = 32 * (Q / δ) := by ring
    have k3 : (32 : ℝ) ≤ (Q * A / δ) ^ 3 :=
      (show (32 : ℝ) ≤ 4 ^ 3 by norm_num).trans (pow_le_pow_left₀ (by norm_num) hq4 3)
    calc 16 * (Q + 1) * (1 / δ) ≤ 32 * (Q / δ) := k1
      _ ≤ (Q * A / δ) ^ 3 * (Q * A / δ) := mul_le_mul k3 hQd (by positivity) (by positivity)
      _ = (Q * A / δ) ^ 4 := by ring
  have := log_le_of_le_pow 4 (div_pos (by norm_num) hδ') hb
  push_cast at this; exact this

lemma l_kc (nQ nA : ℕ) (hQn : (nQ : ℝ) = Q) (hAn : (nA : ℝ) = A) :
    0 ≤ log (16 * (Q + 1) * ((1 + (Q + 1) ^ 3 * A) ^ nQ * (Q + 3) ^ ((nQ + 2) * nA))
        / (δ / (8 * (Q + 1))))
      ∧ log (16 * (Q + 1) * ((1 + (Q + 1) ^ 3 * A) ^ nQ * (Q + 3) ^ ((nQ + 2) * nA))
        / (δ / (8 * (Q + 1)))) ≤ 15 * (Q * A * log (Q * A / δ)) := by
  obtain ⟨hq4, h4q, hQq, -, -⟩ := h.Yq_ge
  have hQ := h.hQ; have hA := h.hA; have hδ := h.hδ
  have hQd := h.Q_div
  have hQA := h.QA_le
  have hl1 := h.lQA1
  have hδ' : 0 < δ / (8 * (Q + 1)) := div_pos hδ (by linarith)
  set Yq := Q * A / δ with hYqd
  have hB1 : 1 ≤ 1 + (Q + 1) ^ 3 * A := by
    have : 0 ≤ (Q + 1) ^ 3 * A := by positivity
    linarith
  have hKc : 1 ≤ (1 + (Q + 1) ^ 3 * A) ^ nQ * (Q + 3) ^ ((nQ + 2) * nA) :=
    one_le_mul_of_one_le_of_one_le (one_le_pow₀ hB1) (one_le_pow₀ (by linarith))
  have hM : 1 ≤ 16 * (Q + 1) / (δ / (8 * (Q + 1))) := by
    rw [le_div_iff₀ hδ']
    have := h.δ'1
    nlinarith
  have e : 16 * (Q + 1) * ((1 + (Q + 1) ^ 3 * A) ^ nQ * (Q + 3) ^ ((nQ + 2) * nA))
        / (δ / (8 * (Q + 1)))
      = 16 * (Q + 1) / (δ / (8 * (Q + 1)))
        * ((1 + (Q + 1) ^ 3 * A) ^ nQ * (Q + 3) ^ ((nQ + 2) * nA)) := by ring
  rw [e, Real.log_mul (by linarith) (by linarith), Real.log_mul (by positivity) (by positivity),
    Real.log_pow, Real.log_pow]
  have hm0 := Real.log_nonneg hM
  have hb0 := Real.log_nonneg hB1
  have hq0 : 0 ≤ log (Q + 3) := Real.log_nonneg (by linarith)
  have hcast : (((nQ + 2) * nA : ℕ) : ℝ) = (Q + 2) * A := by push_cast; rw [hQn, hAn]
  rw [hcast, hQn]
  -- the factor `16 (|Q| + 1)/δ'`
  have k1 : log (16 * (Q + 1) / (δ / (8 * (Q + 1)))) ≤ 7 * log Yq := by
    have e2 : 16 * (Q + 1) / (δ / (8 * (Q + 1))) = 128 * (Q + 1) * (Q + 1) * (1 / δ) := by
      rw [div_div_eq_mul_div]; ring
    have hb : 16 * (Q + 1) / (δ / (8 * (Q + 1))) ≤ Yq ^ 7 := by
      rw [e2]
      have k : 128 * (Q + 1) * (Q + 1) * (1 / δ) ≤ 512 * (Q * (Q / δ)) := by
        have : 128 * (Q + 1) * (Q + 1) ≤ 512 * Q ^ 2 := by nlinarith
        have : 0 ≤ 1 / δ := by positivity
        calc 128 * (Q + 1) * (Q + 1) * (1 / δ) ≤ 512 * Q ^ 2 * (1 / δ) := by gcongr
          _ = 512 * (Q * (Q / δ)) := by ring
      have k2 : Q * (Q / δ) ≤ Yq * Yq := mul_le_mul hQq hQd (by positivity) (by linarith)
      have k3 : (512 : ℝ) ≤ Yq ^ 5 :=
        (show (512 : ℝ) ≤ 4 ^ 5 by norm_num).trans (pow_le_pow_left₀ (by norm_num) hq4 5)
      calc 128 * (Q + 1) * (Q + 1) * (1 / δ) ≤ 512 * (Q * (Q / δ)) := k
        _ ≤ Yq ^ 5 * (Yq * Yq) := mul_le_mul k3 k2 (by positivity) (by positivity)
        _ = Yq ^ 7 := by ring
    have := log_le_of_le_pow 7 (by linarith) hb
    push_cast at this; exact this
  -- `1 + (|Q| + 1)³ |Σ| ≤ Yq⁵`
  have k2 : log (1 + (Q + 1) ^ 3 * A) ≤ 5 * log Yq := by
    have hb : 1 + (Q + 1) ^ 3 * A ≤ Yq ^ 5 := by
      have a1 : 1 + (Q + 1) ^ 3 * A ≤ 9 * (Q ^ 2 * (Q * A)) := by
        have : (Q + 1) ^ 3 ≤ 8 * Q ^ 3 := by
          have := pow_le_pow_left₀ (by linarith) (show Q + 1 ≤ 2 * Q by linarith) 3
          linarith [show (2 * Q) ^ 3 = 8 * Q ^ 3 by ring]
        have : 1 ≤ Q ^ 3 * A := one_le_mul_of_one_le_of_one_le (one_le_pow₀ hQ) (by linarith)
        have : (Q + 1) ^ 3 * A ≤ 8 * Q ^ 3 * A := by gcongr
        nlinarith
      have a2 : Q ^ 2 * (Q * A) ≤ Yq ^ 2 * Yq :=
        mul_le_mul (pow_le_pow_left₀ (by linarith) hQq 2) hQA (by positivity) (by positivity)
      have a3 : (9 : ℝ) ≤ Yq ^ 2 := by nlinarith only [hq4]
      calc 1 + (Q + 1) ^ 3 * A ≤ 9 * (Q ^ 2 * (Q * A)) := a1
        _ ≤ Yq ^ 2 * (Yq ^ 2 * Yq) := mul_le_mul a3 a2 (by positivity) (by positivity)
        _ = Yq ^ 5 := by ring
    have := log_le_of_le_pow 5 (by linarith) hb
    push_cast at this; exact this
  have k3 : log (Q + 3) ≤ log Yq := Real.log_le_log (by linarith) (by linarith)
  have m1 : Q * log (1 + (Q + 1) ^ 3 * A) ≤ Q * (5 * log Yq) :=
    mul_le_mul_of_nonneg_left k2 (by linarith)
  have m2 : (Q + 2) * A * log (Q + 3) ≤ (3 * Q * A) * log Yq := by
    have : (Q + 2) * A ≤ 3 * Q * A := by nlinarith only [hQ, hA]
    exact mul_le_mul this k3 hq0 (by positivity)
  have m3 : 2 * log Yq ≤ Q * A * log Yq := by
    have : 2 ≤ Q * A := by nlinarith only [hQ, hA]
    exact mul_le_mul_of_nonneg_right this (by linarith)
  have m4 : 2 * Q * log Yq ≤ Q * A * log Yq := by
    have : 2 * Q ≤ Q * A := by nlinarith only [hQ, hA]
    exact mul_le_mul_of_nonneg_right this (by linarith)
  have p1 : 0 ≤ Q * log (1 + (Q + 1) ^ 3 * A) := mul_nonneg (by linarith) hb0
  have p2 : 0 ≤ (Q + 2) * A * log (Q + 3) := by positivity
  refine ⟨by linarith only [hm0, p1, p2], ?_⟩
  have p3 : 0 ≤ Q * A * log Yq := by have : 0 ≤ log Yq := by linarith
                                     positivity
  linarith only [k1, m1, m2, m3, m4, p3]

/-- `P ≤ 2¹⁴ Z⁴`. -/
lemma P_Z : triePool (X + 4) Q A δ (ξ / 2) (ξ / 4) p (1 / 2 - s / 2)
    ≤ 2 ^ 14 * (X * A / (δ * ξ * p * s)) ^ 4 := by
  obtain ⟨-, hP⟩ := tc_pool h
  obtain ⟨-, -, hPsZ⟩ := tc_pool_le h
  obtain ⟨hZ8, -, -, hsZ, hpZ, -, hlZZ⟩ := h.Z_facts
  have hs := h.hs; have hp := h.hp
  set Z := X * A / (δ * ξ * p * s)
  have e3 : shZ A X δ ξ p s / (p * s ^ 2) = shZ A X δ ξ p s * (1 / p) * (1 / s) ^ 2 := by
    field_simp
  have k : shZ A X δ ξ p s * (1 / p) * (1 / s) ^ 2 ≤ Z * Z * Z ^ 2 := by
    have hs0 : 0 ≤ 1 / s := by positivity
    have hp0 : 0 ≤ 1 / p := by positivity
    have k := mul_le_mul hlZZ hpZ hp0 (by linarith)
    exact mul_le_mul k (pow_le_pow_left₀ hs0 hsZ 2) (by positivity) (by positivity)
  have e4 : Z * Z * Z ^ 2 = Z ^ 4 := by ring
  linarith only [hP, hPsZ, e3, k, e4]

/-- `|Q| ι² log Z/s⁴ ≤ 9 Z¹²`. -/
lemma Pi_Z : Q * shι Q X ξ ^ 2 * shZ A X δ ξ p s / s ^ 4 ≤ 9 * (X * A / (δ * ξ * p * s)) ^ 12 := by
  obtain ⟨hZ8, hQZ, -, hsZ, -, hιZ, hlZZ⟩ := h.Z_facts
  have hs := h.hs
  have hι0 : 0 ≤ shι Q X ξ := by linarith [h.ι1]
  have hl0 : 0 ≤ shZ A X δ ξ p s := by linarith [h.lZ1]
  set Z := X * A / (δ * ξ * p * s)
  have e : Q * shι Q X ξ ^ 2 * shZ A X δ ξ p s / s ^ 4
      = Q * shι Q X ξ ^ 2 * shZ A X δ ξ p s * (1 / s) ^ 4 := by
    field_simp
  rw [e]
  have k1 : shι Q X ξ ^ 2 ≤ (3 * Z ^ 3) ^ 2 := pow_le_pow_left₀ hι0 hιZ 2
  have k2 : Q * shι Q X ξ ^ 2 ≤ Z * (3 * Z ^ 3) ^ 2 :=
    mul_le_mul hQZ k1 (by positivity) (by linarith)
  have k3 : Q * shι Q X ξ ^ 2 * shZ A X δ ξ p s ≤ Z * (3 * Z ^ 3) ^ 2 * Z :=
    mul_le_mul k2 hlZZ hl0 (by positivity)
  have k4 : (1 / s) ^ 4 ≤ Z ^ 4 := pow_le_pow_left₀ (by positivity) hsZ 4
  calc Q * shι Q X ξ ^ 2 * shZ A X δ ξ p s * (1 / s) ^ 4
      ≤ Z * (3 * Z ^ 3) ^ 2 * Z * Z ^ 4 :=
        mul_le_mul k3 k4 (by positivity) (by positivity)
    _ = 9 * Z ^ 12 := by ring

/-- `Q² log(Q|Σ|/δ) ≤ shPref`. -/
lemma pref_ge : 0 ≤ shPref Q A X δ ξ p s ∧ Q ^ 2 * log (Q * A / δ) ≤ shPref Q A X δ ξ p s
    ∧ Q ≤ shPref Q A X δ ξ p s ∧ Q ^ 2 * shι Q X ξ ^ 2 * shZ A X δ ξ p s ≤ shPref Q A X δ ξ p s
    := by
  have hs := h.hs; have hξ := h.hξ; have hQ := h.hQ
  have hι := h.ι1; have hP := h.lP1; have hZ := h.lZ1
  have hl := h.lQA_le.trans h.lA_le
  have hl1 := h.lQA1
  have h1 : 0 ≤ Q ^ 3 * shι Q X ξ ^ 2 / s ^ 4 * shP Q A X δ ξ p s := by positivity
  have h2 : 0 ≤ Q ^ 2 * (X ^ 2 / (ξ ^ 2 * s ^ 4)) * shZ A X δ ξ p s := by positivity
  have e : shPref Q A X δ ξ p s = Q ^ 3 * shι Q X ξ ^ 2 / s ^ 4 * shP Q A X δ ξ p s
      + Q ^ 2 * (X ^ 2 / (ξ ^ 2 * s ^ 4)) * shZ A X δ ξ p s
      + Q ^ 2 * shι Q X ξ ^ 2 * shZ A X δ ξ p s := by unfold shPref; ring
  have hq2 : 1 ≤ Q ^ 2 := one_le_pow₀ hQ
  have hι2 : 1 ≤ shι Q X ξ ^ 2 := one_le_pow₀ hι
  have k1 : Q ^ 2 * log (Q * A / δ) ≤ Q ^ 2 * shι Q X ξ ^ 2 * shZ A X δ ξ p s := by
    rw [mul_assoc]
    apply mul_le_mul_of_nonneg_left _ (by positivity)
    calc log (Q * A / δ) ≤ shZ A X δ ξ p s := hl
      _ ≤ shι Q X ξ ^ 2 * shZ A X δ ξ p s := le_mul_of_one_le_left (by linarith) hι2
  have k2 : Q ≤ Q ^ 2 * log (Q * A / δ) := by
    calc Q ≤ Q ^ 2 := by nlinarith
      _ ≤ Q ^ 2 * log (Q * A / δ) := le_mul_of_one_le_right (by positivity) hl1
  refine ⟨by rw [e]; positivity, by linarith, by linarith, by linarith⟩

/-- `Q² C ≤ 2⁷¹ shPref`. -/
lemma QC : 0 ≤ trieCount (X + 4) Q A δ (ξ / 2) (ξ / 4) p (1 / 2 - s / 2)
    ∧ Q ^ 2 * trieCount (X + 4) Q A δ (ξ / 2) (ξ / 4) p (1 / 2 - s / 2)
      ≤ 2 ^ 71 * shPref Q A X δ ξ p s := by
  obtain ⟨hC0, hC⟩ := tc_count h
  have hs := h.hs; have hξ := h.hξ; have hQ := h.hQ
  have hZ := h.lZ1
  refine ⟨hC0, ?_⟩
  have hτ : (X / (ξ * s ^ 2) + shι Q X ξ) ^ 2
      ≤ 2 * (X ^ 2 / (ξ ^ 2 * s ^ 4) + shι Q X ξ ^ 2) := by
    have e : X ^ 2 / (ξ ^ 2 * s ^ 4) = (X / (ξ * s ^ 2)) ^ 2 := by ring
    rw [e]
    nlinarith [sq_nonneg (X / (ξ * s ^ 2) - shι Q X ξ)]
  have k := mul_le_mul_of_nonneg_left hC (by positivity : (0 : ℝ) ≤ Q ^ 2)
  have k2 : Q ^ 2 * ((X / (ξ * s ^ 2) + shι Q X ξ) ^ 2 * shZ A X δ ξ p s)
      ≤ Q ^ 2 * (2 * (X ^ 2 / (ξ ^ 2 * s ^ 4) + shι Q X ξ ^ 2) * shZ A X δ ξ p s) := by
    gcongr
  have e : shPref Q A X δ ξ p s = Q ^ 3 * shι Q X ξ ^ 2 / s ^ 4 * shP Q A X δ ξ p s
      + Q ^ 2 * (X ^ 2 / (ξ ^ 2 * s ^ 4) + shι Q X ξ ^ 2) * shZ A X δ ξ p s := rfl
  have e2 : Q ^ 2 * (2 ^ 70 * (Q * shι Q X ξ ^ 2 * shP Q A X δ ξ p s / s ^ 4
        + (X / (ξ * s ^ 2) + shι Q X ξ) ^ 2 * shZ A X δ ξ p s))
      = 2 ^ 70 * (Q ^ 3 * shι Q X ξ ^ 2 / s ^ 4 * shP Q A X δ ξ p s)
        + 2 ^ 70 * (Q ^ 2 * ((X / (ξ * s ^ 2) + shι Q X ξ) ^ 2 * shZ A X δ ξ p s)) := by ring
  have e3 : Q ^ 2 * (2 * (X ^ 2 / (ξ ^ 2 * s ^ 4) + shι Q X ξ ^ 2) * shZ A X δ ξ p s)
      = 2 * (Q ^ 2 * (X ^ 2 / (ξ ^ 2 * s ^ 4) + shι Q X ξ ^ 2) * shZ A X δ ξ p s) := by ring
  have p1 : 0 ≤ Q ^ 3 * shι Q X ξ ^ 2 / s ^ 4 * shP Q A X δ ξ p s := by
    have := h.lP1; positivity
  have p2 : 0 ≤ Q ^ 2 * (X ^ 2 / (ξ ^ 2 * s ^ 4) + shι Q X ξ ^ 2) * shZ A X δ ξ p s := by
    positivity
  rw [e]
  linarith only [k, k2, e2, e3, p1, p2]

end TC

section Tol

variable {Q A X δ ξ p s : ℝ} (h : TC Q A X δ ξ p s)
include h

/-- `E₁`'s log term against `Z⁵⁵`. -/
lemma tc_lL :
    0 ≤ log (128 * (Q + 2) * (triePool (X + 4) Q A δ (ξ / 2) (ξ / 4) p (1 / 2 - s / 2) + 2)
      * (13 + 2 * Real.logb 2 (trieCount (X + 4) Q A δ (ξ / 2) (ξ / 4) p (1 / 2 - s / 2) + 1))
      / (δ / (8 * (Q + 1))))
    ∧ log (128 * (Q + 2) * (triePool (X + 4) Q A δ (ξ / 2) (ξ / 4) p (1 / 2 - s / 2) + 2)
      * (13 + 2 * Real.logb 2 (trieCount (X + 4) Q A δ (ξ / 2) (ξ / 4) p (1 / 2 - s / 2) + 1))
      / (δ / (8 * (Q + 1)))) ≤ 55 * shZ A X δ ξ p s := by
  obtain ⟨hZ8, hQZ, hδZ, -, -, -, -⟩ := h.Z_facts
  obtain ⟨hP1, -⟩ := tc_pool h
  obtain ⟨hC0, -⟩ := tc_count h
  have hC1 := tc_count_le h
  have hPZ := h.P_Z
  have hPi := h.Pi_Z
  have hQ := h.hQ; have hδ := h.hδ
  set Z := X * A / (δ * ξ * p * s) with hZd
  set P := triePool (X + 4) Q A δ (ξ / 2) (ξ / 4) p (1 / 2 - s / 2)
  set C := trieCount (X + 4) Q A δ (ξ / 2) (ξ / 4) p (1 / 2 - s / 2)
  have hl2 : 0 < log 2 := Real.log_pos (by norm_num)
  have hl2' : 1 / 2 < log 2 := by have := Real.log_two_gt_d9; linarith
  have hlb0 : 0 ≤ Real.logb 2 (C + 1) := Real.logb_nonneg (by norm_num) (by linarith)
  have hlb : Real.logb 2 (C + 1) ≤ 2 * (C + 1) := by
    rw [Real.logb, div_le_iff₀ hl2]
    have := Real.log_le_sub_one_of_pos (show 0 < C + 1 by linarith)
    nlinarith
  have hnum1 : 1 ≤ 128 * (Q + 2) * (P + 2) * (13 + 2 * Real.logb 2 (C + 1)) := by
    have k1 : 1 ≤ (Q + 2) * (P + 2) := by nlinarith
    have k2 : 1 ≤ 13 + 2 * Real.logb 2 (C + 1) := by linarith
    have := one_le_mul_of_one_le_of_one_le k1 k2
    nlinarith
  refine ⟨h.log_ge_of hnum1, ?_⟩
  have e : 128 * (Q + 2) * (P + 2) * (13 + 2 * Real.logb 2 (C + 1)) / (δ / (8 * (Q + 1)))
      = 1024 * (Q + 2) * (P + 2) * (13 + 2 * Real.logb 2 (C + 1)) * (Q + 1) * (1 / δ) := by
    rw [div_div_eq_mul_div]; ring
  have hZ1 : 1 ≤ Z := by linarith
  have hb : 128 * (Q + 2) * (P + 2) * (13 + 2 * Real.logb 2 (C + 1)) / (δ / (8 * (Q + 1)))
      ≤ Z ^ 55 := by
    rw [e]
    have hZ4 : 8 ^ 4 ≤ Z ^ 4 := pow_le_pow_left₀ (by norm_num) hZ8 4
    have hZ12 : 8 ^ 12 ≤ Z ^ 12 := pow_le_pow_left₀ (by norm_num) hZ8 12
    have a1 : Q + 2 ≤ 3 * Z := by linarith
    have a2 : P + 2 ≤ 2 ^ 15 * Z ^ 4 := by linarith
    have a3 : 13 + 2 * Real.logb 2 (C + 1) ≤ 2 ^ 80 * Z ^ 12 := by linarith
    have a4 : Q + 1 ≤ 2 * Z := by linarith
    have hP0 : 0 ≤ P + 2 := by linarith
    have hl0 : 0 ≤ 13 + 2 * Real.logb 2 (C + 1) := by linarith
    have k1 : 1024 * (Q + 2) ≤ 1024 * (3 * Z) := by linarith
    have k2 := mul_le_mul k1 a2 hP0 (by positivity)
    have k3 := mul_le_mul k2 a3 hl0 (by positivity)
    have k4 := mul_le_mul k3 a4 (by linarith) (by positivity)
    have k5 := mul_le_mul k4 hδZ (by positivity) (by positivity)
    have h36 : (2 : ℝ) ^ 108 ≤ Z ^ 36 :=
      (show (2 : ℝ) ^ 108 ≤ 8 ^ 36 by norm_num).trans (pow_le_pow_left₀ (by norm_num) hZ8 36)
    calc 1024 * (Q + 2) * (P + 2) * (13 + 2 * Real.logb 2 (C + 1)) * (Q + 1) * (1 / δ)
        ≤ 1024 * (3 * Z) * (2 ^ 15 * Z ^ 4) * (2 ^ 80 * Z ^ 12) * (2 * Z) * Z := k5
      _ = 3 * 2 ^ 106 * Z ^ 19 := by ring
      _ ≤ 2 ^ 108 * Z ^ 19 := by
          have : 0 ≤ Z ^ 19 := by positivity
          nlinarith
      _ ≤ Z ^ 36 * Z ^ 19 := mul_le_mul_of_nonneg_right h36 (by positivity)
      _ = Z ^ 55 := by ring
  have hpos : 0 < 128 * (Q + 2) * (P + 2) * (13 + 2 * Real.logb 2 (C + 1)) / (δ / (8 * (Q + 1)))
    := div_pos (by linarith) (div_pos hδ (by linarith))
  have := log_le_of_le_pow 55 hpos hb
  push_cast at this; exact this

lemma tc_Ia0 : 0 ≤ trieIa (X + 4) Q (ξ / 2) (ξ / 4) (1 / 2 - s / 2) := by
  have := (tc_Iι h).1
  have := h.X1; have := h.hξ; have := h.hs
  unfold trieIa
  rw [show (1 : ℝ) / 2 - (1 / 2 - s / 2) = s / 2 by ring]
  have : 0 ≤ 2 * (10 / (ξ / 2) + 8 / (ξ / 4)) * (X + 4) / (s / 2) := by positivity
  linarith

/-- `trieTolInv ≤ 2¹³⁸ shTol`. -/
lemma tc_tol (nQ nA : ℕ) (hQn : (nQ : ℝ) = Q) (hAn : (nA : ℝ) = A) :
    1 ≤ trieTolInv (X + 4) Q A nQ nA δ (ξ / 2) (ξ / 4) p (1 / 2 - s / 2)
      ∧ trieTolInv (X + 4) Q A nQ nA δ (ξ / 2) (ξ / 4) p (1 / 2 - s / 2)
        ≤ 2 ^ 138 * shTol Q A X δ ξ p s := by
  obtain ⟨hε1, hε⟩ := tc_Iε h
  obtain ⟨hι1, hι⟩ := tc_Iι h
  have hIa := tc_Ia h
  have hIa0 := tc_Ia0 h
  obtain ⟨hC0, -⟩ := tc_count h
  have hC1 := tc_count_le h
  obtain ⟨hP1, -⟩ := tc_pool h
  obtain ⟨hP2, -, hPsZ⟩ := tc_pool_le h
  obtain ⟨hK0, hK⟩ := tc_budget h
  obtain ⟨hlg0, hlg⟩ := h.l_kc nQ nA hQn hAn
  obtain ⟨hlL0, hlL⟩ := tc_lL h
  have hs := h.hs; have hp := h.hp; have hQ := h.hQ; have hp1 := h.hp1
  have hι' := h.ι1; have hZ := h.lZ1; have hρ := h.ρ1; have hl1 := h.lQA1
  have eG : shTol Q A X δ ξ p s
      = shρ X ξ * (Q * shι Q X ξ ^ 3 * shZ A X δ ξ p s / s ^ 6
        + Q ^ 3 * shι Q X ξ ^ 3 * shZ A X δ ξ p s / s ^ 4
        + Q ^ 2 * shZ A X δ ξ p s / (p ^ 2 * s ^ 2))
      + shρ X ξ * (Q ^ 2 * A * log (Q * A / δ)) := by unfold shTol; ring
  have eP : shA A X δ ξ s / (p * s ^ 2) + log (Q / δ) / p ≤ 2 * (shZ A X δ ξ p s / (p * s ^ 2))
    := hPsZ
  set ι := shι Q X ξ
  set lZ := shZ A X δ ξ p s
  set ρ := shρ X ξ
  set lQA := log (Q * A / δ)
  set Iε := trieIε (X + 4) (ξ / 2)
  set Iι := trieIι (X + 4) Q (ξ / 2) (ξ / 4)
  set Ia := trieIa (X + 4) Q (ξ / 2) (ξ / 4) (1 / 2 - s / 2)
  set C := trieCount (X + 4) Q A δ (ξ / 2) (ξ / 4) p (1 / 2 - s / 2)
  set P := triePool (X + 4) Q A δ (ξ / 2) (ξ / 4) p (1 / 2 - s / 2)
  set K₁ := trieBudget (X + 4) Q A δ (ξ / 2)
  set lg := log (16 * (Q + 1) * ((1 + (Q + 1) ^ 3 * A) ^ nQ * (Q + 3) ^ ((nQ + 2) * nA))
    / (δ / (8 * (Q + 1))))
  set lL := log (128 * (Q + 2) * (P + 2) * (13 + 2 * Real.logb 2 (C + 1)) / (δ / (8 * (Q + 1))))
  set G1 := Q * ι ^ 3 * lZ / s ^ 6
  set G2 := Q ^ 3 * ι ^ 3 * lZ / s ^ 4
  set G3 := Q ^ 2 * lZ / (p ^ 2 * s ^ 2)
  set E₁ := 2 ^ 20 * Ia * (C + 1) / (1 / 2 - (1 / 2 - s / 2))
    + (Q + 3) ^ 2 * (2 ^ 24 * Iι * (C + 1) + 2 ^ 28 * Iι ^ 2 * lL + 2 ^ 18 * (P + 2) / p)
  have eT : trieTolInv (X + 4) Q A nQ nA δ (ξ / 2) (ξ / 4) p (1 / 2 - s / 2)
      = 1536 * Iε * E₁ + 1536 * lg * Iε + K₁ := rfl
  have hG1 : 0 ≤ G1 := by positivity
  have hG2 : 0 ≤ G2 := by positivity
  have hG3 : 0 ≤ G3 := by positivity
  -- `E₁`'s first term
  have t1 : 2 ^ 20 * Ia * (C + 1) / (1 / 2 - (1 / 2 - s / 2)) ≤ 2 ^ 114 * G1 := by
    have e1 : 2 ^ 20 * Ia * (C + 1) / (1 / 2 - (1 / 2 - s / 2))
        = 2 ^ 21 * (Ia * (C + 1)) * (1 / s) := by
      rw [show (1 : ℝ) / 2 - (1 / 2 - s / 2) = s / 2 by ring, div_div_eq_mul_div]; ring
    have k : Ia * (C + 1) ≤ (2 ^ 20 * ι / s) * (2 ^ 73 * (Q * ι ^ 2 * lZ / s ^ 4)) :=
      mul_le_mul hIa hC1 (by linarith) (by positivity)
    have k2 := mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left k
      (by norm_num : (0 : ℝ) ≤ 2 ^ 21)) (by positivity : (0 : ℝ) ≤ 1 / s)
    have e2 : 2 ^ 21 * ((2 ^ 20 * ι / s) * (2 ^ 73 * (Q * ι ^ 2 * lZ / s ^ 4))) * (1 / s)
        = 2 ^ 114 * G1 := by
      simp only [G1]; field_simp; try ring
    rw [e1]; linarith
  -- the second
  have a1 : 2 ^ 24 * Iι * (C + 1) ≤ 2 ^ 116 * (Q * ι ^ 3 * lZ / s ^ 4) := by
    have k : Iι * (C + 1) ≤ (2 ^ 19 * ι) * (2 ^ 73 * (Q * ι ^ 2 * lZ / s ^ 4)) :=
      mul_le_mul hι hC1 (by linarith) (by positivity)
    have e2 : 2 ^ 24 * ((2 ^ 19 * ι) * (2 ^ 73 * (Q * ι ^ 2 * lZ / s ^ 4)))
        = 2 ^ 116 * (Q * ι ^ 3 * lZ / s ^ 4) := by ring
    have := mul_le_mul_of_nonneg_left k (by norm_num : (0 : ℝ) ≤ 2 ^ 24)
    rw [mul_assoc]; linarith
  have a2 : 2 ^ 28 * Iι ^ 2 * lL ≤ 2 ^ 72 * (ι ^ 2 * lZ) := by
    have k1 : Iι ^ 2 ≤ (2 ^ 19 * ι) ^ 2 := pow_le_pow_left₀ (by linarith) hι 2
    have k : Iι ^ 2 * lL ≤ (2 ^ 19 * ι) ^ 2 * (55 * lZ) :=
      mul_le_mul k1 hlL hlL0 (by positivity)
    have e2 : 2 ^ 28 * ((2 ^ 19 * ι) ^ 2 * (55 * lZ)) = 55 * 2 ^ 66 * (ι ^ 2 * lZ) := by ring
    have := mul_le_mul_of_nonneg_left k (by norm_num : (0 : ℝ) ≤ 2 ^ 28)
    have : 0 ≤ ι ^ 2 * lZ := by positivity
    rw [mul_assoc]; linarith
  have a3 : 2 ^ 18 * (P + 2) / p ≤ 2 ^ 33 * (lZ / (p ^ 2 * s ^ 2)) := by
    have k : P + 2 ≤ 2 ^ 15 * (lZ / (p * s ^ 2)) := by linarith
    have k2 : 2 ^ 18 * (P + 2) / p ≤ 2 ^ 18 * (2 ^ 15 * (lZ / (p * s ^ 2))) / p :=
      div_le_div_of_nonneg_right (by linarith) hp.le
    have e2 : 2 ^ 18 * (2 ^ 15 * (lZ / (p * s ^ 2))) / p = 2 ^ 33 * (lZ / (p ^ 2 * s ^ 2)) := by
      field_simp; try ring
    linarith
  have hin0 : 0 ≤ 2 ^ 24 * Iι * (C + 1) + 2 ^ 28 * Iι ^ 2 * lL + 2 ^ 18 * (P + 2) / p := by
    have : 0 ≤ 2 ^ 24 * Iι * (C + 1) := mul_nonneg (by positivity) (by linarith)
    have : 0 ≤ 2 ^ 28 * Iι ^ 2 * lL := mul_nonneg (by positivity) hlL0
    have : 0 ≤ 2 ^ 18 * (P + 2) / p := div_nonneg (by linarith) hp.le
    linarith
  have hin1 : 1 ≤ 2 ^ 24 * Iι * (C + 1) + 2 ^ 28 * Iι ^ 2 * lL + 2 ^ 18 * (P + 2) / p := by
    have : 0 ≤ 2 ^ 24 * Iι * (C + 1) := mul_nonneg (by positivity) (by linarith)
    have : 0 ≤ 2 ^ 28 * Iι ^ 2 * lL := mul_nonneg (by positivity) hlL0
    have : 1 ≤ 2 ^ 18 * (P + 2) / p := by rw [le_div_iff₀ hp]; linarith
    linarith
  have hQ3 : (Q + 3) ^ 2 ≤ 16 * Q ^ 2 := by nlinarith only [hQ]
  have hQ31 : 1 ≤ (Q + 3) ^ 2 := by nlinarith only [hQ]
  have t2 := mul_le_mul hQ3 (show 2 ^ 24 * Iι * (C + 1) + 2 ^ 28 * Iι ^ 2 * lL
      + 2 ^ 18 * (P + 2) / p ≤ 2 ^ 116 * (Q * ι ^ 3 * lZ / s ^ 4) + 2 ^ 72 * (ι ^ 2 * lZ)
        + 2 ^ 33 * (lZ / (p ^ 2 * s ^ 2)) by linarith) hin0 (by positivity)
  have e2 : 16 * Q ^ 2 * (2 ^ 116 * (Q * ι ^ 3 * lZ / s ^ 4) + 2 ^ 72 * (ι ^ 2 * lZ)
        + 2 ^ 33 * (lZ / (p ^ 2 * s ^ 2)))
      = 2 ^ 120 * G2 + 2 ^ 76 * (Q ^ 2 * ι ^ 2 * lZ) + 2 ^ 37 * G3 := by
    simp only [G2, G3]; ring
  have m : Q ^ 2 * ι ^ 2 * lZ ≤ G2 := by
    have m1 : Q ^ 2 * ι ^ 2 ≤ Q ^ 3 * ι ^ 3 :=
      mul_le_mul (pow_le_pow_right₀ hQ (by norm_num)) (pow_le_pow_right₀ hι' (by norm_num))
        (by positivity) (by positivity)
    have m2 : Q ^ 2 * ι ^ 2 * lZ ≤ Q ^ 3 * ι ^ 3 * lZ := mul_le_mul_of_nonneg_right m1 (by linarith)
    have m3 : Q ^ 3 * ι ^ 3 * lZ ≤ Q ^ 3 * ι ^ 3 * lZ / s ^ 4 :=
      le_div_self (by positivity) (by positivity) (pow_le_one₀ hs.le (by linarith [h.hs2]))
    linarith
  have ht1 : 0 ≤ 2 ^ 20 * Ia * (C + 1) / (1 / 2 - (1 / 2 - s / 2)) := by
    rw [show (1 : ℝ) / 2 - (1 / 2 - s / 2) = s / 2 by ring]
    exact div_nonneg (mul_nonneg (by positivity) (by linarith)) (by positivity)
  have hE : E₁ ≤ 2 ^ 121 * (G1 + G2 + G3) := by
    simp only [E₁]; linarith
  have hE1 : 1 ≤ E₁ := by
    simp only [E₁]
    have := mul_le_mul hQ31 hin1 zero_le_one (by positivity)
    linarith
  -- the tolerance
  have k1 : Iε * E₁ ≤ (45 * ρ) * (2 ^ 121 * (G1 + G2 + G3)) :=
    mul_le_mul hε hE (by linarith) (by positivity)
  have k2 : lg * Iε ≤ (15 * (Q * A * lQA)) * (45 * ρ) :=
    mul_le_mul hlg hε (by linarith) (by have := h.hA; positivity)
  have k3 : ρ * (Q * A * lQA) ≤ ρ * (Q ^ 2 * A * lQA) := by
    apply mul_le_mul_of_nonneg_left _ (by linarith)
    have hq : Q ≤ Q ^ 2 := by nlinarith only [hQ]
    have : 0 ≤ A * lQA := by have := h.hA; positivity
    have := mul_le_mul_of_nonneg_right hq this
    linarith only [this]
  have hρG : 0 ≤ ρ * (G1 + G2 + G3) := by positivity
  have hρQ : 0 ≤ ρ * (Q ^ 2 * A * lQA) := by have := h.hA; positivity
  have e3 : (45 * ρ) * (2 ^ 121 * (G1 + G2 + G3)) = 45 * 2 ^ 121 * (ρ * (G1 + G2 + G3)) := by ring
  have e4 : (15 * (Q * A * lQA)) * (45 * ρ) = 675 * (ρ * (Q * A * lQA)) := by ring
  have e5 : Q ^ 2 * A * ρ * lQA = ρ * (Q ^ 2 * A * lQA) := by ring
  have hεE : 1 ≤ Iε * E₁ := one_le_mul_of_one_le_of_one_le hε1 hE1
  have hlgε : 0 ≤ lg * Iε := mul_nonneg hlg0 (by linarith)
  rw [eT, eG]
  have eG' : Q * ι ^ 3 * lZ / s ^ 6 + Q ^ 3 * ι ^ 3 * lZ / s ^ 4 + Q ^ 2 * lZ / (p ^ 2 * s ^ 2)
      = G1 + G2 + G3 := rfl
  rw [eG']
  refine ⟨by linarith, ?_⟩
  rw [e3] at k1; rw [e4] at k2; rw [e5] at hK
  have e6 : 1536 * Iε * E₁ = 1536 * (Iε * E₁) := by ring
  have e7 : 1536 * lg * Iε = 1536 * (lg * Iε) := by ring
  rw [e6, e7]
  linarith

/-- `Q HB ≤ 2⁹³ ρ shPref`. -/
lemma tc_hb : 1 ≤ trieHB (X + 4) Q A δ (ξ / 2) (ξ / 4) p (1 / 2 - s / 2)
    ∧ Q * trieHB (X + 4) Q A δ (ξ / 2) (ξ / 4) p (1 / 2 - s / 2)
      ≤ 2 ^ 93 * (shρ X ξ * shPref Q A X δ ξ p s) := by
  obtain ⟨hε1, hε⟩ := tc_Iε h
  obtain ⟨hC0, hQC⟩ := h.QC
  obtain ⟨hsh0, hsh⟩ := h.l_sh
  obtain ⟨hl20, hl2⟩ := h.l_two
  obtain ⟨hP0, hPl, hPQ, -⟩ := h.pref_ge
  have hQ := h.hQ; have hl1 := h.lQA1; have hρ := h.ρ1
  have hlog2 := Real.log_two_lt_d9
  have hlog2' := Real.log_two_gt_d9
  set Iε := trieIε (X + 4) (ξ / 2)
  set C := trieCount (X + 4) Q A δ (ξ / 2) (ξ / 4) p (1 / 2 - s / 2)
  set lsh := log (16 * (Q + 2) / (δ / (8 * (Q + 1))))
  set l2 := log (2 / (δ / (8 * (Q + 1))))
  set lQA := log (Q * A / δ)
  set sP := shPref Q A X δ ξ p s
  set ρ := shρ X ξ
  set SH := 24 * C + 16 * lsh + 2
  have eH : trieHB (X + 4) Q A δ (ξ / 2) (ξ / 4) p (1 / 2 - s / 2)
      = 384 * Iε * ((Q + 1) * SH * log 2 + l2) + 1 := rfl
  have hSH : 0 ≤ SH := by positivity
  have hin : 0 ≤ (Q + 1) * SH * log 2 + l2 := by
    have : 0 ≤ (Q + 1) * SH * log 2 := mul_nonneg (mul_nonneg (by linarith) hSH) (by linarith)
    linarith
  rw [eH]
  have hHB0 : 0 ≤ 384 * Iε * ((Q + 1) * SH * log 2 + l2) :=
    mul_nonneg (mul_nonneg (by norm_num) (by linarith)) hin
  refine ⟨by linarith, ?_⟩
  have m0 : (Q + 1) * log 2 ≤ 2 * Q := by
    have : (Q + 1) * log 2 ≤ (Q + 1) * 1 := mul_le_mul_of_nonneg_left (by linarith) (by linarith)
    linarith
  have m1 : Q * ((Q + 1) * SH * log 2) ≤ Q * SH * (2 * Q) := by
    rw [show Q * ((Q + 1) * SH * log 2) = Q * SH * ((Q + 1) * log 2) by ring]
    exact mul_le_mul_of_nonneg_left m0 (by positivity)
  have m2 : Q ^ 2 * lsh ≤ Q ^ 2 * (7 * lQA) := mul_le_mul_of_nonneg_left hsh (by positivity)
  have m3 : Q ^ 2 ≤ Q ^ 2 * lQA := le_mul_of_one_le_right (by positivity) hl1
  have m4 : Q * l2 ≤ Q * (4 * lQA) := mul_le_mul_of_nonneg_left hl2 (by linarith)
  have m5 : Q * lQA ≤ Q ^ 2 * lQA :=
    mul_le_mul_of_nonneg_right (by nlinarith only [hQ]) (by linarith)
  have hQin : Q * ((Q + 1) * SH * log 2 + l2) ≤ 2 ^ 77 * sP := by
    have e : Q * SH * (2 * Q) = 48 * (Q ^ 2 * C) + 32 * (Q ^ 2 * lsh) + 4 * Q ^ 2 := by ring
    have e2 : Q * ((Q + 1) * SH * log 2 + l2) = Q * ((Q + 1) * SH * log 2) + Q * l2 := by ring
    rw [e2]
    linarith
  have hQin0 : 0 ≤ Q * ((Q + 1) * SH * log 2 + l2) := mul_nonneg (by linarith) hin
  have k := mul_le_mul hε hQin hQin0 (by positivity)
  have e3 : Q * (384 * Iε * ((Q + 1) * SH * log 2 + l2) + 1)
      = 384 * (Iε * (Q * ((Q + 1) * SH * log 2 + l2))) + Q := by ring
  have e4 : 45 * ρ * (2 ^ 77 * sP) = 45 * 2 ^ 77 * (ρ * sP) := by ring
  have hρs : sP ≤ ρ * sP := le_mul_of_one_le_left hP0 hρ
  rw [e3]
  linarith

end Tol

section Totals

variable {Q A X δ ξ p s : ℝ} (h : TC Q A X δ ξ p s)
include h

lemma suff_eq : shA A X δ ξ s / (p * s ^ 2) + log (Q / δ) / p = shSuff Q A X δ ξ p s / Q := by
  have := h.hQ
  unfold shSuff
  field_simp

lemma tc_prefs : 0 ≤ (Q + 1) * (6 * (Q + 2) * trieCount (X + 4) Q A δ (ξ / 2) (ξ / 4) p (1 / 2 - s / 2))
    ∧ 0 ≤ shPref Q A X δ ξ p s
    ∧ (Q + 1) * (6 * (Q + 2) * trieCount (X + 4) Q A δ (ξ / 2) (ξ / 4) p (1 / 2 - s / 2))
      ≤ 2 ^ 77 * shPref Q A X δ ξ p s := by
  obtain ⟨hC0, hQC⟩ := h.QC
  have hsP := h.pref_ge.1
  have hQ := h.hQ
  set C := trieCount (X + 4) Q A δ (ξ / 2) (ξ / 4) p (1 / 2 - s / 2)
  have k : (Q + 1) * (Q + 2) ≤ 6 * Q ^ 2 := by nlinarith only [hQ]
  have k2 := mul_le_mul_of_nonneg_right k hC0
  have e : (Q + 1) * (6 * (Q + 2) * C) = 6 * ((Q + 1) * (Q + 2) * C) := by ring
  rw [e]
  refine ⟨by have : 0 ≤ (Q + 1) * (Q + 2) := by nlinarith only [hQ]
             positivity, hsP, ?_⟩
  linarith

lemma tc_suffs : 0 ≤ (Q + 1) * (triePool (X + 4) Q A δ (ξ / 2) (ξ / 4) p (1 / 2 - s / 2) + 1)
    ∧ 0 ≤ shSuff Q A X δ ξ p s
    ∧ (Q + 1) * (triePool (X + 4) Q A δ (ξ / 2) (ξ / 4) p (1 / 2 - s / 2) + 1)
      ≤ 2 ^ 15 * shSuff Q A X δ ξ p s := by
  obtain ⟨hP1, -⟩ := tc_pool h
  obtain ⟨hP2, -, -⟩ := tc_pool_le h
  have hQ := h.hQ
  rw [suff_eq h] at hP2
  set P := triePool (X + 4) Q A δ (ξ / 2) (ξ / 4) p (1 / 2 - s / 2)
  set S := shSuff Q A X δ ξ p s
  have hS : 0 ≤ S / Q := by linarith
  have hS0 : 0 ≤ S := by
    have := mul_nonneg (by linarith : (0 : ℝ) ≤ Q) hS
    rwa [mul_div_cancel₀ _ (by linarith : (Q : ℝ) ≠ 0)] at this
  refine ⟨by positivity, hS0, ?_⟩
  have k : (Q + 1) * (P + 1) ≤ (2 * Q) * (2 ^ 14 * (S / Q)) :=
    mul_le_mul (by linarith) (by linarith) (by linarith) (by positivity)
  have e : (2 * Q) * (2 ^ 14 * (S / Q)) = 2 ^ 15 * S := by field_simp; try ring
  linarith

lemma tc_probes : 0 ≤ (Q + 1) * trieBudget (X + 4) Q A δ (ξ / 2)
    ∧ 0 ≤ shProbe Q A X δ ξ
    ∧ (Q + 1) * trieBudget (X + 4) Q A δ (ξ / 2) ≤ 2 ^ 18 * shProbe Q A X δ ξ := by
  obtain ⟨hK0, hK⟩ := tc_budget h
  have hQ := h.hQ
  have hρ := h.ρ1; have hl := h.lQA1; have hA := h.hA
  set K₁ := trieBudget (X + 4) Q A δ (ξ / 2)
  refine ⟨by positivity, by unfold shProbe; positivity, ?_⟩
  have k : (Q + 1) * K₁ ≤ (2 * Q) * (2 ^ 17 * (Q ^ 2 * A * shρ X ξ * log (Q * A / δ))) :=
    mul_le_mul (by linarith) hK hK0 (by positivity)
  have e : (2 * Q) * (2 ^ 17 * (Q ^ 2 * A * shρ X ξ * log (Q * A / δ)))
      = 2 ^ 18 * shProbe Q A X δ ξ := by unfold shProbe; ring
  linarith

/-- The reads against `shReads`, `lg` the sifts of a probe and `B` the harvest's probes. -/
lemma tc_reads (lg B : ℝ) (hlg0 : 0 ≤ lg) (hlg : lg ≤ 14 * log (X + 1)) (hB0 : 0 ≤ B)
    (hB : B ≤ trieHB (X + 4) Q A δ (ξ / 2) (ξ / 4) p (1 / 2 - s / 2) + 1) :
    0 ≤ learnerReads (Q + 1)
        (6 * (Q + 2) * trieCount (X + 4) Q A δ (ξ / 2) (ξ / 4) p (1 / 2 - s / 2))
        (triePool (X + 4) Q A δ (ξ / 2) (ξ / 4) p (1 / 2 - s / 2) + 1)
        (trieFam (X + 4) Q A δ (ξ / 2) (ξ / 4) (1 / 2 - s / 2))
        (trieBudget (X + 4) Q A δ (ξ / 2)) Q B lg A 0
      ∧ 0 ≤ shReads Q A X δ ξ p s
      ∧ learnerReads (Q + 1)
        (6 * (Q + 2) * trieCount (X + 4) Q A δ (ξ / 2) (ξ / 4) p (1 / 2 - s / 2))
        (triePool (X + 4) Q A δ (ξ / 2) (ξ / 4) p (1 / 2 - s / 2) + 1)
        (trieFam (X + 4) Q A δ (ξ / 2) (ξ / 4) (1 / 2 - s / 2))
        (trieBudget (X + 4) Q A δ (ξ / 2)) Q B lg A 0
        ≤ 2 ^ 112 * shReads Q A X δ ξ p s := by
  obtain ⟨hC0, hQC⟩ := h.QC
  obtain ⟨hP1, -⟩ := tc_pool h
  obtain ⟨hP2, -, -⟩ := tc_pool_le h
  obtain ⟨hF0, hF⟩ := tc_fam h
  obtain ⟨hK0, hK⟩ := tc_budget h
  obtain ⟨hHB1, hHB⟩ := tc_hb h
  obtain ⟨hsP0, -, -, -⟩ := h.pref_ge
  have hQ := h.hQ; have hA := h.hA; have hρ := h.ρ1; have hl1 := h.lQA1
  have hlX : 1 / 2 ≤ log (X + 1) := half_le_log (by linarith [h.X1])
  have hsF : 0 ≤ shFam A X δ ξ s := by have := h.lA1; have := h.hs; unfold shFam; positivity
  rw [suff_eq h] at hP2
  have hsS : 0 ≤ shSuff Q A X δ ξ p s / Q := by linarith
  have eR : shReads Q A X δ ξ p s
      = shPref Q A X δ ξ p s * (shSuff Q A X δ ξ p s / Q)
        + Q ^ 4 * A * shρ X ξ * log (Q * A / δ) * shFam A X δ ξ s * log (X + 1)
        + Q * shPref Q A X δ ξ p s * shFam A X δ ξ s * log (X + 1) * shρ X ξ := by
    unfold shReads shProbe; ring
  set C := trieCount (X + 4) Q A δ (ξ / 2) (ξ / 4) p (1 / 2 - s / 2)
  set P := triePool (X + 4) Q A δ (ξ / 2) (ξ / 4) p (1 / 2 - s / 2)
  set F := trieFam (X + 4) Q A δ (ξ / 2) (ξ / 4) (1 / 2 - s / 2)
  set K₁ := trieBudget (X + 4) Q A δ (ξ / 2)
  set HB := trieHB (X + 4) Q A δ (ξ / 2) (ξ / 4) p (1 / 2 - s / 2)
  set sP := shPref Q A X δ ξ p s
  set sS := shSuff Q A X δ ξ p s
  set sF := shFam A X δ ξ s
  set ρ := shρ X ξ
  set lQA := log (Q * A / δ)
  set lX := log (X + 1)
  set W1 := sP * (sS / Q)
  set W2 := Q ^ 4 * A * ρ * lQA * sF * lX
  set W3 := Q * sP * sF * lX * ρ
  have hW1 : 0 ≤ W1 := by positivity
  have hW2 : 0 ≤ W2 := by positivity
  have hW3 : 0 ≤ W3 := by positivity
  have e : learnerReads (Q + 1) (6 * (Q + 2) * C) (P + 1) F K₁ Q B lg A 0
      = 6 * ((Q + 1) * (Q + 2) * C) * (P + 2) + (Q + 1) ^ 2 * K₁ * F * lg
        + (Q + 1) * Q * (Q + 2) ^ 2 * A * F + (Q + 1) ^ 2 * F * B * lg := by
    unfold learnerReads readsCluster readsSearch readsResift readsHarvest readsStart; ring
  rw [e, eR]
  have hQ1 : 0 ≤ (Q + 1) * (Q + 2) := by nlinarith only [hQ]
  have hq2 : (Q + 1) ^ 2 ≤ 4 * Q ^ 2 := by nlinarith only [hQ]
  -- clusterings
  have c1 : (Q + 1) * (Q + 2) * C ≤ 6 * 2 ^ 71 * sP := by
    have k : (Q + 1) * (Q + 2) ≤ 6 * Q ^ 2 := by nlinarith only [hQ]
    have := mul_le_mul_of_nonneg_right k hC0
    linarith
  have c2 : (Q + 1) * (Q + 2) * C * (P + 2) ≤ (6 * 2 ^ 71 * sP) * (2 ^ 14 * (sS / Q)) :=
    mul_le_mul c1 hP2 (by linarith) (by positivity)
  have c3 : (6 * 2 ^ 71 * sP) * (2 ^ 14 * (sS / Q)) = 6 * 2 ^ 85 * W1 := by ring
  -- search
  have s1 : (Q + 1) ^ 2 * K₁ ≤ (4 * Q ^ 2) * (2 ^ 17 * (Q ^ 2 * A * ρ * lQA)) :=
    mul_le_mul hq2 hK hK0 (by positivity)
  have s2 : (Q + 1) ^ 2 * K₁ * F ≤ (4 * Q ^ 2) * (2 ^ 17 * (Q ^ 2 * A * ρ * lQA)) * (2 ^ 11 * sF) :=
    mul_le_mul s1 hF hF0 (by positivity)
  have s3 : (Q + 1) ^ 2 * K₁ * F * lg
      ≤ (4 * Q ^ 2) * (2 ^ 17 * (Q ^ 2 * A * ρ * lQA)) * (2 ^ 11 * sF) * (14 * lX) :=
    mul_le_mul s2 hlg hlg0 (by positivity)
  have s4 : (4 * Q ^ 2) * (2 ^ 17 * (Q ^ 2 * A * ρ * lQA)) * (2 ^ 11 * sF) * (14 * lX)
      = 56 * 2 ^ 28 * W2 := by ring
  -- re-sifts
  have r1 : (Q + 1) * Q * (Q + 2) ^ 2 * A ≤ 18 * (Q ^ 4 * A) := by
    have k1 : (Q + 1) * Q * (Q + 2) ^ 2 ≤ 18 * Q ^ 4 := by
      have a : (Q + 1) ≤ 2 * Q := by linarith
      have b : (Q + 2) ^ 2 ≤ 9 * Q ^ 2 := by nlinarith only [hQ]
      have := mul_le_mul (mul_le_mul_of_nonneg_right a (by linarith : 0 ≤ Q)) b (by positivity)
        (by positivity)
      linarith [show 2 * Q * Q * (9 * Q ^ 2) = 18 * Q ^ 4 by ring]
    have := mul_le_mul_of_nonneg_right k1 (by linarith : 0 ≤ A)
    linarith
  have r2 : (Q + 1) * Q * (Q + 2) ^ 2 * A * F ≤ 18 * (Q ^ 4 * A) * (2 ^ 11 * sF) :=
    mul_le_mul r1 hF hF0 (by positivity)
  have r3 : Q ^ 4 * A * sF ≤ 2 * W2 := by
    have k1 : 1 ≤ ρ * lQA := one_le_mul_of_one_le_of_one_le hρ hl1
    have k2 : 1 ≤ 2 * (ρ * lQA * lX) := by nlinarith
    have : 0 ≤ Q ^ 4 * A * sF := by positivity
    have := mul_le_mul_of_nonneg_left k2 this
    have e : Q ^ 4 * A * sF * (2 * (ρ * lQA * lX)) = 2 * W2 := by ring
    linarith
  -- harvest
  have hHB0 : 0 ≤ HB := by linarith
  have h1 : B ≤ 2 * HB := by linarith
  have h2 : (Q + 1) ^ 2 * B ≤ 8 * Q * (Q * HB) := by
    have := mul_le_mul hq2 h1 hB0 (by positivity)
    linarith [show 4 * Q ^ 2 * (2 * HB) = 8 * Q * (Q * HB) by ring]
  have h3 : 8 * Q * (Q * HB) ≤ 8 * Q * (2 ^ 93 * (ρ * sP)) :=
    mul_le_mul_of_nonneg_left hHB (by positivity)
  have h4 : (Q + 1) ^ 2 * B * F * lg ≤ (8 * Q * (2 ^ 93 * (ρ * sP))) * (2 ^ 11 * sF) * (14 * lX) := by
    have n0 : (0 : ℝ) ≤ 2 ^ 93 := by norm_num
    have n1 : 0 ≤ 8 * Q * (2 ^ 93 * (ρ * sP)) :=
      mul_nonneg (by linarith) (mul_nonneg n0 (mul_nonneg (by linarith) hsP0))
    have n2 : 0 ≤ 2 ^ 11 * sF := mul_nonneg (by norm_num) hsF
    exact mul_le_mul (mul_le_mul (h2.trans h3) hF hF0 n1) hlg hlg0 (mul_nonneg n1 n2)
  have h5 : (8 * Q * (2 ^ 93 * (ρ * sP))) * (2 ^ 11 * sF) * (14 * lX) = 112 * 2 ^ 104 * W3 := by
    ring
  have e6 : (Q + 1) ^ 2 * F * B * lg = (Q + 1) ^ 2 * B * F * lg := by ring
  have hc0 : 0 ≤ (Q + 1) * (Q + 2) * C * (P + 2) := by
    have := mul_nonneg hQ1 hC0; have : 0 ≤ P + 2 := by linarith
    positivity
  have hs0 : 0 ≤ (Q + 1) ^ 2 * K₁ * F * lg := by positivity
  have hr0 : 0 ≤ (Q + 1) * Q * (Q + 2) ^ 2 * A * F := by
    have : 0 ≤ (Q + 1) * Q := by nlinarith only [hQ]
    positivity
  have hh0 : 0 ≤ (Q + 1) ^ 2 * B * F * lg := by positivity
  rw [e6]
  refine ⟨by linarith, by linarith, ?_⟩
  linarith

/-- `shTol ≤ 2⁹ (|Σ| V)¹⁸`, `V = X log(1/δ)/(ξps)`. -/
lemma tc_tolPow : 1 ≤ X * log (1 / δ) / (ξ * p * s) ∧ 0 ≤ shTol Q A X δ ξ p s
    ∧ shTol Q A X δ ξ p s ≤ 2 ^ 9 * (A * (X * log (1 / δ) / (ξ * p * s))) ^ 18 := by
  have hX := h.X1; have hQ := h.hQ; have hA := h.hA; have hδ := h.hδ
  have hξ := h.hξ; have hp := h.hp; have hs := h.hs
  have iξ := h.iξ; have ip := h.ip; have is := h.is2; have iδ := h.iδ
  have hι := h.ι1; have hZ := h.lZ1; have hρ := h.ρ1; have hl1 := h.lQA1
  have hlδ : 0.69 ≤ log (1 / δ) := by
    have := Real.log_le_log (by norm_num) iδ
    have := Real.log_two_gt_d9
    linarith
  have eT : shTol Q A X δ ξ p s = shρ X ξ * (Q * shι Q X ξ ^ 3 * shZ A X δ ξ p s
      * ((1 / s) ^ 6 + Q ^ 2 * (1 / s) ^ 4)
      + Q ^ 2 * shZ A X δ ξ p s * (1 / p) ^ 2 * (1 / s) ^ 2 + Q ^ 2 * A * log (Q * A / δ)) := by
    unfold shTol; field_simp
  have eZ : X * A / (δ * ξ * p * s) = A * (1 / δ) * (X / (ξ * p * s)) := by field_simp
  have eQ : Q * A / δ = Q * A * (1 / δ) := by ring
  have eV : X * log (1 / δ) / (ξ * p * s) = X / (ξ * p * s) * log (1 / δ) := by ring
  have eW : X / (ξ * p * s) = X * ((1 / ξ) * (1 / p) * (1 / s)) := by field_simp
  have hι0 : 0 ≤ shι Q X ξ := by linarith
  have eZl : shZ A X δ ξ p s = log A + log (1 / δ) + log (X / (ξ * p * s)) := by
    unfold shZ
    rw [eZ, Real.log_mul (by positivity) (by positivity), Real.log_mul (by positivity)
      (by positivity)]
  have eQl : log (Q * A / δ) = log Q + log A + log (1 / δ) := by
    rw [eQ, Real.log_mul (by positivity) (by positivity), Real.log_mul (by positivity)
      (by positivity)]
  have eρ : shρ X ξ = 1 / ξ + log (X + 1) := rfl
  have eι : shι Q X ξ = (X + Q ^ 2) * (1 / ξ) + Q ^ 2 * log (X + 1) := by unfold shι; ring
  have lX := h.lX_le
  have lA : log A ≤ A := (Real.log_le_sub_one_of_pos (by linarith)).trans (by linarith)
  have lQ : log Q ≤ Q := (Real.log_le_sub_one_of_pos (by linarith)).trans (by linarith)
  rw [eV]
  set lδ := log (1 / δ)
  set W := X / (ξ * p * s) with hWd
  have k3 : 1 ≤ (1 / ξ) * (1 / p) := one_le_mul_of_one_le_of_one_le iξ ip
  have k4 : 2 ≤ (1 / ξ) * (1 / p) * (1 / s) := by
    have := mul_le_mul k3 is (by norm_num) (by linarith); linarith
  have hW2 : 2 * X ≤ W := by
    rw [eW]; have := mul_le_mul_of_nonneg_left k4 (by linarith : 0 ≤ X); linarith
  have hWξ : 1 / ξ ≤ W := by
    rw [eW]
    have : 1 ≤ X * (1 / p) * (1 / s) := by
      have := one_le_mul_of_one_le_of_one_le hX ip
      exact one_le_mul_of_one_le_of_one_le this (by linarith)
    rw [show X * ((1 / ξ) * (1 / p) * (1 / s)) = (1 / ξ) * (X * (1 / p) * (1 / s)) by ring]
    exact le_mul_of_one_le_right (by positivity) this
  have hWp : 1 / p ≤ W := by
    rw [eW]
    have : 1 ≤ X * (1 / ξ) * (1 / s) := by
      have := one_le_mul_of_one_le_of_one_le hX iξ
      exact one_le_mul_of_one_le_of_one_le this (by linarith)
    rw [show X * ((1 / ξ) * (1 / p) * (1 / s)) = (1 / p) * (X * (1 / ξ) * (1 / s)) by ring]
    exact le_mul_of_one_le_right (by positivity) this
  have hWs : 1 / s ≤ W := by
    rw [eW]
    have : 1 ≤ X * (1 / ξ) * (1 / p) := by
      have := one_le_mul_of_one_le_of_one_le hX iξ
      exact one_le_mul_of_one_le_of_one_le this ip
    rw [show X * ((1 / ξ) * (1 / p) * (1 / s)) = (1 / s) * (X * (1 / ξ) * (1 / p)) by ring]
    exact le_mul_of_one_le_right (by positivity) this
  have hW1 : 1 ≤ W := by linarith
  have hlW : log W ≤ W := (Real.log_le_sub_one_of_pos (by linarith)).trans (by linarith)
  have hV : 0.69 * W ≤ W * lδ := by
    have := mul_le_mul_of_nonneg_left hlδ (by linarith : 0 ≤ W); linarith
  have hV1 : 1 ≤ W * lδ := by linarith
  have hlV : lδ ≤ W * lδ := le_mul_of_one_le_left (by linarith) hW1
  set V := W * lδ with hVd
  set U := A * V with hUd
  have hVU : 2 * V ≤ U := by
    rw [hUd]; have := mul_le_mul_of_nonneg_right hA (by linarith : 0 ≤ V); linarith
  have hAU : A ≤ U := by rw [hUd]; exact le_mul_of_one_le_right (by linarith) hV1
  have hWU : W ≤ U := by linarith
  have hU1 : 1 ≤ U := by linarith
  have bQ : Q ≤ U := by linarith [h.hQX]
  have bρ : shρ X ξ ≤ 2 * U := by rw [eρ]; linarith
  have bι : shι Q X ξ ≤ 3 * U ^ 3 := by
    rw [eι]
    have hQW : Q ≤ W := by linarith [h.hQX]
    have a1 : (X + Q ^ 2) * (1 / ξ) ≤ (U + U ^ 2) * U :=
      mul_le_mul (add_le_add (by linarith) (pow_le_pow_left₀ (by linarith) (hQW.trans hWU) 2))
        (hWξ.trans hWU) (by positivity) (by positivity)
    have a2 : Q ^ 2 * log (X + 1) ≤ U ^ 2 * U :=
      mul_le_mul (pow_le_pow_left₀ (by linarith) (hQW.trans hWU) 2) (by linarith) h.lX0
        (by positivity)
    have : U ^ 2 ≤ U ^ 3 := pow_le_pow_right₀ hU1 (by norm_num)
    have e1 : (U + U ^ 2) * U = U ^ 2 + U ^ 3 := by ring
    have e2 : U ^ 2 * U = U ^ 3 := by ring
    linarith
  have bZ : shZ A X δ ξ p s ≤ 3 * U := by rw [eZl]; linarith
  have bQA : log (Q * A / δ) ≤ 3 * U := by rw [eQl]; linarith [h.hQX]
  have bs : 1 / s ≤ U := by linarith
  have bp : 1 / p ≤ U := by linarith
  refine ⟨hV1, by rw [eT]; positivity, ?_⟩
  rw [eT]
  have hZ0 : 0 ≤ shZ A X δ ξ p s := by linarith
  have hQA0 : 0 ≤ log (Q * A / δ) := by linarith
  calc shρ X ξ * (Q * shι Q X ξ ^ 3 * shZ A X δ ξ p s * ((1 / s) ^ 6 + Q ^ 2 * (1 / s) ^ 4)
        + Q ^ 2 * shZ A X δ ξ p s * (1 / p) ^ 2 * (1 / s) ^ 2 + Q ^ 2 * A * log (Q * A / δ))
      ≤ (2 * U) * (U * (3 * U ^ 3) ^ 3 * (3 * U) * (U ^ 6 + U ^ 2 * U ^ 4)
        + U ^ 2 * (3 * U) * U ^ 2 * U ^ 2 + U ^ 2 * U * (3 * U)) := by
        gcongr
    _ = 324 * U ^ 18 + 6 * U ^ 8 + 6 * U ^ 5 := by ring
    _ ≤ 2 ^ 9 * U ^ 18 := by
        have : U ^ 8 ≤ U ^ 18 := pow_le_pow_right₀ hU1 (by norm_num)
        have : U ^ 5 ≤ U ^ 18 := pow_le_pow_right₀ hU1 (by norm_num)
        have : 0 ≤ U ^ 18 := by positivity
        linarith

/-- The simple claim's prefix, suffix and reads shapes. -/
lemma tc_simple :
    (0 ≤ shPref Q A X δ ξ p s ∧ 0 ≤ Q ^ 3 * shι Q X ξ ^ 2 * shZ A X δ ξ p s / s ^ 4
      ∧ shPref Q A X δ ξ p s ≤ 4 * (Q ^ 3 * shι Q X ξ ^ 2 * shZ A X δ ξ p s / s ^ 4))
    ∧ (0 ≤ shSuff Q A X δ ξ p s ∧ 0 ≤ Q * shA A X δ ξ s / (p * s ^ 2)
      ∧ shSuff Q A X δ ξ p s ≤ 2 * (Q * shA A X δ ξ s / (p * s ^ 2)))
    ∧ (0 ≤ shReads Q A X δ ξ p s ∧ 0 ≤ Q ^ 3 * shι Q X ξ ^ 2 * shZ A X δ ξ p s / s ^ 4
          * (Q * shA A X δ ξ s / (p * s ^ 2) / Q + Q * shFam A X δ ξ s * log (X + 1) * shρ X ξ)
        + shProbe Q A X δ ξ * shFam A X δ ξ s * Q * log (X + 1)
      ∧ shReads Q A X δ ξ p s ≤ 8 * (Q ^ 3 * shι Q X ξ ^ 2 * shZ A X δ ξ p s / s ^ 4
          * (Q * shA A X δ ξ s / (p * s ^ 2) / Q + Q * shFam A X δ ξ s * log (X + 1) * shρ X ξ)
        + shProbe Q A X δ ξ * shFam A X δ ξ s * Q * log (X + 1))) := by
  have hX := h.X1; have hQ := h.hQ; have hA := h.hA; have hδ := h.hδ
  have hξ := h.hξ; have hp := h.hp; have hs := h.hs
  have hι := h.ι1; have hZ := h.lZ1; have hρ := h.ρ1; have hl1 := h.lQA1; have hA1 := h.lA1
  have hP := h.lP_le; have hP1 := h.lP1
  have hXι := h.Xι
  have hlX := h.lX0
  obtain ⟨hsP0, -, -, -⟩ := h.pref_ge
  have hs4 : s ^ 4 ≤ 1 := pow_le_one₀ hs.le (by linarith [h.hs2])
  set ι := shι Q X ξ
  set lZ := shZ A X δ ξ p s
  set sp := Q ^ 3 * ι ^ 2 * lZ / s ^ 4
  set ss := Q * shA A X δ ξ s / (p * s ^ 2)
  have hsp : 0 ≤ sp := by positivity
  have hss : 0 ≤ ss := by positivity
  -- prefixes
  have p1 : Q ^ 3 * ι ^ 2 / s ^ 4 * shP Q A X δ ξ p s ≤ 2 * sp := by
    have := mul_le_mul_of_nonneg_left hP (by positivity : (0 : ℝ) ≤ Q ^ 3 * ι ^ 2 / s ^ 4)
    have e : Q ^ 3 * ι ^ 2 / s ^ 4 * (2 * lZ) = 2 * sp := by ring
    linarith
  have p2 : X ^ 2 / (ξ ^ 2 * s ^ 4) ≤ ι ^ 2 / s ^ 4 := by
    rw [show X ^ 2 / (ξ ^ 2 * s ^ 4) = (X / ξ) ^ 2 / s ^ 4 by ring]
    exact div_le_div_of_nonneg_right (pow_le_pow_left₀ (by positivity) hXι 2) (by positivity)
  have p3 : ι ^ 2 ≤ ι ^ 2 / s ^ 4 := le_div_self (by positivity) (by positivity) hs4
  have p4 : Q ^ 2 * (X ^ 2 / (ξ ^ 2 * s ^ 4) + ι ^ 2) * lZ ≤ 2 * sp := by
    have k : X ^ 2 / (ξ ^ 2 * s ^ 4) + ι ^ 2 ≤ 2 * (ι ^ 2 / s ^ 4) := by linarith
    have k2 : Q ^ 2 ≤ Q ^ 3 := pow_le_pow_right₀ hQ (by norm_num)
    have k3 := mul_le_mul k2 k (by positivity) (by positivity)
    have k4 := mul_le_mul_of_nonneg_right k3 (by linarith : 0 ≤ lZ)
    have e : Q ^ 3 * (2 * (ι ^ 2 / s ^ 4)) * lZ = 2 * sp := by ring
    linarith
  have hPref : shPref Q A X δ ξ p s ≤ 4 * sp := by
    have e : shPref Q A X δ ξ p s = Q ^ 3 * ι ^ 2 / s ^ 4 * shP Q A X δ ξ p s
        + Q ^ 2 * (X ^ 2 / (ξ ^ 2 * s ^ 4) + ι ^ 2) * lZ := rfl
    linarith
  -- suffixes
  have hSuff : shSuff Q A X δ ξ p s ≤ 2 * ss := by
    have := h.lQp
    have e : shSuff Q A X δ ξ p s = Q * (shA A X δ ξ s / (p * s ^ 2) + log (Q / δ) / p) := rfl
    have k := mul_le_mul_of_nonneg_left (show shA A X δ ξ s / (p * s ^ 2) + log (Q / δ) / p
      ≤ 2 * (shA A X δ ξ s / (p * s ^ 2)) by linarith) (by linarith : 0 ≤ Q)
    have e2 : Q * (2 * (shA A X δ ξ s / (p * s ^ 2))) = 2 * ss := by ring
    linarith
  have hSuff0 : 0 ≤ shSuff Q A X δ ξ p s := by
    have := h.lQ0
    unfold shSuff; positivity
  -- reads
  have hF : 0 ≤ shFam A X δ ξ s := by unfold shFam; positivity
  have hPr : 0 ≤ shProbe Q A X δ ξ := by unfold shProbe; positivity
  set sF := shFam A X δ ξ s
  set ρ := shρ X ξ
  set lX := log (X + 1)
  set sP := shPref Q A X δ ξ p s
  set sS := shSuff Q A X δ ξ p s
  set pr := shProbe Q A X δ ξ
  have r1 : sP * sS / Q ≤ 8 * (sp * (ss / Q)) := by
    have k := mul_le_mul hPref hSuff hSuff0 (by positivity)
    have := div_le_div_of_nonneg_right k (by linarith : (0 : ℝ) ≤ Q)
    have e : 4 * sp * (2 * ss) / Q = 8 * (sp * (ss / Q)) := by ring
    linarith
  have r3 : Q * sP * sF * lX * ρ ≤ 4 * (sp * (Q * sF * lX * ρ)) := by
    have k := mul_le_mul_of_nonneg_left hPref (by linarith : (0 : ℝ) ≤ Q)
    have := mul_le_mul_of_nonneg_right k (by positivity : (0 : ℝ) ≤ sF * lX * ρ)
    have e1 : Q * sP * sF * lX * ρ = Q * sP * (sF * lX * ρ) := by ring
    have e2 : Q * (4 * sp) * (sF * lX * ρ) = 4 * (sp * (Q * sF * lX * ρ)) := by ring
    linarith
  have hR : shReads Q A X δ ξ p s = sP * sS / Q + pr * sF * Q * lX + Q * sP * sF * lX * ρ := rfl
  have q1 : 0 ≤ sp * (ss / Q) := by positivity
  have q2 : 0 ≤ sp * (Q * sF * lX * ρ) := by positivity
  have q3 : 0 ≤ pr * sF * Q * lX := by positivity
  have e3 : sp * (ss / Q + Q * sF * lX * ρ) + pr * sF * Q * lX
      = sp * (ss / Q) + sp * (Q * sF * lX * ρ) + pr * sF * Q * lX := by ring
  have q4 : 0 ≤ sP * sS / Q := by have := hSuff0; positivity
  have q5 : 0 ≤ Q * sP * sF * lX * ρ := by positivity
  refine ⟨⟨hsP0, hsp, hPref⟩, ⟨hSuff0, hss, hSuff⟩, ⟨by rw [hR]; linarith, by rw [e3]; linarith,
    ?_⟩⟩
  rw [hR, e3]
  linarith

end Totals

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



end OrthoDFA

end
