import RLStar.Proofs.BigO
import RLStar.Proofs.LoopDFAArith

/-!
# The trie tolerance's arithmetic

`trie_loop_tol`'s bounds at `ℓ = L₁ + 1`, `ε = c/(8 + 4c (2 + ⌈log₂ L₁⌉))`: the rates' inverses
`trieIε`, `trieIι`, `trieIτ`, `trieIa`, the counts they give, and the tolerance

    T = 1536 Iε (E₁ + log(16 (|Q| + 1) Kc/δ')) + B

with `B` the probe budget, against the rates and caps `trie_loop_core` needs, once `2⁴ T⁵ ≤ |Σ|^k`.
-/

noncomputable section

namespace OrthoDFA

open Real

/-- A bound on `1/ε`. -/
def trieIε (ℓ c : ℝ) : ℝ := 8 / c + 12 * (1 + Real.log ℓ)

/-- A bound on `1/ι`. -/
def trieIι (ℓ Q c ξ : ℝ) : ℝ := 8 * (10 / c + 8 / ξ) * ℓ + 4608 * Q * (Q + 1) * trieIε ℓ c

/-- A bound on `1/τ`. -/
def trieIτ (ℓ Q c ξ η₀ : ℝ) : ℝ :=
  (10 / c + 8 / ξ) * ℓ / (2 * (1 / 2 - η₀) ^ 2) + trieIι ℓ Q c ξ

/-- A bound on `1/min (εcov s/2) ι`. -/
def trieIa (ℓ Q c ξ η₀ : ℝ) : ℝ := 2 * (10 / c + 8 / ξ) * ℓ / (1 / 2 - η₀) + trieIι ℓ Q c ξ

/-- Bounds the suffix pool `PB`. -/
def triePool (ℓ Q A δ c ξ pAP η₀ : ℝ) : ℝ :=
  let δ' := δ / (8 * (Q + 1))
  let s := 1 / 2 - η₀
  (50 * (max (Real.log (20 * trieIτ ℓ Q c ξ η₀)) (Real.log (8 * loopW₀ Q A δ / δ')) + 3) / s ^ 2
      + 20 / s + 28) / pAP
    + 120 * Real.log (3 / (δ' / 2)) / pAP

/-- Bounds the family size `FV`. -/
def trieFam (ℓ Q A δ c ξ η₀ : ℝ) : ℝ :=
  let δ' := δ / (8 * (Q + 1))
  let s := 1 / 2 - η₀
  (50 * (max (Real.log (20 * trieIτ ℓ Q c ξ η₀)) (Real.log (8 * loopW₀ Q A δ / δ')) + 3) / s ^ 2
      + 20 / s + 28) / 4

/-- Bounds the prefix count `RB`. -/
def trieCount (ℓ Q A δ c ξ pAP η₀ : ℝ) : ℝ :=
  let δ' := δ / (8 * (Q + 1))
  let P := triePool ℓ Q A δ c ξ pAP η₀
  2 ^ 22 * (Q + 2) * trieIι ℓ Q c ξ ^ 2 * Real.log (128 * (P + 1) / δ') / (1 / 2 - η₀) ^ 4
    + 2 ^ 27 * trieIτ ℓ Q c ξ η₀ ^ 2
      * Real.log (16 * (Q + 4) * P * trieIτ ℓ Q c ξ η₀ / (δ' ^ 2 * pAP))

/-- Bounds the probe budget `B₃`. -/
def trieBudget (ℓ Q A δ c : ℝ) : ℝ :=
  let S := Q * ((Q + 2) * A + 1) + 2 * A + 1
  36 * S * trieIε ℓ c * Real.log (16 * S * (S + 1) / (δ / (8 * (Q + 1)))) + S + 1

/-- The draws a harvested population makes in a round. -/
def trieSH (ℓ Q A δ c ξ pAP η₀ : ℝ) : ℝ :=
  24 * trieCount ℓ Q A δ c ξ pAP η₀ + 16 * Real.log (16 * (Q + 2) / (δ / (8 * (Q + 1)))) + 2

/-- The probes a round's harvested draws walk. -/
def trieHB (ℓ Q A δ c ξ pAP η₀ : ℝ) : ℝ :=
  384 * trieIε ℓ c * ((Q + 1) * trieSH ℓ Q A δ c ξ pAP η₀ * Real.log 2
    + Real.log (2 / (δ / (8 * (Q + 1))))) + 1

/-- A walk start's prefix mass of at most `1/(2⁴ trieTolInv⁵)` meets the trie loop's conditions;
`nQ` and `nA` stand for `|Q|` and `|Σ|` as exponents. -/
def trieTolInv (ℓ Q A : ℝ) (nQ nA : ℕ) (δ c ξ pAP η₀ : ℝ) : ℝ :=
  let δ' := δ / (8 * (Q + 1))
  let C := trieCount ℓ Q A δ c ξ pAP η₀
  let P := triePool ℓ Q A δ c ξ pAP η₀
  let E₁ := 2 ^ 20 * trieIa ℓ Q c ξ η₀ * (C + 1) / (1 / 2 - η₀)
    + (Q + 3) ^ 2 * (2 ^ 24 * trieIι ℓ Q c ξ * (C + 1)
      + 2 ^ 28 * trieIι ℓ Q c ξ ^ 2
        * Real.log (128 * (Q + 2) * (P + 2) * (13 + 2 * Real.logb 2 (C + 1)) / δ')
      + 2 ^ 18 * (P + 2) / pAP)
  let Kc := (1 + (Q + 1) ^ 3 * A) ^ nQ * (Q + 3) ^ ((nQ + 2) * nA)
  1536 * trieIε ℓ c * E₁ + 1536 * Real.log (16 * (Q + 1) * Kc / δ') * trieIε ℓ c
    + trieBudget ℓ Q A δ c

lemma clog_le_log {n : ℕ} (hn : 1 < n) :
    2 + (Nat.clog 2 n : ℝ) ≤ 3 * (1 + Real.log ((n : ℝ) + 1)) := by
  have h1 := Nat.pow_pred_clog_lt_self (by norm_num : 1 < 2) hn
  set c := Nat.clog 2 n
  have hc : c ≤ (c - 1) + 1 := by omega
  have hc' : (c : ℝ) ≤ ((c - 1 : ℕ) : ℝ) + 1 := by exact_mod_cast hc
  have h2 : (2 : ℝ) ^ (c - 1) < (n : ℝ) := by
    rw [Nat.pred_eq_sub_one] at h1
    exact_mod_cast h1
  have hn0 : (0 : ℝ) < n := by positivity
  have h3 : ((c - 1 : ℕ) : ℝ) * Real.log 2 < Real.log ((n : ℝ) + 1) := by
    rw [← Real.log_pow]
    exact Real.log_lt_log (by positivity) (by linarith)
  have hn2 : (2 : ℝ) ≤ n := by exact_mod_cast hn
  have hl : Real.log 2 ≤ Real.log ((n : ℝ) + 1) := Real.log_le_log (by norm_num) (by linarith)
  have hl2 := Real.log_two_gt_d9
  have hc0 : (0 : ℝ) ≤ c := Nat.cast_nonneg _
  have h4 : (c : ℝ) * Real.log 2 ≤ (((c - 1 : ℕ) : ℝ) + 1) * Real.log 2 :=
    mul_le_mul_of_nonneg_right hc' (by linarith)
  have h5 : (c : ℝ) * 0.6931471803 ≤ c * Real.log 2 := mul_le_mul_of_nonneg_left hl2.le hc0
  nlinarith

/-- The rates' inverses, from `ε = c/(8 + 4cλ)` with `λ ≤ 3 (1 + log ℓ)`. -/
lemma trie_rate_bounds {Q ℓ N lam c ξ η₀ ε εcov ι τ s : ℝ} (hQ : 1 ≤ Q) (hℓ : 1 ≤ ℓ)
    (hN1 : 1 ≤ N) (hNℓ : N ≤ ℓ) (hlam0 : 0 ≤ lam) (hlam : lam ≤ 3 * (1 + Real.log ℓ))
    (hc : 0 < c) (hc1 : c ≤ 1) (hξ : 0 < ξ) (hη0 : 0 ≤ η₀) (hη : η₀ < 1 / 2)
    (he : ε = c / (8 + 4 * c * lam)) (hεc : εcov = min (c / (10 * N)) (ξ / (8 * N)))
    (hι : ι = min (εcov / 8) (ε / (4608 * Q * (Q + 1)))) (hs : s = 1 / 2 - η₀)
    (hτ : τ = min ((εcov / 2) * (1 - 2 * η₀) ^ 2) ι) :
    0 < ε ∧ 1 / ε ≤ trieIε ℓ c ∧ 0 < ι ∧ 1 / ι ≤ trieIι ℓ Q c ξ ∧ 0 < τ
      ∧ 1 / τ ≤ trieIτ ℓ Q c ξ η₀ ∧ 1 / min (εcov / 2 * s) ι ≤ trieIa ℓ Q c ξ η₀
      ∧ 1 ≤ trieIε ℓ c ∧ 1 ≤ trieIι ℓ Q c ξ ∧ trieIι ℓ Q c ξ ≤ trieIτ ℓ Q c ξ η₀
      ∧ trieIι ℓ Q c ξ ≤ trieIa ℓ Q c ξ η₀ ∧ ℓ ≤ trieIa ℓ Q c ξ η₀ := by
  have hs0 : 0 < s := by rw [hs]; linarith
  have hs1 : s ≤ 1 / 2 := by rw [hs]; linarith
  have hlℓ : 0 ≤ Real.log ℓ := Real.log_nonneg hℓ
  have hε0 : 0 < ε := by rw [he]; exact div_pos hc (by positivity)
  have hIε1 : 8 ≤ trieIε ℓ c := by
    unfold trieIε
    have : 8 ≤ 8 / c := by rw [le_div_iff₀ hc]; linarith
    linarith
  have hinvε : 1 / ε ≤ trieIε ℓ c := by
    rw [he, one_div_div]
    have e : (8 + 4 * c * lam) / c = 8 / c + 4 * lam := by field_simp
    rw [e]; unfold trieIε; linarith
  set Ce := 10 / c + 8 / ξ with hCe
  have hCe0 : 0 < Ce := by positivity
  have h10 : 10 ≤ Ce := by
    have : 10 ≤ 10 / c := by rw [le_div_iff₀ hc]; linarith
    have : 0 ≤ 8 / ξ := by positivity
    linarith
  have hεc0 : 0 < εcov := by rw [hεc]; exact lt_min (by positivity) (by positivity)
  have hinvεc : 1 / εcov ≤ Ce * ℓ := by
    rw [hεc]
    rcases min_choice (c / (10 * N)) (ξ / (8 * N)) with h | h <;> rw [h, one_div_div]
    · calc 10 * N / c ≤ 10 * ℓ / c := by gcongr
        _ = 10 / c * ℓ := by ring
        _ ≤ Ce * ℓ := by
          have : 10 / c ≤ Ce := by have : 0 ≤ 8 / ξ := by positivity
                                   linarith
          exact mul_le_mul_of_nonneg_right this (by linarith)
    · calc 8 * N / ξ ≤ 8 * ℓ / ξ := by gcongr
        _ = 8 / ξ * ℓ := by ring
        _ ≤ Ce * ℓ := by
          have : 8 / ξ ≤ Ce := by have : 0 ≤ 10 / c := by positivity
                                  linarith
          exact mul_le_mul_of_nonneg_right this (by linarith)
  have hQQ : 0 < 4608 * Q * (Q + 1) := by positivity
  have hι0 : 0 < ι := by rw [hι]; exact lt_min (by positivity) (by positivity)
  have hIι : trieIι ℓ Q c ξ = 8 * Ce * ℓ + 4608 * Q * (Q + 1) * trieIε ℓ c := rfl
  have hinvι : 1 / ι ≤ trieIι ℓ Q c ξ := by
    rw [hι]
    refine (one_div_min_le (by positivity) (by positivity)).trans ?_
    rw [one_div_div, one_div_div, hIι]
    have h1 : 8 / εcov ≤ 8 * Ce * ℓ := by
      have := mul_le_mul_of_nonneg_left hinvεc (by norm_num : (0 : ℝ) ≤ 8)
      rw [show 8 / εcov = 8 * (1 / εcov) by ring]; linarith
    have h2 : 4608 * Q * (Q + 1) / ε ≤ 4608 * Q * (Q + 1) * trieIε ℓ c := by
      rw [show 4608 * Q * (Q + 1) / ε = 4608 * Q * (Q + 1) * (1 / ε) by ring]
      exact mul_le_mul_of_nonneg_left hinvε hQQ.le
    linarith
  have ha : 2 * s ≤ 1 - 2 * η₀ := by rw [hs]; linarith
  have hτ0 : 0 < τ := by
    rw [hτ]
    exact lt_min (mul_pos (by positivity) (pow_pos (by linarith) 2)) hι0
  have hIι0 : 0 ≤ 8 * Ce * ℓ := by positivity
  have hIιτ : trieIι ℓ Q c ξ ≤ trieIτ ℓ Q c ξ η₀ := by
    unfold trieIτ
    have : 0 ≤ (10 / c + 8 / ξ) * ℓ / (2 * (1 / 2 - η₀) ^ 2) := by
      have : 0 < 1 / 2 - η₀ := by linarith
      positivity
    linarith
  have hIιa : trieIι ℓ Q c ξ ≤ trieIa ℓ Q c ξ η₀ := by
    unfold trieIa
    have : 0 ≤ 2 * (10 / c + 8 / ξ) * ℓ / (1 / 2 - η₀) := by
      have : 0 < 1 / 2 - η₀ := by linarith
      positivity
    linarith
  have hinvτ : 1 / τ ≤ trieIτ ℓ Q c ξ η₀ := by
    rw [hτ]
    have h12 : 0 < 1 - 2 * η₀ := by linarith
    refine (one_div_min_le (by positivity) hι0).trans ?_
    have hsq : (2 * s) ^ 2 ≤ (1 - 2 * η₀) ^ 2 := pow_le_pow_left₀ (by positivity) ha 2
    have hlow : 2 * s ^ 2 * εcov ≤ (εcov / 2) * (1 - 2 * η₀) ^ 2 := by
      have := mul_le_mul_of_nonneg_left hsq (show 0 ≤ εcov / 2 by positivity)
      nlinarith
    have h1 : 1 / ((εcov / 2) * (1 - 2 * η₀) ^ 2) ≤ Ce * ℓ / (2 * s ^ 2) :=
      calc 1 / ((εcov / 2) * (1 - 2 * η₀) ^ 2) ≤ 1 / (2 * s ^ 2 * εcov) :=
            one_div_le_one_div_of_le (by positivity) hlow
        _ = 1 / (2 * s ^ 2) * (1 / εcov) := by field_simp
        _ ≤ 1 / (2 * s ^ 2) * (Ce * ℓ) := mul_le_mul_of_nonneg_left hinvεc (by positivity)
        _ = Ce * ℓ / (2 * s ^ 2) := by ring
    unfold trieIτ
    rw [← hCe, ← hs]
    linarith
  have hinvA : 1 / min (εcov / 2 * s) ι ≤ trieIa ℓ Q c ξ η₀ := by
    refine (one_div_min_le (by positivity) hι0).trans ?_
    have h1 : 1 / (εcov / 2 * s) ≤ 2 * Ce * ℓ / s :=
      calc 1 / (εcov / 2 * s) = 2 / s * (1 / εcov) := by field_simp
        _ ≤ 2 / s * (Ce * ℓ) := mul_le_mul_of_nonneg_left hinvεc (by positivity)
        _ = 2 * Ce * ℓ / s := by ring
    unfold trieIa
    rw [← hCe, ← hs]
    linarith
  have hIι1 : 1 ≤ trieIι ℓ Q c ξ := by
    rw [hIι]
    have : 0 ≤ 4608 * Q * (Q + 1) * trieIε ℓ c := by
      have : 0 ≤ trieIε ℓ c := by linarith
      positivity
    nlinarith
  have hℓa : ℓ ≤ trieIa ℓ Q c ξ η₀ := by
    unfold trieIa
    rw [← hCe, ← hs]
    have h1 : ℓ ≤ 2 * Ce * ℓ / s := by
      rw [le_div_iff₀ hs0]
      nlinarith
    have : 0 ≤ trieIι ℓ Q c ξ := by linarith
    linarith
  exact ⟨hε0, hinvε, hι0, hinvι, hτ0, hinvτ, hinvA, by linarith, hIι1, hIιτ, hIιa, hℓa⟩

lemma tol_T_le {E₁ aε lg K₁ T : ℝ} (haε : 1 ≤ aε) (hE₁ : 1 ≤ E₁) (hlg : 0 ≤ lg)
    (hK₁ : 0 ≤ K₁) (hT : T = 1536 * aε * E₁ + 1536 * lg * aε + K₁) :
    1 ≤ T ∧ E₁ ≤ T ∧ aε ≤ T ∧ lg ≤ T ∧ K₁ ≤ T ∧ 1536 * aε * E₁ + 1536 * lg * aε ≤ T := by
  have ha0 : 0 ≤ aε := by linarith only [haε]
  have h1 : E₁ ≤ aε * E₁ := le_mul_of_one_le_left (by linarith only [hE₁]) haε
  have h2 : aε ≤ aε * E₁ := le_mul_of_one_le_right ha0 hE₁
  have h3 : lg ≤ lg * aε := le_mul_of_one_le_right hlg haε
  have h4 : 0 ≤ lg * aε := mul_nonneg hlg ha0
  have h5 : 0 ≤ aε * E₁ := by linarith only [h2, ha0]
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩ <;> rw [hT] <;>
    nlinarith only [h1, h2, h3, h4, h5, haε, hE₁, hK₁]

/-- The walk starts' `B₃ ≤ T`, so `T⁵` pays their mass. -/
lemma tol_B₃ {B₃ T Ak : ℝ} (hB₃0 : 0 ≤ B₃) (hBT : B₃ ≤ T) (hT1 : 1 ≤ T)
    (hAk : 2 ^ 4 * T ^ 5 ≤ Ak) :
    16 * B₃ ^ 2 * Ak⁻¹ ≤ 1 := by
  have h2 : B₃ ^ 2 ≤ T ^ 2 := pow_le_pow_left₀ hB₃0 hBT 2
  have h5 : T ^ 2 ≤ T ^ 5 := pow_le_pow_right₀ hT1 (by norm_num)
  have hAk0 : 0 < Ak := lt_of_lt_of_le (by positivity) hAk
  rw [← div_eq_mul_inv, div_le_one hAk0]
  linarith only [h2, h5, hAk]

lemma tol_E₁_ge {Q R₀ Θι Θa P₀ pAP s l16 l128 E₁ : ℝ} (hR₀ : 0 ≤ R₀) (hΘι : 1 ≤ Θι)
    (hΘa : 1 ≤ Θa) (hP₀ : 1 ≤ P₀) (hp : 0 < pAP) (hp1 : pAP ≤ 1) (hs : 0 < s) (hs1 : s ≤ 1)
    (hl16 : 0 ≤ l16) (hl : l16 ≤ l128)
    (hE₁ : E₁ = 2 ^ 20 * Θa * (R₀ + 1) / s + (Q + 3) ^ 2 * (2 ^ 24 * Θι * (R₀ + 1)
      + 2 ^ 28 * Θι ^ 2 * l128 + 2 ^ 18 * (P₀ + 2) / pAP)) :
    (Q + 3) ^ 2 * (2 ^ 24 * (R₀ + 1) + 2 ^ 28 * l16 + (P₀ + 2)) ≤ E₁
      ∧ Θa ≤ E₁ := by
  have hP2 : 0 ≤ P₀ + 2 := by linarith only [hP₀]
  have hR1 : 0 ≤ R₀ + 1 := by linarith only [hR₀]
  have a1 := mul_le_mul_of_nonneg_right hΘι hR1
  have a2 := mul_le_mul (one_le_pow₀ hΘι : (1 : ℝ) ≤ Θι ^ 2) hl hl16 (sq_nonneg Θι)
  have a3 : P₀ + 2 ≤ 2 ^ 18 * (P₀ + 2) / pAP := by
    rw [le_div_iff₀ (by positivity)]
    have h5 : pAP ≤ 2 ^ 18 := by linarith only [hp1]
    nlinarith only [mul_le_mul_of_nonneg_left h5 hP2]
  have a0 : 0 ≤ 2 ^ 20 * Θa * (R₀ + 1) / s :=
    div_nonneg (mul_nonneg (mul_nonneg (by norm_num) (by linarith only [hΘa])) hR1) hs.le
  have hsum : 2 ^ 24 * (R₀ + 1) + 2 ^ 28 * l16 + (P₀ + 2)
      ≤ 2 ^ 24 * Θι * (R₀ + 1) + 2 ^ 28 * Θι ^ 2 * l128 + 2 ^ 18 * (P₀ + 2) / pAP := by
    linarith only [a1, a2, a3]
  have := mul_le_mul_of_nonneg_left hsum (sq_nonneg (Q + 3))
  have hb0 : 0 ≤ (Q + 3) ^ 2 * (2 ^ 24 * (R₀ + 1) + 2 ^ 28 * l16 + (P₀ + 2)) := by positivity
  refine ⟨by rw [hE₁]; linarith only [this, a0], ?_⟩
  have hΘa' : Θa ≤ 2 ^ 20 * Θa * (R₀ + 1) / s := by
    rw [le_div_iff₀ hs]
    have h1 : Θa * s ≤ Θa * 1 := mul_le_mul_of_nonneg_left hs1 (by linarith only [hΘa])
    nlinarith only [h1, hΘa, hR₀, mul_nonneg (by linarith only [hΘa] : (0 : ℝ) ≤ Θa) hR₀]
  rw [hE₁]; linarith only [hΘa', this, hb0]

lemma tol_FV {X pAP lp PB P E₁ T : ℝ} (hX : 0 ≤ X) (hp : 0 < pAP) (hp1 : pAP ≤ 1)
    (hlp : 0 ≤ lp) (hPBd : PB = X / pAP + lp) (hPB : PB ≤ P) (hPE : P ≤ E₁) (hT : E₁ ≤ T) :
    X / 4 ≤ T := by
  have h2 : X ≤ X / pAP := le_div_self hX hp hp1
  rw [hPBd] at hPB
  linarith only [h2, hlp, hPB, hPE, hT, hX]

/-- The family's misread tail, `F ≤ T` letters past a length-`L` cut. -/
lemma tol_k6 {A T c δ X : ℝ} {F Ls : ℕ} (hA : 2 ≤ A) (hT1 : 1 ≤ T) (hδ : 0 < δ)
    (hFT : (F : ℝ) ≤ T) (hAT : A ≤ T) (hLsT : (Ls : ℝ) + 1 ≤ 2 * T) (hcT : 1 ≤ c * T)
    (hδT : Real.log (16 / δ) ≤ 16 * T) (hAL : 2 ^ 4 * T ^ 5 ≤ X) :
    2 * ((F : ℝ) + 1) ^ 3 * (A ^ Ls + 1) ^ F * Real.exp (-2 * X * c ^ 2) ≤ δ / 8 := by
  have hT0 : 0 ≤ T := by linarith only [hT1]
  have hA0 : 0 < A := by linarith only [hA]
  have hA1 : 1 ≤ A := by linarith only [hA]
  have hF0 : (0 : ℝ) ≤ F := Nat.cast_nonneg _
  have e1 : ((F : ℝ) + 1) ^ 3 ≤ Real.exp (3 * F) := by
    calc ((F : ℝ) + 1) ^ 3 ≤ Real.exp F ^ 3 :=
          pow_le_pow_left₀ (by linarith only [hF0])
            (by linarith only [Real.add_one_le_exp (F : ℝ)]) 3
      _ = Real.exp (3 * F) := by rw [← Real.exp_nat_mul]; norm_num
  have e2 : (A ^ Ls + 1) ^ F ≤ Real.exp (A * ((Ls : ℝ) + 1) * F) := by
    have b1 : A ^ Ls + 1 ≤ A ^ (Ls + 1) := by
      have h1 : 1 ≤ A ^ Ls := one_le_pow₀ hA1
      rw [pow_succ]
      nlinarith only [mul_le_mul_of_nonneg_left hA (by linarith only [h1] : (0 : ℝ) ≤ A ^ Ls), h1]
    have b2 : A ^ (Ls + 1) ≤ Real.exp (A * ((Ls : ℝ) + 1)) := by
      calc A ^ (Ls + 1) ≤ Real.exp A ^ (Ls + 1) :=
            pow_le_pow_left₀ hA0.le (by linarith only [Real.add_one_le_exp A]) _
        _ = Real.exp (A * ((Ls : ℝ) + 1)) := by
            rw [← Real.exp_nat_mul]; push_cast; ring_nf
    calc (A ^ Ls + 1) ^ F ≤ Real.exp (A * ((Ls : ℝ) + 1)) ^ F :=
          pow_le_pow_left₀ (by positivity) (b1.trans b2) F
      _ = Real.exp (A * ((Ls : ℝ) + 1) * F) := by rw [← Real.exp_nat_mul]; ring_nf
  have hexp : 3 * (F : ℝ) + A * ((Ls : ℝ) + 1) * F + Real.log (16 / δ) ≤ 2 * X * c ^ 2 := by
    have l1 : Real.log (16 / δ) ≤ 16 * T := hδT
    have l2 : A * ((Ls : ℝ) + 1) * F ≤ T * (2 * T) * T :=
      mul_le_mul (mul_le_mul hAT hLsT (by positivity) hT0) hFT hF0
        (mul_nonneg hT0 (by linarith only [hT1]))
    have hT3 : T ≤ T ^ 3 := le_self_pow₀ hT1 (by norm_num)
    have l3 : 3 * (F : ℝ) + A * ((Ls : ℝ) + 1) * F + Real.log (16 / δ) ≤ 21 * T ^ 3 := by
      linarith only [l1, l2, hT3, hFT]
    have l4 : 2 ^ 5 * T ^ 3 ≤ 2 * X * c ^ 2 := by
      have h3 : T ^ 3 ≤ T ^ 5 * c ^ 2 := by
        have h' : 1 ≤ (c * T) ^ 2 := one_le_pow₀ hcT
        have e : T ^ 5 * c ^ 2 = T ^ 3 * (c * T) ^ 2 := by ring
        rw [e]; exact le_mul_of_one_le_right (pow_nonneg hT0 3) h'
      have := mul_le_mul_of_nonneg_right hAL (sq_nonneg c)
      linarith only [h3, this]
    linarith only [l3, l4, pow_nonneg hT0 3]
  have hlog : Real.exp (3 * F) * Real.exp (A * ((Ls : ℝ) + 1) * F)
      * Real.exp (-2 * X * c ^ 2) ≤ δ / 16 := by
    rw [← Real.exp_add, ← Real.exp_add]
    calc Real.exp (3 * F + A * ((Ls : ℝ) + 1) * F + -2 * X * c ^ 2)
        ≤ Real.exp (-Real.log (16 / δ)) := Real.exp_le_exp.2 (by linarith only [hexp])
      _ = δ / 16 := by
          rw [Real.exp_neg, Real.exp_log (by positivity), inv_div]
  have hE0 : 0 ≤ Real.exp (-2 * X * c ^ 2) := (Real.exp_pos _).le
  have hP : 0 ≤ (A ^ Ls + 1) ^ F := by positivity
  calc 2 * ((F : ℝ) + 1) ^ 3 * (A ^ Ls + 1) ^ F * Real.exp (-2 * X * c ^ 2)
      ≤ 2 * Real.exp (3 * F) * Real.exp (A * ((Ls : ℝ) + 1) * F)
        * Real.exp (-2 * X * c ^ 2) := by gcongr
    _ = 2 * (Real.exp (3 * F) * Real.exp (A * ((Ls : ℝ) + 1) * F)
        * Real.exp (-2 * X * c ^ 2)) := by ring
    _ ≤ 2 * (δ / 16) := by gcongr
    _ = δ / 8 := by ring

/-- `|Σ| ≤ log Kc`: `Kc` has `(|Q| + 3)^(2|Σ|)` as a factor. -/
lemma A_le_log_kc {Q : ℝ} {nQ nA : ℕ} (hQ : 0 ≤ Q) :
    (nA : ℝ) ≤ Real.log ((1 + (Q + 1) ^ 3 * nA) ^ nQ * (Q + 3) ^ ((nQ + 2) * nA)) := by
  have h1 : (1 : ℝ) ≤ (1 + (Q + 1) ^ 3 * nA) ^ nQ :=
    one_le_pow₀ (by have : (0 : ℝ) ≤ (Q + 1) ^ 3 * nA := by positivity
                    linarith)
  have h3 : (3 : ℝ) ^ (2 * nA) ≤ (Q + 3) ^ ((nQ + 2) * nA) :=
    (pow_le_pow_left₀ (by norm_num) (by linarith) _).trans
      (pow_le_pow_right₀ (by linarith) (Nat.mul_le_mul_right _ (by omega)))
  have h4 : (3 : ℝ) ^ (2 * nA) ≤ (1 + (Q + 1) ^ 3 * nA) ^ nQ * (Q + 3) ^ ((nQ + 2) * nA) :=
    h3.trans (le_mul_of_one_le_left (by positivity) h1)
  have h5 := Real.log_le_log (by positivity) h4
  rw [Real.log_pow] at h5
  have hl3 : 1 ≤ Real.log 3 := one_le_log_of_three_le le_rfl
  push_cast at h5
  nlinarith [Nat.cast_nonneg (α := ℝ) nA]

end OrthoDFA

end
