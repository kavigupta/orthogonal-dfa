import RLStar.LowLevelClaims.AdaptiveHits
import RLStar.Proofs.SuccessiveDraws
import Mathlib.Analysis.Complex.ExponentialBounds

/-!
# Adaptive hits against a binomial's lower tail

Where every set is heavy, swapping the light ones for the whole space changes no hit, and
`hitCount_lower` bounds the swapped count. The tail itself is bounded through the generating
function at `1/2`:

    P(Bin(n, q) ≤ c) ≤ 2^c (1 − q/2)^n ≤ exp(c log 2 − n q / 2).
-/

namespace OrthoDFA

open MeasureTheory

lemma adaptiveHits_eq_hitCount {S : Type*} : ∀ {n : ℕ} (A : List S → Set S) (y : Fin n → S),
    adaptiveHits A y = hitCount A (List.ofFn y)
  | 0, A, y => by simp [adaptiveHits, hitCount]
  | n + 1, A, y => by
    rw [List.ofFn_succ, hitCount, ← adaptiveHits_eq_hitCount (fun l' => A (y 0 :: l'))
      (fun j : Fin n => y j.succ)]
    simp only [adaptiveHits, Finset.card_filter, Fin.sum_univ_succ, Fin.val_zero, List.take_zero,
      Fin.val_succ]
    congr 1
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [List.ofFn_succ]
    classical
    exact if_congr (by rw [List.take_succ_cons]) rfl rfl

/-- P(Bin(n, q) ≤ c) ≤ exp(−n q / 8) for c ≤ n q / 2. -/
theorem one_sub_binomSfGe_le_exp_half (n c : ℕ) {q : ℝ} (hq0 : 0 ≤ q) (hq1 : q ≤ 1)
    (hc : (c : ℝ) ≤ n * q / 2) : 1 - binomSfGe n q (c + 1) ≤ Real.exp (-(n * q / 8)) := by
  set t : ℕ → ℝ := fun i => (n.choose i : ℝ) * q ^ i * (1 - q) ^ (n - i) with ht
  have h1q : 0 ≤ 1 - q := by linarith
  have ht0 : ∀ i, 0 ≤ t i := fun i => by positivity
  have hlow : 1 - binomSfGe n q (c + 1)
      = ∑ i ∈ Finset.range (n + 1), if i ≤ c then t i else 0 := by
    rw [← binomSfGe_zero_right n q, binomSfGe_eq_range, binomSfGe_eq_range,
      ← Finset.sum_sub_distrib]
    refine Finset.sum_congr rfl fun i _ => ?_
    by_cases h : i ≤ c
    · simp [ht, h, show ¬ c + 1 ≤ i by omega]
    · simp [h, show c + 1 ≤ i by omega]
  have hmgf : (∑ i ∈ Finset.range (n + 1), if i ≤ c then t i else 0)
      ≤ 2 ^ c * (1 - q / 2) ^ n := by
    calc (∑ i ∈ Finset.range (n + 1), if i ≤ c then t i else 0)
        ≤ ∑ i ∈ Finset.range (n + 1), 2 ^ c * ((1 / 2) ^ i * t i) := by
          refine Finset.sum_le_sum fun i _ => ?_
          by_cases h : i ≤ c
          · rw [if_pos h, ← mul_assoc]
            have h2 : (1 : ℝ) ≤ 2 ^ c * (1 / 2) ^ i := by
              rw [one_div_pow, ← div_eq_mul_one_div,
                one_le_div (by positivity)]
              exact pow_le_pow_right₀ (by norm_num) h
            nlinarith [ht0 i]
          · rw [if_neg h]
            have := ht0 i
            positivity
      _ = 2 ^ c * (1 - q / 2) ^ n := by
          rw [← Finset.mul_sum, show 1 - q / 2 = q / 2 + (1 - q) by ring, add_pow]
          congr 1
          refine Finset.sum_congr rfl fun i _ => ?_
          simp only [ht]
          ring
  have hexp : (1 - q / 2) ^ n ≤ Real.exp (n * (-(q / 2))) := by
    rw [Real.exp_nat_mul]
    exact pow_le_pow_left₀ (by linarith) (by linarith [Real.add_one_le_exp (-(q / 2))]) n
  have htwo : (2 : ℝ) ^ c = Real.exp (c * Real.log 2) := by
    rw [Real.exp_nat_mul, Real.exp_log (by norm_num)]
  have hlog : c * Real.log 2 ≤ n * q / 2 * (3 / 4) := by
    have hl0 : 0 ≤ Real.log 2 := Real.log_nonneg (by norm_num)
    have hl1 : Real.log 2 ≤ 3 / 4 := by linarith [Real.log_two_lt_d9]
    have hc0 : (0 : ℝ) ≤ c := Nat.cast_nonneg c
    nlinarith
  calc 1 - binomSfGe n q (c + 1)
      ≤ 2 ^ c * (1 - q / 2) ^ n := hlow ▸ hmgf
    _ ≤ Real.exp (c * Real.log 2) * Real.exp (n * (-(q / 2))) := by
        rw [htwo]; exact mul_le_mul_of_nonneg_left hexp (Real.exp_pos _).le
    _ = Real.exp (c * Real.log 2 + n * (-(q / 2))) := (Real.exp_add _ _).symm
    _ ≤ Real.exp (-(n * q / 8)) := Real.exp_le_exp.2 (by linarith)

theorem adaptiveHitsLower_holds : AdaptiveHitsLower := by
  intro S _ _ _ D _ n c q A hq0 hq1
  classical
  refine ⟨?_, one_sub_binomSfGe_le_exp_half n c hq0 hq1⟩
  set A' : List S → Set S := fun l => if q ≤ D.real (A l) then A l else Set.univ with hA'
  have hheavy : ∀ l : List S, l.length < n → q ≤ D.real (A' l) := fun l _ => by
    simp only [hA']
    split_ifs with h
    · exact h
    · rw [probReal_univ]; exact hq1
  refine le_trans (measureReal_mono ?_ (measure_ne_top _ _))
    (hitCount_lower D hq0 hq1 n A' hheavy c)
  intro y hy
  obtain ⟨hy, hc⟩ := hy
  have hsame : adaptiveHits A' y = adaptiveHits A y := by
    simp only [adaptiveHits]
    congr 1
    refine Finset.filter_congr fun i _ => ?_
    simp only [hA', if_pos (hy i)]
  change hitCount A' (List.ofFn y) ≤ c
  rw [← adaptiveHits_eq_hitCount, hsame]
  exact hc

end OrthoDFA
