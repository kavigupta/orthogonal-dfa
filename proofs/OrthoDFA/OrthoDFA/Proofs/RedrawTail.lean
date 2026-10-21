import OrthoDFA.Proofs.SuccessiveDraws
import Mathlib.Analysis.Complex.ExponentialBounds
import Mathlib.Data.Nat.Choose.Bounds

/-!
# What a sampler's rejections cost

`rejTail n a r` is the chance that `n` draws, each rejected with chance at most `a` whatever came
before, are rejected `r + 1` times: `P[Bin(n, min a 1) ≥ r + 1]`. It is at most `n·a`, and once
`8na ≤ (r + 1)x` at most `(x/2)^(r+1)`.
-/

namespace OrthoDFA

open MeasureTheory

/-- `P[Bin(n, min a 1) ≥ r + 1]`. -/
noncomputable def rejTail (n : ℕ) (a : ℝ) (r : ℕ) : ℝ := binomSfGe n (min a 1) (r + 1)

/-- `P[Bin(n, t) ≥ j] ≤ C(n, j) tʲ`: some `j` of the draws all land. -/
theorem binomSfGe_le_choose {t : ℝ} (ht0 : 0 ≤ t) (ht1 : t ≤ 1) :
    ∀ n j, binomSfGe n t j ≤ (n.choose j : ℝ) * t ^ j
  | n, 0 => by rw [binomSfGe_zero_right]; simp
  | 0, j + 1 => by rw [binomSfGe_zero_left]; simp
  | n + 1, j + 1 => by
    rw [binomSfGe_succ, Nat.choose_succ_succ', Nat.cast_add, pow_succ]
    have h1 := binomSfGe_le_choose ht0 ht1 n j
    have h2 := binomSfGe_le_choose ht0 ht1 n (j + 1)
    have hB0 : 0 ≤ binomSfGe n t (j + 1) := binomSfGe_nonneg ht0 ht1 _
    have e1 : t * binomSfGe n t j ≤ t * ((n.choose j : ℝ) * t ^ j) :=
      mul_le_mul_of_nonneg_left h1 ht0
    have e2 : (1 - t) * binomSfGe n t (j + 1) ≤ binomSfGe n t (j + 1) := by nlinarith
    rw [pow_succ] at h2
    nlinarith

/-- More draws reach a count at least as often. -/
theorem binomSfGe_le_succ {t : ℝ} (ht0 : 0 ≤ t) (ht1 : t ≤ 1) (n : ℕ) :
    ∀ j, binomSfGe n t j ≤ binomSfGe (n + 1) t j
  | 0 => by rw [binomSfGe_zero_right, binomSfGe_zero_right]
  | j + 1 => by
    rw [binomSfGe_succ]
    have h := binomSfGe_antitone (n := n) ht0 ht1 j
    nlinarith

theorem binomSfGe_mono_n {t : ℝ} (ht0 : 0 ≤ t) (ht1 : t ≤ 1) {n n' : ℕ} (h : n ≤ n') (j : ℕ) :
    binomSfGe n t j ≤ binomSfGe n' t j := by
  induction n', h using Nat.le_induction with
  | base => exact le_rfl
  | succ k _ ih => exact ih.trans (binomSfGe_le_succ ht0 ht1 k j)

lemma rejTail_nonneg (n : ℕ) {a : ℝ} (ha : 0 ≤ a) (r : ℕ) : 0 ≤ rejTail n a r :=
  binomSfGe_nonneg (le_min ha zero_le_one) (min_le_right _ _) _

lemma rejTail_le_one (n : ℕ) {a : ℝ} (ha : 0 ≤ a) (r : ℕ) : rejTail n a r ≤ 1 := by
  unfold rejTail
  calc binomSfGe n (min a 1) (r + 1) ≤ binomSfGe n (min a 1) 0 :=
        binomSfGe_antitone' (le_min ha zero_le_one) (min_le_right _ _) (Nat.zero_le _)
    _ = 1 := binomSfGe_zero_right _ _

/-- More draws, or a larger chance, reject as often or more. -/
theorem rejTail_mono {n n' : ℕ} {a a' : ℝ} (ha : 0 ≤ a) (hn : n ≤ n') (haa : a ≤ a') (r : ℕ) :
    rejTail n a r ≤ rejTail n' a' r := by
  unfold rejTail
  have h0 : 0 ≤ min a 1 := le_min ha zero_le_one
  refine le_trans (binomSfGe_mono_n h0 (min_le_right _ _) hn _) ?_
  exact binomSfGe_mono h0 (min_le_right _ _) (min_le_min_right 1 haa) n' (r + 1)

/-- A rejection at all: at most `n·a`. -/
theorem rejTail_le_lin (n : ℕ) {a : ℝ} (ha : 0 ≤ a) (r : ℕ) : rejTail n a r ≤ n * a := by
  unfold rejTail
  have h0 : 0 ≤ min a 1 := le_min ha zero_le_one
  refine le_trans (binomSfGe_antitone' h0 (min_le_right _ _) (show 1 ≤ r + 1 by omega)) ?_
  refine le_trans (binomSfGe_le_choose h0 (min_le_right _ _) n 1) ?_
  rw [Nat.choose_one_right, pow_one]
  exact mul_le_mul_of_nonneg_left (min_le_left _ _) (Nat.cast_nonneg _)

/-- Many rejections are exponentially rare: with `8na ≤ (r + 1)x`, at most `(x/2)^(r+1)`. -/
theorem rejTail_le_pow {n : ℕ} {a x : ℝ} (ha : 0 ≤ a) (hx : 0 ≤ x) {r : ℕ}
    (h : 8 * (n * a) ≤ ((r : ℝ) + 1) * x) : rejTail n a r ≤ (x / 2) ^ (r + 1) := by
  unfold rejTail
  set j : ℕ := r + 1 with hj
  set t : ℝ := min a 1 with ht
  have ht0 : 0 ≤ t := le_min ha zero_le_one
  have ht1 : t ≤ 1 := min_le_right _ _
  have hjR : (j : ℝ) = (r : ℝ) + 1 := by rw [hj]; push_cast; ring
  have hj0 : (0 : ℝ) < j := by rw [hjR]; positivity
  refine le_trans (binomSfGe_le_choose ht0 ht1 n j) ?_
  have hch : (n.choose j : ℝ) ≤ (n : ℝ) ^ j / (j.factorial : ℝ) := Nat.choose_le_pow_div j n
  have hfac : (j : ℝ) ^ j / (j.factorial : ℝ) ≤ Real.exp j :=
    Real.pow_div_factorial_le_exp (j : ℝ) hj0.le j
  have hfpos : (0 : ℝ) < (j.factorial : ℝ) := by exact_mod_cast Nat.factorial_pos j
  have hnt : (n : ℝ) * t ≤ j * x / 8 := by
    have : (n : ℝ) * t ≤ n * a := mul_le_mul_of_nonneg_left (min_le_left _ _) (Nat.cast_nonneg _)
    rw [hjR]; linarith
  have hnt0 : 0 ≤ (n : ℝ) * t := by positivity
  calc (n.choose j : ℝ) * t ^ j ≤ (n : ℝ) ^ j / (j.factorial : ℝ) * t ^ j :=
        mul_le_mul_of_nonneg_right hch (pow_nonneg ht0 _)
    _ = ((n : ℝ) * t) ^ j / (j.factorial : ℝ) := by rw [mul_pow]; ring
    _ ≤ ((j : ℝ) * x / 8) ^ j / (j.factorial : ℝ) :=
        div_le_div_of_nonneg_right (pow_le_pow_left₀ hnt0 hnt j) hfpos.le
    _ = (x / 8) ^ j * ((j : ℝ) ^ j / (j.factorial : ℝ)) := by
        rw [show (j : ℝ) * x / 8 = (j : ℝ) * (x / 8) by ring, mul_pow]; ring
    _ ≤ (x / 8) ^ j * Real.exp j :=
        mul_le_mul_of_nonneg_left hfac (pow_nonneg (by positivity) _)
    _ = (x / 8 * Real.exp 1) ^ j := by
        rw [mul_pow, ← Real.exp_nat_mul, mul_one]
    _ ≤ (x / 2) ^ j := by
        refine pow_le_pow_left₀ (by positivity) ?_ j
        have he : Real.exp 1 ≤ 4 := by linarith [Real.exp_one_lt_d9]
        nlinarith

/-- A set of strings holds at most its size times the largest atom. -/
lemma measureReal_finset_le_card {S : Type*} [MeasurableSpace S] [MeasurableSingletonClass S]
    (D : Measure S) [IsFiniteMeasure D] {ρ : ℝ} (hρ : ∀ t, D.real {t} ≤ ρ) (F : Finset S) :
    D.real ↑F ≤ F.card * ρ := by
  rw [← sum_measureReal_singleton]
  calc ∑ t ∈ F, D.real {t} ≤ ∑ _t ∈ F, ρ := Finset.sum_le_sum (fun t _ => hρ t)
    _ = F.card * ρ := by rw [Finset.sum_const, nsmul_eq_mul]

end OrthoDFA
