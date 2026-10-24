import Mathlib.Analysis.Asymptotics.Defs
import Mathlib.Analysis.Complex.ExponentialBounds
import Mathlib.Analysis.SpecialFunctions.Log.Basic

/-! Lemmas the big-O claims share. -/

namespace OrthoDFA

open Asymptotics Filter Real

/-- On a product of reals, `a ≤ b` means past `a` in every coordinate. -/
lemma isBigO_of_ge {ι : Type*} [SemilatticeSup ι] [Nonempty ι] {f g : ι → ℝ} (a : ι) (C : ℝ)
    (h : ∀ b, a ≤ b → |f b| ≤ C * |g b|) : f =O[atTop] g :=
  IsBigO.of_bound C (eventually_atTop.2 ⟨a, fun b hb => by
    simpa only [Real.norm_eq_abs] using h b hb⟩)

lemma one_div_min_le {x y : ℝ} (hx : 0 < x) (hy : 0 < y) : 1 / min x y ≤ 1 / x + 1 / y := by
  rcases min_cases x y with ⟨h, _⟩ | ⟨h, _⟩ <;> rw [h] <;>
    linarith [one_div_pos.2 hx, one_div_pos.2 hy]

lemma div_one_div' (a b : ℝ) : a / (1 / b) = a * b := by rw [one_div, div_inv_eq_mul]

lemma abs_le_mul_abs_of {w g C : ℝ} (hC : 0 ≤ C) (h0 : 0 ≤ w) (h : w ≤ C * g) :
    |w| ≤ C * |g| := by
  rw [abs_of_nonneg h0]
  exact h.trans (mul_le_mul_of_nonneg_left (le_abs_self g) hC)

lemma log_two_le_log {x : ℝ} (hx : 2 ≤ x) : log 2 ≤ log x :=
  Real.log_le_log (by norm_num) hx

lemma one_le_log_of_three_le {x : ℝ} (hx : 3 ≤ x) : 1 ≤ log x :=
  (le_log_iff_exp_le (by linarith)).2 (by linarith [exp_one_lt_three])

lemma log_le_mul_log_of_le_pow {y P : ℝ} (k : ℕ) (hy : 0 < y) (h : y ≤ P ^ k) :
    log y ≤ k * log P := by
  rw [← Real.log_pow]; exact Real.log_le_log hy h

lemma log_le_log_max {x y : ℝ} (hx : 0 ≤ x) (h : x ≤ y) : log x ≤ log (max 1 y) := by
  rcases hx.eq_or_lt with rfl | hx
  · rw [Real.log_zero]; exact Real.log_nonneg (le_max_left _ _)
  · exact Real.log_le_log hx (h.trans (le_max_right _ _))

lemma pow_length_le_prod {l : List ℝ} {c : ℝ} (hc : 0 ≤ c) (h : ∀ x ∈ l, c ≤ x) :
    c ^ l.length ≤ l.prod := by
  simpa using List.prod_map_le_prod_map₀ (fun _ => c) id (fun _ _ => hc) h

lemma prod_le_prod_of_sublist {l₁ l₂ : List ℝ} (hs : l₁.Sublist l₂) (h : ∀ x ∈ l₂, 1 ≤ x) :
    l₁.prod ≤ l₂.prod := by
  induction hs with
  | slnil => rfl
  | cons a _ ih =>
    rw [List.forall_mem_cons] at h
    exact (ih h.2).trans (le_mul_of_one_le_left (zero_le_one.trans (List.one_le_prod h.2)) h.1)
  | cons_cons a _ ih =>
    rw [List.forall_mem_cons] at h
    simp only [List.prod_cons]
    exact mul_le_mul_of_nonneg_left (ih h.2) (by linarith [h.1])

end OrthoDFA
