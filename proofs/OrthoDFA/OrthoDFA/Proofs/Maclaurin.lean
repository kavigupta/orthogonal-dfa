import OrthoDFA.Proofs.BandKappa
import Mathlib.RingTheory.MvPolynomial.Symmetric.Defs

/-!
# Maclaurin's inequality, from Newton's

For `x ≥ 0` over `N` entries,

    e_W(x) / C(N, W) ≤ (e_1(x) / N)^W.

At rates `pᵢ = xᵢ/(1 + xᵢ)` the Poisson-binomial law is `P(X = j) = e_j(x) ∏ (1 − pᵢ)`, and
`P(X = j)/C(N, j)` is log-concave (`ulcLaw_ulc`), so its successive ratios fall from the first.
-/

namespace OrthoDFA

open Polynomial

lemma esymm_cons_succ (x : ℝ) (M : Multiset ℝ) (j : ℕ) :
    (x ::ₘ M).esymm (j + 1) = M.esymm (j + 1) + x * M.esymm j := by
  simp only [Multiset.esymm, Multiset.powersetCard_cons, Multiset.map_add, Multiset.sum_add,
    Multiset.map_map, Function.comp_def, Multiset.prod_cons]
  rw [Multiset.sum_map_mul_left]

lemma esymm_zero_right (M : Multiset ℝ) : M.esymm 0 = 1 := by
  simp [Multiset.esymm]

lemma esymm_one_right (M : Multiset ℝ) : M.esymm 1 = M.sum := by
  simp [Multiset.esymm, Multiset.powersetCard_one, Multiset.map_map]

lemma bitPoly_odds_coeff (M : Multiset ℝ) (hM : ∀ x ∈ M, 0 ≤ x) (j : ℕ) :
    (bitPoly (M.map fun x => x / (1 + x))).coeff j
      = (M.map fun x => (1 + x)⁻¹).prod * M.esymm j := by
  induction M using Multiset.induction_on generalizing j with
  | empty =>
    cases j with
    | zero => simp [bitPoly, Multiset.esymm]
    | succ j =>
      simp [bitPoly, Multiset.esymm, Polynomial.coeff_one, Multiset.powersetCard_zero_right]
  | cons x M ih =>
    have hx : 0 ≤ x := hM x (Multiset.mem_cons_self _ _)
    have hM' : ∀ y ∈ M, 0 ≤ y := fun y hy => hM y (Multiset.mem_cons_of_mem hy)
    have h1 : (1 : ℝ) + x ≠ 0 := by linarith
    rw [Multiset.map_cons, bitPoly_cons, Multiset.map_cons, Multiset.prod_cons]
    cases j with
    | zero =>
      rw [coeff_bitLin_mul_zero, ih hM', esymm_zero_right, esymm_zero_right]
      field_simp
      ring
    | succ j =>
      rw [coeff_bitLin_mul_succ, ih hM', ih hM', esymm_cons_succ]
      field_simp
      ring

/-- A log-concave sequence positive at `0` has no ratio above its first. -/
lemma ULC.ratio_le {n : ℕ} {g : ℕ → ℝ} (h : ULC n g) (h0 : 0 < g 0) :
    ∀ j, g (j + 1) * g 0 ≤ g j * g 1 := by
  intro j
  induction j with
  | zero => exact le_of_eq (mul_comm _ _)
  | succ j ih =>
    rcases (h.nonneg (j + 1)).lt_or_eq with hp | hz
    · have hj : 0 < g j := by
        rcases j with _ | j
        · exact h0
        · exact h.conn 0 (j + 1) (j + 2) (by omega) (by omega) h0 hp
      refine le_of_mul_le_mul_left ?_ hj
      have hlc := h.lc j
      calc g j * (g (j + 1 + 1) * g 0) = (g j * g (j + 2)) * g 0 := by ring
        _ ≤ g (j + 1) ^ 2 * g 0 := mul_le_mul_of_nonneg_right hlc h0.le
        _ = g (j + 1) * (g (j + 1) * g 0) := by ring
        _ ≤ g (j + 1) * (g j * g 1) := mul_le_mul_of_nonneg_left ih hp.le
        _ = g j * (g (j + 1) * g 1) := by ring
    · have h2 : g (j + 2) = 0 := by
        by_contra hne
        have := h.conn 0 (j + 1) (j + 2) (by omega) (by omega) h0
          (lt_of_le_of_ne (h.nonneg _) (Ne.symm hne))
        linarith
      rw [show j + 1 + 1 = j + 2 by omega, h2, ← hz]
      simp

lemma ULC.pow_le {n : ℕ} {g : ℕ → ℝ} (h : ULC n g) (h0 : 0 < g 0) :
    ∀ W, g W * g 0 ^ W ≤ g 0 * g 1 ^ W := by
  intro W
  induction W with
  | zero => simp
  | succ W ih =>
    have hr := h.ratio_le h0 W
    have h1 := h.nonneg 1
    calc g (W + 1) * g 0 ^ (W + 1) = (g (W + 1) * g 0) * g 0 ^ W := by ring
      _ ≤ (g W * g 1) * g 0 ^ W := mul_le_mul_of_nonneg_right hr (pow_nonneg h0.le _)
      _ = g 1 * (g W * g 0 ^ W) := by ring
      _ ≤ g 1 * (g 0 * g 1 ^ W) := mul_le_mul_of_nonneg_left ih h1
      _ = g 0 * g 1 ^ (W + 1) := by ring

/-- Maclaurin's inequality. -/
theorem esymm_le_maclaurin (M : Multiset ℝ) (hM : ∀ x ∈ M, 0 ≤ x) (W : ℕ) :
    M.esymm W ≤ (M.card.choose W : ℝ) * (M.sum / M.card) ^ W := by
  rcases Nat.eq_zero_or_pos W with rfl | hW
  · simp [esymm_zero_right]
  rcases lt_or_ge M.card W with hlt | hle
  · rw [Multiset.esymm, Multiset.powersetCard_eq_empty _ hlt, Nat.choose_eq_zero_of_lt hlt]
    simp
  obtain ⟨L, rfl⟩ := Quot.exists_rep M
  set M : Multiset ℝ := Quot.mk _ L with hMdef
  have hML : M = (L : Multiset ℝ) := rfl
  set ps : List ℝ := L.map fun x => x / (1 + x) with hps
  have hps01 : ∀ p ∈ ps, 0 ≤ p ∧ p ≤ 1 := by
    intro p hp
    obtain ⟨x, hx, rfl⟩ := List.mem_map.1 hp
    have hx0 : 0 ≤ x := hM x (by rw [hML]; exact hx)
    exact ⟨div_nonneg hx0 (by linarith), (div_le_one (by linarith)).2 (by linarith)⟩
  have hlen : ps.length = M.card := by rw [hps, List.length_map, hML, Multiset.coe_card]
  set g := ulcLaw ps with hg
  have hulc : ULC ps.length g := ulcLaw_ulc ps hps01
  set K : ℝ := (M.map fun x => (1 + x)⁻¹).prod with hK
  have hKpos : 0 < K := by
    refine Multiset.prod_pos fun y hy => ?_
    obtain ⟨x, hx, rfl⟩ := Multiset.mem_map.1 hy
    have := hM x hx
    positivity
  have hcoe : ((ps : Multiset ℝ)) = M.map fun x => x / (1 + x) := by
    rw [hps, ← Multiset.map_coe, hML]
  have hgj : ∀ j, (M.card.choose j : ℝ) * g j = K * M.esymm j := by
    intro j
    rw [← hlen, ← pbLaw_eq_choose_mul, pbLaw_eq_coeff, hcoe, bitPoly_odds_coeff M hM]
  have hg0 : g 0 = K := by
    have := hgj 0
    rw [Nat.choose_zero_right, esymm_zero_right] at this
    simpa using this
  have hn : (0 : ℝ) < M.card := by exact_mod_cast lt_of_lt_of_le hW hle
  have hg1 : g 1 = K * (M.sum / M.card) := by
    have := hgj 1
    rw [Nat.choose_one_right, esymm_one_right] at this
    field_simp
    linarith
  have hpow := hulc.pow_le (hg0 ▸ hKpos) W
  rw [hg0, hg1, mul_pow] at hpow
  have hgW : g W ≤ K * (M.sum / M.card) ^ W := by
    have hKW : 0 < K ^ W := pow_pos hKpos W
    refine le_of_mul_le_mul_right ?_ hKW
    calc g W * K ^ W ≤ K * (K ^ W * (M.sum / M.card) ^ W) := hpow
      _ = K * (M.sum / M.card) ^ W * K ^ W := by ring
  have hC : (0 : ℝ) ≤ M.card.choose W := Nat.cast_nonneg _
  have := hgj W
  have hKe : K * M.esymm W ≤ K * ((M.card.choose W : ℝ) * (M.sum / M.card) ^ W) := by
    rw [← this]
    calc (M.card.choose W : ℝ) * g W ≤ M.card.choose W * (K * (M.sum / M.card) ^ W) :=
          mul_le_mul_of_nonneg_left hgW hC
      _ = K * ((M.card.choose W : ℝ) * (M.sum / M.card) ^ W) := by ring
  exact le_of_mul_le_mul_left hKe hKpos

/-- Maclaurin's inequality over a finset: the mean of `∏_{v ∈ F} φ v` over the `W`-subsets `F`
of `C` is at most the `W`-th power of `φ`'s mean. -/
theorem sum_powersetCard_prod_le {ι : Type*} (C : Finset ι) (φ : ι → ℝ)
    (hφ : ∀ v ∈ C, 0 ≤ φ v) (W : ℕ) :
    ∑ F ∈ C.powersetCard W, ∏ v ∈ F, φ v
      ≤ (C.card.choose W : ℝ) * ((∑ v ∈ C, φ v) / C.card) ^ W := by
  have h := esymm_le_maclaurin (C.val.map φ) (fun x hx => by
    obtain ⟨v, hv, rfl⟩ := Multiset.mem_map.1 hx
    exact hφ v hv) W
  rw [Finset.esymm_map_val, Multiset.card_map] at h
  exact h

end OrthoDFA
