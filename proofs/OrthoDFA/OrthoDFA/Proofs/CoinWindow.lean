import OrthoDFA.Proofs.Coins

/-!
# The coins on a window of strings

On a fixed set `V` of strings, the fair coins put about half on each side, and for each of a
finite family of subsets `F v ⊆ V` the reserve side holds no more than the table side plus a
deviation of the subset's own size.
-/

namespace OrthoDFA.Thin

open MeasureTheory ProbabilityTheory

variable {S : Type*}

open scoped Classical in
/-- More than half of a window's coins plus `g₀` on the table side has chance at most
`exp(-2 g₀²/|V|)`. -/
theorem coins_window_upper (V : Finset S) {g₀ : ℝ} (hg₀ : 0 ≤ g₀) (hV : 0 < V.card) :
    (coins S).real {g | (V.card : ℝ) / 2 + g₀ ≤ ((V.filter (fun s => g s = true)).card : ℝ)}
      ≤ Real.exp (-2 * g₀ ^ 2 / V.card) := by
  have hVr : (0 : ℝ) < V.card := by exact_mod_cast hV
  have h := coins_count_upper V true (τ := g₀ / V.card) (by positivity)
  refine le_trans (measureReal_mono (fun g hg => ?_) (measure_ne_top _ _)) (le_trans h ?_)
  · simp only [Set.mem_setOf_eq] at hg ⊢
    have e : (V.card : ℝ) * (1 / 2 + g₀ / V.card) = (V.card : ℝ) / 2 + g₀ := by
      field_simp
    linarith
  · refine Real.exp_le_exp.2 (le_of_eq ?_)
    field_simp

open scoped Classical in
/-- For each `v` of a finite family, the reserve side of `F v` exceeds its table side by
`√(2 |F v| L)` with chance at most `|family| e^{-L}`. -/
theorem coins_family_balance {ι : Type*} (I : Finset ι) (F : ι → Finset S) {L : ℝ} (hL : 0 ≤ L) :
    (coins S).real {g | ∃ v ∈ I, Real.sqrt (2 * (F v).card * L)
        < (((F v).filter (fun s => g s = false)).card : ℝ)
          - (((F v).filter (fun s => g s = true)).card : ℝ)}
      ≤ I.card * Real.exp (-L) := by
  have hone : ∀ v ∈ I, (coins S).real {g | Real.sqrt (2 * (F v).card * L)
      < (((F v).filter (fun s => g s = false)).card : ℝ)
        - (((F v).filter (fun s => g s = true)).card : ℝ)} ≤ Real.exp (-L) := by
    intro v _
    rcases Nat.eq_zero_or_pos (F v).card with h0 | hpos
    · have hempty : {g : S → Bool | Real.sqrt (2 * (F v).card * L)
          < (((F v).filter (fun s => g s = false)).card : ℝ)
            - (((F v).filter (fun s => g s = true)).card : ℝ)} = ∅ := by
        ext g
        simp [Finset.card_eq_zero.1 h0]
      rw [hempty, measureReal_empty]
      positivity
    · have hn : (0 : ℝ) < (F v).card := by exact_mod_cast hpos
      set τ := Real.sqrt (L / (2 * (F v).card)) with hτ
      have h := coins_count_upper (F v) false (τ := τ) (Real.sqrt_nonneg _)
      refine le_trans (measureReal_mono (fun g hg => ?_) (measure_ne_top _ _)) (le_trans h ?_)
      · simp only [Set.mem_setOf_eq] at hg ⊢
        have hsplit : (((F v).filter (fun s => g s = false)).card : ℝ)
            + (((F v).filter (fun s => g s = true)).card : ℝ) = (F v).card := by
          have := Finset.card_filter_add_card_filter_not (s := F v)
            (fun s => g s = false)
          have e : (F v).filter (fun s => ¬ g s = false) = (F v).filter (fun s => g s = true) := by
            congr 1; ext s; simp
          rw [e] at this
          exact_mod_cast this
        have hsq : Real.sqrt (2 * (F v).card * L) = 2 * (F v).card * τ := by
          rw [hτ, show 2 * ((F v).card : ℝ) * L = (2 * (F v).card) ^ 2 * (L / (2 * (F v).card))
            by field_simp, Real.sqrt_mul (by positivity), Real.sqrt_sq (by positivity)]
        rw [hsq] at hg
        nlinarith
      · refine Real.exp_le_exp.2 (le_of_eq ?_)
        rw [hτ, Real.sq_sqrt (by positivity)]
        field_simp
  calc (coins S).real {g | ∃ v ∈ I, Real.sqrt (2 * (F v).card * L)
        < (((F v).filter (fun s => g s = false)).card : ℝ)
          - (((F v).filter (fun s => g s = true)).card : ℝ)}
      ≤ (coins S).real (⋃ v ∈ I, {g | Real.sqrt (2 * (F v).card * L)
        < (((F v).filter (fun s => g s = false)).card : ℝ)
          - (((F v).filter (fun s => g s = true)).card : ℝ)}) := by
        refine measureReal_mono (fun g hg => ?_) (measure_ne_top _ _)
        obtain ⟨v, hv, h⟩ := hg
        exact Set.mem_biUnion hv h
    _ ≤ ∑ v ∈ I, (coins S).real {g | Real.sqrt (2 * (F v).card * L)
        < (((F v).filter (fun s => g s = false)).card : ℝ)
          - (((F v).filter (fun s => g s = true)).card : ℝ)} := measureReal_biUnion_finset_le _ _
    _ ≤ ∑ _v ∈ I, Real.exp (-L) := Finset.sum_le_sum hone
    _ = I.card * Real.exp (-L) := by rw [Finset.sum_const, nsmul_eq_mul]

/-- `X − Y ≤ √(2(X+Y)L)` gives `X ≤ Y + 2√(LY) + 2L`. -/
lemma le_of_balance {X Y L : ℝ} (hY : 0 ≤ Y) (hL : 0 ≤ L) (hX : 0 ≤ X)
    (h : X - Y ≤ Real.sqrt (2 * (X + Y) * L)) :
    X ≤ Y + 2 * Real.sqrt (L * Y) + 2 * L := by
  by_contra hcon
  push Not at hcon
  set D := X - Y with hD
  have hD0 : 2 * Real.sqrt (L * Y) + 2 * L < D := by linarith
  have hsq : D ^ 2 ≤ 2 * (2 * Y + D) * L := by
    have hDnn : 0 ≤ D := by nlinarith [Real.sqrt_nonneg (L * Y)]
    have := pow_le_pow_left₀ hDnn h 2
    rw [Real.sq_sqrt (by nlinarith)] at this
    nlinarith
  have hs := Real.sq_sqrt (mul_nonneg hL hY)
  have hs0 := Real.sqrt_nonneg (L * Y)
  nlinarith

end OrthoDFA.Thin
