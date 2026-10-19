import OrthoDFA.Proofs.Coins
import Mathlib.Probability.ConditionalProbability

/-!
# From the reserve side's law back to the population's

A population with small atoms puts about half its mass on each side of the hash, on any set the
hash on a few strings chooses.  So a guarantee on the reserve side's conditional law is one on
the population's, up to a small slack.
-/

namespace OrthoDFA.Thin

open MeasureTheory ProbabilityTheory
open scoped ENNReal

variable {S : Type*} [MeasurableSpace S] [MeasurableSingletonClass S] [Countable S]

lemma measureReal_eq_tsum (D : Measure S) [IsFiniteMeasure D] (X : Set S) :
    D.real X = ∑' s, X.indicator (fun s => D.real {s}) s := by
  rw [measureReal_def, ← Measure.tsum_indicator_apply_singleton D X
    (Set.to_countable X).measurableSet, ENNReal.tsum_toReal_eq (fun s => by
      by_cases h : s ∈ X <;> simp [h, measure_ne_top])]
  refine tsum_congr (fun s => ?_)
  by_cases h : s ∈ X <;> simp [h, measureReal_def]

lemma summable_singleton (D : Measure S) [IsFiniteMeasure D] :
    Summable (fun s => D.real {s}) := by
  classical
  refine summable_of_sum_le (c := D.real Set.univ) (fun s => measureReal_nonneg) (fun F => ?_)
  rw [sum_measureReal_singleton]
  exact measureReal_mono (Set.subset_univ _) (measure_ne_top _ _)

/-- A finite set holding all but `ε` of a finite measure. -/
lemma exists_finset_compl_le (D : Measure S) [IsFiniteMeasure D] {ε : ℝ} (hε : 0 < ε) :
    ∃ F : Finset S, D.real (↑F)ᶜ ≤ ε := by
  classical
  have hs := (summable_singleton D).hasSum
  have htend := hs.eventually (Metric.ball_mem_nhds _ hε)
  obtain ⟨F, hF⟩ := Filter.eventually_atTop.1 htend
  refine ⟨F, ?_⟩
  have h1 := hF F le_rfl
  rw [Real.dist_eq] at h1
  have huniv : D.real Set.univ = ∑' s, D.real {s} := by
    rw [measureReal_eq_tsum]; simp
  rw [measureReal_compl (Finset.measurableSet F), huniv, ← sum_measureReal_singleton]
  rw [abs_lt] at h1
  linarith

/-- The strings the hash `g` puts on side `b`, outside `H`. -/
def onSide (H : Finset S) (b : Bool) (g : S → Bool) : Set S := {s | g s = b ∧ s ∉ H}

open scoped Classical in
/-- On a set the coins on `U` choose, the population's table side and reserve side hold the same
mass up to `t + D(U) + ε`, off an `exp(-t²/(2ρ))` chance, when no string has more than `ρ`. -/
theorem side_balance (D : Measure S) [IsProbabilityMeasure D] {ρ : ℝ}
    (hρ : ∀ s, D.real {s} ≤ ρ) (H U : Finset S) (W : (S → Bool) → Set S)
    (hW : ∀ g g', (∀ s ∈ U, g s = g' s) → W g = W g') {t ε : ℝ} (ht : 0 < t) (hε : 0 < ε) :
    (coins S).real {g | t + D.real ↑U + ε
        < |D.real (W g ∩ onSide H true g) - D.real (W g ∩ onSide H false g)|}
      ≤ 2 * Real.exp (-t ^ 2 / (2 * ρ)) := by
  obtain ⟨F, hF⟩ := exists_finset_compl_le D hε
  set a : S → ℝ := fun s => D.real {s} with ha
  have ha0 : ∀ s, 0 ≤ a s := fun s => measureReal_nonneg
  have hsa := summable_singleton D
  set W' : (S → Bool) → Set S := fun g => W g \ ↑H with hW'
  have hW'U : ∀ g g', (∀ s ∈ U, g s = g' s) → W' g = W' g' := fun g g' h => by
    simp only [hW', hW g g' h]
  set f : (S → Bool) → S → ℝ := fun g s => (W' g).indicator (fun s => a s * sgn (g s)) s with hf
  have hfa : ∀ g s, |f g s| ≤ a s := by
    intro g s
    simp only [hf]
    by_cases h : s ∈ W' g
    · rw [Set.indicator_of_mem h, abs_mul]
      have : |sgn (g s)| = 1 := by cases g s <;> simp [sgn]
      rw [this, mul_one, abs_of_nonneg (ha0 s)]
    · rw [Set.indicator_of_notMem h, abs_zero]; exact ha0 s
  have hfs : ∀ g, Summable (f g) := fun g =>
    Summable.of_norm_bounded hsa (fun s => by rw [Real.norm_eq_abs]; exact hfa g s)
  have hdiff : ∀ g, D.real (W g ∩ onSide H true g) - D.real (W g ∩ onSide H false g)
      = ∑' s, f g s := by
    intro g
    rw [measureReal_eq_tsum, measureReal_eq_tsum, ← Summable.tsum_sub (hsa.indicator _)
      (hsa.indicator _)]
    refine tsum_congr (fun s => ?_)
    simp only [hf, hW', onSide]
    by_cases hs : s ∈ W g <;> by_cases hH : s ∈ H <;> cases hg : g s <;>
      simp [hs, hH, hg, sgn, Set.indicator, Finset.mem_coe, ha]
  have hsplit : ∀ g, |∑' s, f g s|
      ≤ |∑ s ∈ F.filter (fun s => s ∈ W' g ∧ s ∉ U), a s * sgn (g s)| + D.real ↑U + ε := by
    intro g
    rw [← (hfs g).sum_add_tsum_compl (s := F)]
    have h1 : |∑' s : ↑(↑F : Set S)ᶜ, f g s| ≤ ε := by
      refine le_trans (norm_tsum_le_tsum_norm ((hfs g).subtype _).norm) ?_
      refine le_trans ((((hfs g).subtype _).norm).tsum_le_tsum (fun s => hfa g s)
        (hsa.subtype _)) ?_
      calc ∑' i : ↑(↑F : Set S)ᶜ, a i = D.real (↑F : Set S)ᶜ := by
            rw [tsum_subtype]; exact (measureReal_eq_tsum D _).symm
        _ ≤ ε := hF
    have h2 : ∑ s ∈ F, f g s = ∑ s ∈ F.filter (fun s => s ∈ W' g ∧ s ∉ U), a s * sgn (g s)
        + ∑ s ∈ F.filter (fun s => s ∈ U), f g s := by
      rw [← Finset.sum_filter_add_sum_filter_not F (fun s => s ∈ U), add_comm]
      congr 1
      rw [Finset.sum_filter, Finset.sum_filter]
      refine Finset.sum_congr rfl (fun s _ => ?_)
      simp only [hf]
      by_cases h1 : s ∈ W' g <;> by_cases h2 : s ∈ U <;> simp [h1, h2]
    have h3 : |∑ s ∈ F.filter (fun s => s ∈ U), f g s| ≤ D.real ↑U := by
      refine le_trans (Finset.abs_sum_le_sum_abs _ _) (le_trans
        (Finset.sum_le_sum (fun s _ => hfa g s)) ?_)
      rw [← sum_measureReal_singleton]
      exact Finset.sum_le_sum_of_subset_of_nonneg (fun s hs => (Finset.mem_filter.1 hs).2)
        (fun s _ _ => ha0 s)
    rw [h2]
    calc |∑ s ∈ F.filter (fun s => s ∈ W' g ∧ s ∉ U), a s * sgn (g s)
          + ∑ s ∈ F.filter (fun s => s ∈ U), f g s + ∑' s : ↑(↑F : Set S)ᶜ, f g s|
        ≤ |∑ s ∈ F.filter (fun s => s ∈ W' g ∧ s ∉ U), a s * sgn (g s)|
          + |∑ s ∈ F.filter (fun s => s ∈ U), f g s| + |∑' s : ↑(↑F : Set S)ᶜ, f g s| :=
          abs_add_three _ _ _
      _ ≤ _ := by linarith
  have hsub : {g : S → Bool | t + D.real ↑U + ε
      < |D.real (W g ∩ onSide H true g) - D.real (W g ∩ onSide H false g)|}
      ⊆ {g | t ≤ |∑ s ∈ F.filter (fun s => s ∈ W' g ∧ s ∉ U), a s * sgn (g s)|} := by
    intro g hg
    simp only [Set.mem_setOf_eq] at hg ⊢
    rw [hdiff g] at hg
    linarith [hsplit g]
  rcases eq_or_lt_of_le (Finset.sum_nonneg (fun s _ => sq_nonneg (a s)) :
      (0 : ℝ) ≤ ∑ s ∈ F, a s ^ 2) with h0 | hpos
  · have hz : ∀ s ∈ F, a s = 0 := fun s hs =>
      pow_eq_zero_iff (n := 2) (by norm_num) |>.1
        ((Finset.sum_eq_zero_iff_of_nonneg (fun s _ => sq_nonneg (a s))).1 h0.symm s hs)
    have hempty : {g : S → Bool | t ≤ |∑ s ∈ F.filter (fun s => s ∈ W' g ∧ s ∉ U),
        a s * sgn (g s)|} = ∅ := by
      ext g
      simp only [Set.mem_setOf_eq, Set.mem_empty_iff_false, iff_false, not_le]
      rw [Finset.sum_eq_zero (fun s hs => by rw [hz s (Finset.mem_filter.1 hs).1, zero_mul]),
        abs_zero]
      exact ht
    rw [hempty] at hsub
    rw [Set.subset_empty_iff.1 hsub, measureReal_empty]
    positivity
  obtain ⟨s₀, -⟩ : F.Nonempty := by
    by_contra h
    rw [Finset.not_nonempty_iff_eq_empty] at h
    simp [h] at hpos
  have hρ0 : 0 ≤ ρ := le_trans measureReal_nonneg (hρ s₀)
  have hsq : ∑ s ∈ F, a s ^ 2 ≤ ρ := by
    calc ∑ s ∈ F, a s ^ 2 ≤ ∑ s ∈ F, ρ * a s :=
          Finset.sum_le_sum (fun s _ => by rw [sq]; exact mul_le_mul_of_nonneg_right (hρ s) (ha0 s))
      _ = ρ * D.real ↑F := by rw [← Finset.mul_sum, sum_measureReal_singleton]
      _ ≤ ρ := by
          have : D.real ↑F ≤ 1 := measureReal_le_one
          nlinarith
  have hsp := coins_split_tail U F a W' hW'U ht
  refine le_trans (measureReal_mono (hsub.trans (fun g hg => by convert hg))
    (measure_ne_top _ _)) (le_trans hsp ?_)
  exact mul_le_mul_of_nonneg_left (exp_tail_mono hpos hsq) (by norm_num)

lemma cond_real_apply (D : Measure S) [IsFiniteMeasure D] (R W : Set S) :
    (D[|R]).real W = (D.real R)⁻¹ * D.real (R ∩ W) := by
  rw [measureReal_def, cond_apply (Set.to_countable R).measurableSet, ENNReal.toReal_mul,
    ENNReal.toReal_inv, ← measureReal_def, ← measureReal_def]

lemma sideOf_partition (D : Measure S) [IsFiniteMeasure D] (H : Finset S) (g : S → Bool)
    (W : Set S) :
    D.real W = D.real (W ∩ onSide H true g) + D.real (W ∩ onSide H false g)
      + D.real (W ∩ ↑H) := by
  have hm : ∀ X : Set S, MeasurableSet X := fun X => (Set.to_countable X).measurableSet
  have h1 : W ∩ onSide H true g ∪ W ∩ onSide H false g ∪ W ∩ ↑H = W := by
    ext s
    simp only [onSide, Set.mem_union, Set.mem_inter_iff, Set.mem_setOf_eq, Finset.mem_coe]
    constructor
    · rintro ((⟨h, -⟩ | ⟨h, -⟩) | ⟨h, -⟩) <;> exact h
    · intro h
      by_cases hH : s ∈ H
      · exact Or.inr ⟨h, hH⟩
      · cases hg : g s
        · exact Or.inl (Or.inr ⟨h, rfl, hH⟩)
        · exact Or.inl (Or.inl ⟨h, rfl, hH⟩)
  conv_lhs => rw [← h1]
  rw [measureReal_union _ (hm _), measureReal_union _ (hm _)]
  · rw [Set.disjoint_left]
    rintro s ⟨-, h1, -⟩ ⟨-, h2, -⟩
    rw [h1] at h2; exact Bool.noConfusion h2
  · rw [Set.disjoint_left]
    rintro s (⟨-, -, h1⟩ | ⟨-, -, h1⟩) ⟨-, h2⟩ <;> exact h1 h2

/-- A population's mass on a set is its reserve side's conditional mass there, up to the sides'
imbalance on the set and overall, and the harvested strings' mass. -/
theorem le_cond_add (D : Measure S) [IsProbabilityMeasure D] (H : Finset S) (g : S → Bool)
    (W : Set S) {e : ℝ} (he : 0 ≤ e) (he1 : e ≤ 1 / 8) (hH : D.real ↑H ≤ 1 / 4)
    (hW : |D.real (W ∩ onSide H true g) - D.real (W ∩ onSide H false g)| ≤ e)
    (hS : |D.real (onSide H true g) - D.real (onSide H false g)| ≤ e) :
    D.real W ≤ (D[|onSide H false g]).real W + 2 * e + D.real ↑H := by
  have hpW := sideOf_partition D H g W
  have hpU := sideOf_partition D H g Set.univ
  simp only [Set.univ_inter, probReal_univ] at hpU
  have hWH : D.real (W ∩ ↑H) ≤ D.real ↑H := measureReal_mono Set.inter_subset_right
  have hR0 : 0 < D.real (onSide H false g) := by
    rw [abs_le] at hS; linarith [measureReal_nonneg (μ := D) (s := ↑H)]
  have hRle : D.real (onSide H false g) ≤ (1 + e) / 2 := by
    rw [abs_le] at hS; linarith [measureReal_nonneg (μ := D) (s := ↑H)]
  have hc := cond_real_apply D (onSide H false g) W
  have hcomm : onSide H false g ∩ W = W ∩ onSide H false g := Set.inter_comm _ _
  rw [hcomm] at hc
  have hWR : D.real (W ∩ onSide H false g)
      = (D[|onSide H false g]).real W * D.real (onSide H false g) := by
    rw [hc, mul_comm, ← mul_assoc, mul_inv_cancel₀ hR0.ne', one_mul]
  have hc1 : (D[|onSide H false g]).real W ≤ 1 := by
    have : IsProbabilityMeasure (D[|onSide H false g]) :=
      cond_isProbabilityMeasure (by
        intro h0; rw [measureReal_def, h0] at hR0; simp at hR0)
    exact measureReal_le_one
  have hc0 : 0 ≤ (D[|onSide H false g]).real W := measureReal_nonneg
  rw [abs_le] at hW
  nlinarith

end OrthoDFA.Thin
