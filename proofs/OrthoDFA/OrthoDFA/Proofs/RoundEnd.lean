import OrthoDFA.RoundEnd

/-!
# A replay that keeps a prefix of its draw past the anchor is spread
-/

namespace OrthoDFA

open MeasureTheory
open scoped ENNReal

variable {α : Type*} [Countable α]

instance (L : ℕ) : IsFiniteMeasure (anchorLaw L) := by
  constructor
  rcases Nat.eq_zero_or_pos L with rfl | hL
  · simp [anchorLaw]
  · simp only [anchorLaw, Measure.smul_apply, Measure.coe_finsetSum, Finset.sum_apply,
      measure_univ, Finset.sum_const, Finset.card_range, nsmul_eq_mul, mul_one, smul_eq_mul]
    rw [ENNReal.inv_mul_cancel (by exact_mod_cast hL.ne') (by simp)]
    exact ENNReal.one_lt_top

theorem anchorLaw_real_le (L i : ℕ) :
    (anchorLaw L).real {e | e ≤ i} = ((min (i + 1) L : ℕ) : ℝ) / L := by
  have hcount : ∑ k ∈ Finset.range L, (Measure.dirac k : Measure ℕ) {e | e ≤ i}
      = ((min (i + 1) L : ℕ) : ℝ≥0∞) := by
    simp only [Measure.dirac_apply, Set.indicator_apply, Set.mem_ofPred_eq, Pi.one_apply]
    rw [Finset.sum_ite, Finset.sum_const_zero, add_zero, Finset.sum_const, nsmul_eq_mul, mul_one]
    have hf : ((Finset.range L).filter fun k => k ≤ i) = Finset.range (min (i + 1) L) := by
      ext k
      simp only [Finset.mem_filter, Finset.mem_range]
      omega
    rw [hf, Finset.card_range]
  simp only [measureReal_def, anchorLaw, Measure.smul_apply, Measure.coe_finsetSum,
    Finset.sum_apply, smul_eq_mul, hcount, ENNReal.toReal_mul, ENNReal.toReal_inv,
    ENNReal.toReal_natCast]
  ring

/-- Every string the replay of `w` from `e` harvests starts as `w` does, up to some point at or
past `e`. -/
theorem spread_of_prefix (D : Measure (FreeMonoid α)) [IsFiniteMeasure D] (L : ℕ)
    (rp : FreeMonoid α → ℕ → List (FreeMonoid α))
    (h : ∀ w e x, x ∈ rp w e →
      ∃ i, e ≤ i ∧ i ≤ x.toList.length ∧ w.toList.take i = x.toList.take i) :
    Spread D L rp := by
  intro t
  have hsub : {q : FreeMonoid α × ℕ | t ∈ rp q.1 q.2} ⊆
      ⋃ i ∈ Finset.range (t.toList.length + 1),
        {p : FreeMonoid α | p.toList.take i = t.toList.take i} ×ˢ {e : ℕ | e ≤ i} := by
    rintro ⟨p, e⟩ hq
    obtain ⟨i, hei, hit, htake⟩ := h p e t hq
    simp only [Set.mem_iUnion, Finset.mem_range, Set.mem_prod, Set.mem_ofPred_eq]
    exact ⟨i, by omega, htake, hei⟩
  calc (D.prod (anchorLaw L)).real {q | t ∈ rp q.1 q.2}
      ≤ (D.prod (anchorLaw L)).real (⋃ i ∈ Finset.range (t.toList.length + 1),
          {p : FreeMonoid α | p.toList.take i = t.toList.take i} ×ˢ {e : ℕ | e ≤ i}) :=
        measureReal_mono hsub (measure_ne_top _ _)
    _ ≤ ∑ i ∈ Finset.range (t.toList.length + 1), (D.prod (anchorLaw L)).real
          ({p : FreeMonoid α | p.toList.take i = t.toList.take i} ×ˢ {e : ℕ | e ≤ i}) :=
        measureReal_biUnion_finset_le _ _
    _ = _ := by
        refine Finset.sum_congr rfl fun i _ => ?_
        rw [measureReal_prod_prod, anchorLaw_real_le]
        ring

theorem spread_nil (D : Measure (FreeMonoid α)) (L : ℕ) : Spread D L fun _ _ => [] := by
  intro t
  simp only [List.not_mem_nil, Set.ofPred_false, measureReal_empty]
  exact Finset.sum_nonneg fun i _ => mul_nonneg (by positivity) measureReal_nonneg

/-- The ending is the same at any spread replay. -/
theorem EndsWell.respread {σ : Type*} {M : DFA α σ} {bad : σ → Prop}
    {D : Measure (FreeMonoid α)} {ε : ℝ} {dis : Set (FreeMonoid α)} {L L' : ℕ}
    {rp rp' : FreeMonoid α → ℕ → List (FreeMonoid α)} (hs : Spread D L rp) {e : RoundEnd α}
    (h : EndsWell M bad D ε dis L' rp' e) : EndsWell M bad D ε dis L rp e := by
  cases e with
  | consistent => exact h
  | harvest zs => exact ⟨h.1, hs⟩
  | failed => exact h

end OrthoDFA
