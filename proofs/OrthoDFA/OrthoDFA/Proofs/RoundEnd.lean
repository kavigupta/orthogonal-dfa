import OrthoDFA.RoundEnd

/-!
# The replay's anchor

A replay that keeps a prefix of its draw past the anchor is spread, and an anchor at a position a
draw visits a state from has chance `1 / (L + 1 − ⌈L/2⌉)`.
-/

namespace OrthoDFA

open MeasureTheory
open scoped ENNReal

variable {α : Type*} [Countable α]

theorem startLength_le (L : ℕ) : startLength L ≤ L := by unfold startLength; omega

theorem anchorCount_pos (L : ℕ) : 0 < L + 1 - startLength L := by
  have := startLength_le L; omega

theorem anchorLaw_apply (L : ℕ) (s : Set ℕ) [DecidablePred (· ∈ s)] :
    anchorLaw L s = ((L + 1 - startLength L : ℕ) : ℝ≥0∞)⁻¹
      * (((Finset.Icc (startLength L) L).filter (· ∈ s)).card : ℝ≥0∞) := by
  simp only [anchorLaw, Measure.smul_apply, Measure.coe_finsetSum, Finset.sum_apply,
    smul_eq_mul]
  congr 1
  rw [Finset.card_filter, Nat.cast_sum]
  refine Finset.sum_congr rfl fun k _ => ?_
  by_cases hk : k ∈ s <;> simp [Set.indicator, hk]

instance (L : ℕ) : IsProbabilityMeasure (anchorLaw L) := by
  constructor
  simp only [anchorLaw, Measure.smul_apply, Measure.coe_finsetSum, Finset.sum_apply, measure_univ,
    Finset.sum_const, Nat.card_Icc, nsmul_eq_mul, mul_one, smul_eq_mul]
  exact ENNReal.inv_mul_cancel (by exact_mod_cast (anchorCount_pos L).ne') (by simp)

theorem anchorLaw_real_apply (L : ℕ) (s : Set ℕ) [DecidablePred (· ∈ s)] :
    (anchorLaw L).real s
      = (((Finset.Icc (startLength L) L).filter (· ∈ s)).card : ℝ)
        / (L + 1 - startLength L : ℕ) := by
  rw [measureReal_def, anchorLaw_apply, ENNReal.toReal_mul, ENNReal.toReal_inv,
    ENNReal.toReal_natCast, ENNReal.toReal_natCast]
  ring

theorem anchorLaw_real_le (L i : ℕ) :
    (anchorLaw L).real {e | e ≤ i}
      = ((min i L + 1 - startLength L : ℕ) : ℝ) / (L + 1 - startLength L : ℕ) := by
  rw [anchorLaw_real_apply]
  congr 2
  rw [show min i L + 1 - startLength L = (Finset.Icc (startLength L) (min i L)).card by
    rw [Nat.card_Icc]]
  congr 1
  ext k
  simp only [Finset.mem_filter, Finset.mem_Icc, Set.mem_ofPred_eq]
  omega

theorem anchorLaw_lt (L : ℕ) : anchorLaw L {e | e < startLength L} = 0 := by
  rw [anchorLaw_apply]
  rw [Finset.filter_eq_empty_iff.2 fun k hk => by
    simp only [Finset.mem_Icc] at hk; simp only [Set.mem_ofPred_eq]; omega]
  simp

/-- Every string the replay of `w` from `e` harvests starts as `w` does, up to some point at or
past `e`. -/
theorem spread_of_prefix (D : Measure (FreeMonoid α)) [IsFiniteMeasure D] (L : ℕ)
    (rp : FreeMonoid α → ℕ → List (FreeMonoid α))
    (h : ∀ w e x, x ∈ rp w e →
      ∃ i, e ≤ i ∧ i ≤ x.toList.length ∧ w.toList.take i = x.toList.take i) :
    Spread D L rp := by
  intro t
  set A := anchorLaw L
  have hsub : {q : FreeMonoid α × ℕ | t ∈ rp q.1 q.2} ⊆
      (Set.univ ×ˢ {e : ℕ | e < startLength L}) ∪
      ⋃ i ∈ Finset.Icc (startLength L) t.toList.length,
        {p : FreeMonoid α | p.toList.take i = t.toList.take i} ×ˢ {e : ℕ | e ≤ i} := by
    rintro ⟨p, e⟩ hq
    obtain ⟨i, hei, hit, htake⟩ := h p e t hq
    by_cases he : e < startLength L
    · exact .inl ⟨Set.mem_univ _, he⟩
    · refine .inr ?_
      simp only [Set.mem_iUnion, Finset.mem_Icc, Set.mem_prod, Set.mem_ofPred_eq]
      exact ⟨i, ⟨by omega, hit⟩, htake, hei⟩
  have h0 : (D.prod A) (Set.univ ×ˢ {e : ℕ | e < startLength L}) = 0 := by
    rw [Measure.prod_prod, anchorLaw_lt, mul_zero]
  calc (D.prod A).real {q | t ∈ rp q.1 q.2}
      ≤ (D.prod A).real ((Set.univ ×ˢ {e : ℕ | e < startLength L}) ∪
          ⋃ i ∈ Finset.Icc (startLength L) t.toList.length,
          {p : FreeMonoid α | p.toList.take i = t.toList.take i} ×ˢ {e : ℕ | e ≤ i}) :=
        measureReal_mono hsub (measure_ne_top _ _)
    _ ≤ (D.prod A).real (Set.univ ×ˢ {e : ℕ | e < startLength L}) + (D.prod A).real
          (⋃ i ∈ Finset.Icc (startLength L) t.toList.length,
          {p : FreeMonoid α | p.toList.take i = t.toList.take i} ×ˢ {e : ℕ | e ≤ i}) :=
        measureReal_union_le _ _
    _ ≤ 0 + ∑ i ∈ Finset.Icc (startLength L) t.toList.length, (D.prod A).real
          ({p : FreeMonoid α | p.toList.take i = t.toList.take i} ×ˢ {e : ℕ | e ≤ i}) := by
        rw [measureReal_def, h0, ENNReal.toReal_zero]
        gcongr
        exact measureReal_biUnion_finset_le _ _
    _ = _ := by
        rw [zero_add]
        refine Finset.sum_congr rfl fun i _ => ?_
        rw [measureReal_prod_prod, anchorLaw_real_le]
        ring

theorem spread_nil (D : Measure (FreeMonoid α)) (L : ℕ) : Spread D L fun _ _ => [] := by
  intro t
  simp only [List.not_mem_nil, Set.ofPred_false, measureReal_empty]
  exact Finset.sum_nonneg fun i _ => mul_nonneg (by positivity) measureReal_nonneg

theorem yields_zero (D : Measure (FreeMonoid α)) (L : ℕ)
    (rp : FreeMonoid α → ℕ → List (FreeMonoid α)) : Yields D L rp 0 :=
  measureReal_nonneg

/-- An anchor lands on a position a draw visits a state satisfying `P` from with chance at least
`1 / (L + 1 − ⌈L/2⌉)`. -/
theorem coverage_le_anchor {σ : Type*} (D : Measure (FreeMonoid α)) [IsFiniteMeasure D] (L : ℕ)
    (M : DFA α σ) (P : σ → Prop) :
    coverage D L M P / (L + 1 - startLength L : ℕ)
      ≤ (D.prod (anchorLaw L)).real
          {q | q.2 ≤ q.1.toList.length ∧ P (M.eval (q.1.toList.take q.2))} := by
  classical
  set S := {q : FreeMonoid α × ℕ | q.2 ≤ q.1.toList.length ∧ P (M.eval (q.1.toList.take q.2))}
  set C := {x : FreeMonoid α | ∃ i, startLength L ≤ i ∧ i ≤ L ∧ i ≤ x.toList.length ∧
    P (M.eval (x.toList.take i))}
  set c : ℝ≥0∞ := ((L + 1 - startLength L : ℕ) : ℝ≥0∞)⁻¹
  have hS : MeasurableSet S := S.to_countable.measurableSet
  have hC : MeasurableSet C := C.to_countable.measurableSet
  have hpt : ∀ x, C.indicator (fun _ => c) x ≤ anchorLaw L (Prod.mk x ⁻¹' S) := by
    intro x
    by_cases hx : x ∈ C
    · rw [Set.indicator_of_mem hx, anchorLaw_apply]
      obtain ⟨i, h1, h2, h3, h4⟩ := hx
      have : 1 ≤ ((Finset.Icc (startLength L) L).filter (· ∈ Prod.mk x ⁻¹' S)).card :=
        Finset.card_pos.2 ⟨i, Finset.mem_filter.2 ⟨Finset.mem_Icc.2 ⟨h1, h2⟩, h3, h4⟩⟩
      calc c = c * 1 := (mul_one c).symm
        _ ≤ _ := by gcongr; exact_mod_cast this
    · rw [Set.indicator_of_notMem hx]; exact bot_le
  have hle : c * D C ≤ (D.prod (anchorLaw L)) S := by
    rw [Measure.prod_apply hS, ← lintegral_indicator_const hC]
    exact lintegral_mono hpt
  calc coverage D L M P / (L + 1 - startLength L : ℕ)
      = (c * D C).toReal := by
        rw [ENNReal.toReal_mul, ENNReal.toReal_inv, ENNReal.toReal_natCast]
        unfold coverage
        rw [measureReal_def]
        ring
    _ ≤ (D.prod (anchorLaw L)).real S := ENNReal.toReal_mono (measure_ne_top _ _) hle

/-- The ending is the same at any replay that is spread and yields. -/
theorem EndsWell.respread {σ : Type*} {M : DFA α σ} {bad : σ → Prop}
    {D : Measure (FreeMonoid α)} {ε : ℝ} {dis : Set (FreeMonoid α)} {L L' : ℕ}
    {rp rp' : FreeMonoid α → ℕ → List (FreeMonoid α)} {y y' : ℝ} (hs : Spread D L rp)
    (hy : Yields D L rp y) {e : RoundEnd α}
    (h : EndsWell M bad D ε dis L' rp' y' e) : EndsWell M bad D ε dis L rp y e := by
  cases e with
  | consistent => exact h
  | harvest zs => exact ⟨h.1, hs, hy⟩
  | failed => exact h

end OrthoDFA
