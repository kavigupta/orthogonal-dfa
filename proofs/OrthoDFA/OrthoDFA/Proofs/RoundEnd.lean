import OrthoDFA.RoundEnd

/-!
# The replay's anchor

A replay that keeps a prefix of its draw past the anchor is spread, and an anchor at a position a
draw visits a state from has chance `1 / (L + 1 − s)`. Neither needs anything of `s`.
-/

namespace OrthoDFA

open MeasureTheory
open scoped ENNReal

variable {α : Type*} [Countable α]

theorem anchorCount_pos {s L : ℕ} (h : s ≤ L) : 0 < L + 1 - s := by omega

theorem uniformAnchor_apply (s L : ℕ) (A : Set ℕ) [DecidablePred (· ∈ A)] :
    uniformAnchor s L A = ((L + 1 - s : ℕ) : ℝ≥0∞)⁻¹
      * (((Finset.Icc (s) L).filter (· ∈ A)).card : ℝ≥0∞) := by
  simp only [uniformAnchor, Measure.smul_apply, Measure.coe_finsetSum, Finset.sum_apply,
    smul_eq_mul]
  congr 1
  rw [Finset.card_filter, Nat.cast_sum]
  refine Finset.sum_congr rfl fun k _ => ?_
  by_cases hk : k ∈ A <;> simp [Set.indicator, hk]

theorem uniformAnchor_univ (s L : ℕ) :
    uniformAnchor s L Set.univ = ((L + 1 - s : ℕ) : ℝ≥0∞)⁻¹ * (L + 1 - s : ℕ) := by
  simp only [uniformAnchor, Measure.smul_apply, Measure.coe_finsetSum, Finset.sum_apply,
    measure_univ, Finset.sum_const, Nat.card_Icc, nsmul_eq_mul, mul_one, smul_eq_mul]

instance (s L : ℕ) : IsFiniteMeasure (uniformAnchor s L) := by
  constructor
  rw [uniformAnchor_univ]
  rcases Nat.eq_zero_or_pos (L + 1 - s) with h | h
  · rw [h]; simp
  · rw [ENNReal.inv_mul_cancel (by exact_mod_cast h.ne') (by simp)]; exact ENNReal.one_lt_top

theorem uniformAnchor_isProb {s L : ℕ} (h : s ≤ L) : IsProbabilityMeasure (uniformAnchor s L) := by
  constructor
  rw [uniformAnchor_univ]
  exact ENNReal.inv_mul_cancel (by exact_mod_cast (anchorCount_pos h).ne') (by simp)

theorem uniformAnchor_real_apply (s L : ℕ) (A : Set ℕ) [DecidablePred (· ∈ A)] :
    (uniformAnchor s L).real A
      = (((Finset.Icc (s) L).filter (· ∈ A)).card : ℝ)
        / (L + 1 - s : ℕ) := by
  rw [measureReal_def, uniformAnchor_apply, ENNReal.toReal_mul, ENNReal.toReal_inv,
    ENNReal.toReal_natCast, ENNReal.toReal_natCast]
  ring

theorem uniformAnchor_real_le (s L i : ℕ) :
    (uniformAnchor s L).real {e | e ≤ i}
      = ((min i L + 1 - s : ℕ) : ℝ) / (L + 1 - s : ℕ) := by
  rw [uniformAnchor_real_apply]
  congr 2
  rw [show min i L + 1 - s = (Finset.Icc (s) (min i L)).card by
    rw [Nat.card_Icc]]
  congr 1
  ext k
  simp only [Finset.mem_filter, Finset.mem_Icc, Set.mem_ofPred_eq]
  omega

theorem uniformAnchor_lt (s L : ℕ) : uniformAnchor s L {e | e < s} = 0 := by
  rw [uniformAnchor_apply]
  rw [Finset.filter_eq_empty_iff.2 fun k hk => by
    simp only [Finset.mem_Icc] at hk; simp only [Set.mem_ofPred_eq]; omega]
  simp

/-- Every string the replay of `w` from `e` harvests starts as `w` does, up to some point at or
past `e`. -/
theorem spread_of_prefix (D : Measure (FreeMonoid α)) [IsFiniteMeasure D] (s L : ℕ)
    (rp : FreeMonoid α → ℕ → List (FreeMonoid α))
    (h : ∀ w e x, x ∈ rp w e →
      ∃ i, e ≤ i ∧ i ≤ x.toList.length ∧ w.toList.take i = x.toList.take i) :
    Spread D s L rp := by
  intro t
  set A := uniformAnchor s L
  have hsub : {q : FreeMonoid α × ℕ | t ∈ rp q.1 q.2} ⊆
      (Set.univ ×ˢ {e : ℕ | e < s}) ∪
      ⋃ i ∈ Finset.Icc (s) t.toList.length,
        {p : FreeMonoid α | p.toList.take i = t.toList.take i} ×ˢ {e : ℕ | e ≤ i} := by
    rintro ⟨p, e⟩ hq
    obtain ⟨i, hei, hit, htake⟩ := h p e t hq
    by_cases he : e < s
    · exact .inl ⟨Set.mem_univ _, he⟩
    · refine .inr ?_
      simp only [Set.mem_iUnion, Finset.mem_Icc, Set.mem_prod, Set.mem_ofPred_eq]
      exact ⟨i, ⟨by omega, hit⟩, htake, hei⟩
  have h0 : (D.prod A) (Set.univ ×ˢ {e : ℕ | e < s}) = 0 := by
    rw [Measure.prod_prod, uniformAnchor_lt, mul_zero]
  calc (D.prod A).real {q | t ∈ rp q.1 q.2}
      ≤ (D.prod A).real ((Set.univ ×ˢ {e : ℕ | e < s}) ∪
          ⋃ i ∈ Finset.Icc (s) t.toList.length,
          {p : FreeMonoid α | p.toList.take i = t.toList.take i} ×ˢ {e : ℕ | e ≤ i}) :=
        measureReal_mono hsub (measure_ne_top _ _)
    _ ≤ (D.prod A).real (Set.univ ×ˢ {e : ℕ | e < s}) + (D.prod A).real
          (⋃ i ∈ Finset.Icc (s) t.toList.length,
          {p : FreeMonoid α | p.toList.take i = t.toList.take i} ×ˢ {e : ℕ | e ≤ i}) :=
        measureReal_union_le _ _
    _ ≤ 0 + ∑ i ∈ Finset.Icc (s) t.toList.length, (D.prod A).real
          ({p : FreeMonoid α | p.toList.take i = t.toList.take i} ×ˢ {e : ℕ | e ≤ i}) := by
        rw [measureReal_def, h0, ENNReal.toReal_zero]
        gcongr
        exact measureReal_biUnion_finset_le _ _
    _ = _ := by
        rw [zero_add]
        refine Finset.sum_congr rfl fun i _ => ?_
        rw [measureReal_prod_prod, uniformAnchor_real_le]
        ring

theorem spread_nil (D : Measure (FreeMonoid α)) (s L : ℕ) : Spread D s L fun _ _ => [] := by
  intro t
  simp only [List.not_mem_nil, Set.ofPred_false, measureReal_empty]
  exact Finset.sum_nonneg fun i _ => mul_nonneg (by positivity) measureReal_nonneg

theorem yields_zero (D : Measure (FreeMonoid α)) (s L : ℕ)
    (rp : FreeMonoid α → ℕ → List (FreeMonoid α)) : Yields D s L rp 0 :=
  measureReal_nonneg

/-- An anchor lands on a position a draw visits a state satisfying `P` from with chance at least
`1 / (L + 1 − s)`. -/
theorem coverage_le_anchor {σ : Type*} (D : Measure (FreeMonoid α)) [IsFiniteMeasure D]
    (s L : ℕ) (M : DFA α σ) (P : σ → Prop) :
    coverage D s L M P / (L + 1 - s : ℕ)
      ≤ (D.prod (uniformAnchor s L)).real
          {q | q.2 ≤ q.1.toList.length ∧ P (M.eval (q.1.toList.take q.2))} := by
  classical
  set S := {q : FreeMonoid α × ℕ | q.2 ≤ q.1.toList.length ∧ P (M.eval (q.1.toList.take q.2))}
  set C := {x : FreeMonoid α | ∃ i, s ≤ i ∧ i ≤ L ∧ i ≤ x.toList.length ∧
    P (M.eval (x.toList.take i))}
  set c : ℝ≥0∞ := ((L + 1 - s : ℕ) : ℝ≥0∞)⁻¹
  have hS : MeasurableSet S := S.to_countable.measurableSet
  have hC : MeasurableSet C := C.to_countable.measurableSet
  have hpt : ∀ x, C.indicator (fun _ => c) x ≤ uniformAnchor s L (Prod.mk x ⁻¹' S) := by
    intro x
    by_cases hx : x ∈ C
    · rw [Set.indicator_of_mem hx, uniformAnchor_apply]
      obtain ⟨i, h1, h2, h3, h4⟩ := hx
      have : 1 ≤ ((Finset.Icc (s) L).filter (· ∈ Prod.mk x ⁻¹' S)).card :=
        Finset.card_pos.2 ⟨i, Finset.mem_filter.2 ⟨Finset.mem_Icc.2 ⟨h1, h2⟩, h3, h4⟩⟩
      calc c = c * 1 := (mul_one c).symm
        _ ≤ _ := by gcongr; exact_mod_cast this
    · rw [Set.indicator_of_notMem hx]; exact bot_le
  have hle : c * D C ≤ (D.prod (uniformAnchor s L)) S := by
    rw [Measure.prod_apply hS, ← lintegral_indicator_const hC]
    exact lintegral_mono hpt
  calc coverage D s L M P / (L + 1 - s : ℕ)
      = (c * D C).toReal := by
        rw [ENNReal.toReal_mul, ENNReal.toReal_inv, ENNReal.toReal_natCast]
        unfold coverage
        rw [measureReal_def]
        ring
    _ ≤ (D.prod (uniformAnchor s L)).real S := ENNReal.toReal_mono (measure_ne_top _ _) hle

/-- The ending is the same at any replay that is spread and yields. -/
theorem EndsWell.respread {σ : Type*} {M : DFA α σ} {bad : σ → Prop}
    {D : Measure (FreeMonoid α)} {ε : ℝ} {dis : Set (FreeMonoid α)} {s L s' L' : ℕ}
    {rp rp' : FreeMonoid α → ℕ → List (FreeMonoid α)} {y y' : ℝ} (hs : Spread D s L rp)
    (hy : Yields D s L rp y) {e : RoundEnd α}
    (h : EndsWell M bad D ε dis s' L' rp' y' e) : EndsWell M bad D ε dis s L rp y e := by
  cases e with
  | consistent => exact h
  | harvest zs => exact ⟨h.1, hs, hy⟩
  | failed => exact h

end OrthoDFA
