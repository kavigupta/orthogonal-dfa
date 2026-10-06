import OrthoDFA.Clustering
import OrthoDFA.Proofs.BinomTail

/-!
# What `drift_verdict` certifies

Three facts about `misAdmits`, none of which mentions the run:

* a Clopper-Pearson interval misses a binomial's rate with probability at most its level
  (`sum_not_cpCovers_le`);
* over i.i.d. draws, the number landing on a side of the cut is binomial, and given that number
  so is the number of those reading 1 (`pi_count_eq`), so each of `misAdmits`' four intervals
  misses its rate with probability at most its level;
* intervals that hold the rates bound `misShare` by its value at the counts, plus a width
  (`misAdmits_of_point`), which is how the gate is shown to pass.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory Finset
open scoped ENNReal unitInterval

/-! ## A Clopper-Pearson interval misses at its level -/

lemma binomTerm_nonneg (k i : ℕ) {a : ℝ} (ha0 : 0 ≤ a) (ha1 : a ≤ 1) :
    0 ≤ (k.choose i : ℝ) * a ^ i * (1 - a) ^ (k - i) := by
  have : 0 ≤ 1 - a := by linarith
  positivity

lemma binomSfGe_anti (k : ℕ) {a : ℝ} (ha0 : 0 ≤ a) (ha1 : a ≤ 1) {h h' : ℕ} (hh : h ≤ h') :
    binomSfGe k a h' ≤ binomSfGe k a h :=
  Finset.sum_le_sum_of_subset_of_nonneg (Finset.Icc_subset_Icc_left hh)
    (fun i _ _ => binomTerm_nonneg k i ha0 ha1)

lemma binomCdfLe_mono (k : ℕ) {a : ℝ} (ha0 : 0 ≤ a) (ha1 : a ≤ 1) {h h' : ℕ} (hh : h ≤ h') :
    binomCdfLe k a h ≤ binomCdfLe k a h' :=
  Finset.sum_le_sum_of_subset_of_nonneg (Finset.range_subset_range.2 (by omega))
    (fun i _ _ => binomTerm_nonneg k i ha0 ha1)

open scoped Classical in
/-- The upper tail read at its own count is a p-value: it falls below `c` with probability at
most `c`. -/
theorem sum_sf_lt_le (k : ℕ) {a : ℝ} (ha0 : 0 ≤ a) (ha1 : a ≤ 1) (c : ℝ) (hc : 0 ≤ c) :
    ∑ h ∈ range (k + 1), (if binomSfGe k a h < c
      then (k.choose h : ℝ) * a ^ h * (1 - a) ^ (k - h) else 0) ≤ c := by
  classical
  rw [← Finset.sum_filter]
  set H := (range (k + 1)).filter (fun h => binomSfGe k a h < c) with hH
  rcases H.eq_empty_or_nonempty with he | hne
  · rw [he, Finset.sum_empty]; exact hc
  set h₀ := H.min' hne with hh₀
  have hmem := Finset.min'_mem H hne
  rw [← hh₀] at hmem
  have hEq : H = Icc h₀ k := by
    ext h
    simp only [hH, Finset.mem_filter, Finset.mem_range, Finset.mem_Icc]
    constructor
    · rintro ⟨hk, hlt⟩
      refine ⟨Finset.min'_le H h ?_, by omega⟩
      exact Finset.mem_filter.2 ⟨Finset.mem_range.2 hk, hlt⟩
    · rintro ⟨h1, h2⟩
      refine ⟨by omega, lt_of_le_of_lt (binomSfGe_anti k ha0 ha1 h1) ?_⟩
      exact (Finset.mem_filter.1 hmem).2
  rw [hEq]
  exact le_of_lt (by rw [hEq] at hmem; exact (Finset.mem_filter.1 (hEq ▸ hmem : h₀ ∈ H)).2)

open scoped Classical in
/-- The lower tail likewise. -/
theorem sum_cdf_lt_le (k : ℕ) {a : ℝ} (ha0 : 0 ≤ a) (ha1 : a ≤ 1) (c : ℝ) (hc : 0 ≤ c) :
    ∑ h ∈ range (k + 1), (if binomCdfLe k a h < c
      then (k.choose h : ℝ) * a ^ h * (1 - a) ^ (k - h) else 0) ≤ c := by
  classical
  rw [← Finset.sum_filter]
  set H := (range (k + 1)).filter (fun h => binomCdfLe k a h < c) with hH
  rcases H.eq_empty_or_nonempty with he | hne
  · rw [he, Finset.sum_empty]; exact hc
  set h₀ := H.max' hne with hh₀
  have hmem := Finset.max'_mem H hne
  rw [← hh₀] at hmem
  have hk₀ : h₀ < k + 1 := Finset.mem_range.1 (Finset.mem_filter.1 hmem).1
  have hEq : H = range (h₀ + 1) := by
    ext h
    simp only [hH, Finset.mem_filter, Finset.mem_range]
    constructor
    · rintro ⟨hk, hlt⟩
      have := Finset.le_max' H h (Finset.mem_filter.2 ⟨Finset.mem_range.2 hk, hlt⟩)
      omega
    · intro hh
      refine ⟨by omega, lt_of_le_of_lt (binomCdfLe_mono k ha0 ha1 (by omega : h ≤ h₀)) ?_⟩
      exact (Finset.mem_filter.1 hmem).2
  rw [hEq]
  exact le_of_lt (Finset.mem_filter.1 hmem).2

open scoped Classical in
/-- `clopper_pearson`'s interval at `level` misses the rate with probability at most `level`. -/
theorem sum_not_cpCovers_le (k : ℕ) {a : ℝ} (ha0 : 0 ≤ a) (ha1 : a ≤ 1) (level : ℝ)
    (hl : 0 ≤ level) :
    ∑ h ∈ range (k + 1), (if ¬ cpCovers k h level a
      then (k.choose h : ℝ) * a ^ h * (1 - a) ^ (k - h) else 0) ≤ level := by
  classical
  have hsplit : ∀ h ∈ range (k + 1), (if ¬ cpCovers k h level a
      then (k.choose h : ℝ) * a ^ h * (1 - a) ^ (k - h) else 0)
      ≤ (if binomSfGe k a h < level / 2
          then (k.choose h : ℝ) * a ^ h * (1 - a) ^ (k - h) else 0)
        + (if binomCdfLe k a h < level / 2
          then (k.choose h : ℝ) * a ^ h * (1 - a) ^ (k - h) else 0) := by
    intro h _
    have ht := binomTerm_nonneg k h ha0 ha1
    by_cases hc : cpCovers k h level a
    · rw [if_neg (not_not.2 hc)]
      split_ifs <;> linarith
    · rw [if_pos hc]
      unfold cpCovers at hc
      rw [not_and_or, not_le, not_le] at hc
      rcases hc with h1 | h2
      · rw [if_pos h1]; split_ifs <;> linarith
      · rw [if_pos h2]; split_ifs <;> linarith
  refine le_trans (Finset.sum_le_sum hsplit) ?_
  rw [Finset.sum_add_distrib]
  linarith [sum_sf_lt_le k ha0 ha1 (level / 2) (by linarith),
    sum_cdf_lt_le k ha0 ha1 (level / 2) (by linarith)]

/-! ## The counts over i.i.d. draws -/

section Count

variable {β : Type*}

open scoped Classical in
/-- The draws whose membership in `A` and in `A ∩ B` the index sets `K ⊇ H` record. -/
lemma pattern_eq_pi (A B : Set β) {n : ℕ} (K H : Finset (Fin n)) (hHK : H ⊆ K) :
    {x : Fin n → β | univ.filter (fun i => x i ∈ A) = K
        ∧ univ.filter (fun i => x i ∈ A ∧ x i ∈ B) = H}
      = Set.pi Set.univ (fun i => if i ∈ H then A ∩ B else if i ∈ K then A \ B else Aᶜ) := by
  classical
  ext x
  simp only [Set.mem_setOf_eq, Set.mem_pi, Set.mem_univ, forall_const]
  constructor
  · rintro ⟨hK, hH⟩ i
    by_cases hi : i ∈ H
    · rw [if_pos hi]
      rw [← hH] at hi
      exact (Finset.mem_filter.1 hi).2
    · rw [if_neg hi]
      by_cases hk : i ∈ K
      · rw [if_pos hk]
        rw [← hK] at hk
        refine ⟨(Finset.mem_filter.1 hk).2, fun hb => hi ?_⟩
        rw [← hH]
        exact Finset.mem_filter.2 ⟨Finset.mem_univ _, (Finset.mem_filter.1 hk).2, hb⟩
      · rw [if_neg hk]
        intro ha
        exact hk (hK ▸ Finset.mem_filter.2 ⟨Finset.mem_univ _, ha⟩)
  · intro hx
    have hA : ∀ i, x i ∈ A ↔ i ∈ K := by
      intro i
      have := hx i
      by_cases hi : i ∈ H
      · rw [if_pos hi] at this
        exact ⟨fun _ => hHK hi, fun _ => this.1⟩
      · rw [if_neg hi] at this
        by_cases hk : i ∈ K
        · rw [if_pos hk] at this
          exact ⟨fun _ => hk, fun _ => this.1⟩
        · rw [if_neg hk] at this
          exact ⟨fun h => absurd h this, fun h => absurd h hk⟩
    have hB : ∀ i, (x i ∈ A ∧ x i ∈ B) ↔ i ∈ H := by
      intro i
      have := hx i
      by_cases hi : i ∈ H
      · rw [if_pos hi] at this
        exact ⟨fun _ => hi, fun _ => this⟩
      · rw [if_neg hi] at this
        by_cases hk : i ∈ K
        · rw [if_pos hk] at this
          exact ⟨fun h => absurd h.2 this.2, fun h => absurd h hi⟩
        · rw [if_neg hk] at this
          exact ⟨fun h => absurd h.1 this, fun h => absurd h hi⟩
    refine ⟨?_, ?_⟩
    · ext i; simp [hA i]
    · ext i; simp [hB i]

open scoped Classical in
lemma prod_pattern {n : ℕ} (K H : Finset (Fin n)) (hHK : H ⊆ K) (w₁₁ w₁₀ w₀ : ℝ) :
    ∏ i, (if i ∈ H then w₁₁ else if i ∈ K then w₁₀ else w₀)
      = w₁₁ ^ H.card * w₁₀ ^ (K.card - H.card) * w₀ ^ (n - K.card) := by
  classical
  set t : Fin n → ℝ := fun i => if i ∈ H then w₁₁ else if i ∈ K then w₁₀ else w₀ with ht
  have h1 : (∏ i ∈ univ \ K, t i) * ∏ i ∈ K, t i = ∏ i, t i :=
    Finset.prod_sdiff (Finset.subset_univ K)
  have h2 : (∏ i ∈ K \ H, t i) * ∏ i ∈ H, t i = ∏ i ∈ K, t i := Finset.prod_sdiff hHK
  have e1 : ∏ i ∈ univ \ K, t i = w₀ ^ (n - K.card) := by
    rw [Finset.prod_congr rfl (fun i hi => by
      have hk : i ∉ K := (Finset.mem_sdiff.1 hi).2
      have hh : i ∉ H := fun h => hk (hHK h)
      simp only [ht, if_neg hh, if_neg hk] : ∀ i ∈ univ \ K, t i = w₀), Finset.prod_const,
      Finset.card_sdiff_of_subset (Finset.subset_univ K), Finset.card_univ, Fintype.card_fin]
  have e2 : ∏ i ∈ K \ H, t i = w₁₀ ^ (K.card - H.card) := by
    rw [Finset.prod_congr rfl (fun i hi => by
      have hk : i ∈ K := (Finset.mem_sdiff.1 hi).1
      have hh : i ∉ H := (Finset.mem_sdiff.1 hi).2
      simp only [ht, if_neg hh, if_pos hk] : ∀ i ∈ K \ H, t i = w₁₀), Finset.prod_const,
      Finset.card_sdiff_of_subset hHK]
  have e3 : ∏ i ∈ H, t i = w₁₁ ^ H.card := by
    rw [Finset.prod_congr rfl (fun i hi => by simp only [ht, if_pos hi] :
      ∀ i ∈ H, t i = w₁₁), Finset.prod_const]
  rw [← h1, ← h2, e1, e2, e3]
  ring

open scoped Classical in
/-- Over `n` i.i.d. draws, the chance that the counts in `A ∩ B` and in `A` satisfy `F`: the
count in `A` is binomial, and given it, so is the count in `B` among those. -/
theorem pi_count_eq [MeasurableSpace β] [Countable β] [MeasurableSingletonClass β]
    (ν : Measure β) [IsProbabilityMeasure ν] (A B : Set β) (n : ℕ)
    (F : ℕ → ℕ → Prop) :
    (Measure.pi fun _ : Fin n => ν).real
        {x | F (univ.filter (fun i => x i ∈ A ∧ x i ∈ B)).card
          (univ.filter (fun i => x i ∈ A)).card}
      = ∑ k ∈ range (n + 1), (n.choose k : ℝ) * (ν.real Aᶜ ^ (n - k)
          * ∑ h ∈ range (k + 1), (k.choose h : ℝ) * (if F h k then
              ν.real (A ∩ B) ^ h * ν.real (A \ B) ^ (k - h) else 0)) := by
  classical
  set w₁₁ := ν.real (A ∩ B)
  set w₁₀ := ν.real (A \ B)
  set w₀ := ν.real Aᶜ
  set s : Finset (Finset (Fin n) × Finset (Fin n)) :=
    (univ.powerset ×ˢ univ.powerset).filter (fun KH => KH.2 ⊆ KH.1 ∧ F KH.2.card KH.1.card)
    with hs
  set f : Finset (Fin n) × Finset (Fin n) → Set (Fin n → β) := fun KH =>
    {x | univ.filter (fun i => x i ∈ A) = KH.1 ∧ univ.filter (fun i => x i ∈ A ∧ x i ∈ B) = KH.2}
    with hf
  have hcov : {x : Fin n → β | F (univ.filter (fun i => x i ∈ A ∧ x i ∈ B)).card
        (univ.filter (fun i => x i ∈ A)).card} = ⋃ KH ∈ s, f KH := by
    ext x
    simp only [Set.mem_setOf_eq, Set.mem_iUnion, hs, hf, Finset.mem_filter, Finset.mem_product,
      Finset.mem_powerset, exists_prop]
    constructor
    · intro h
      refine ⟨(univ.filter (fun i => x i ∈ A), univ.filter (fun i => x i ∈ A ∧ x i ∈ B)),
        ⟨⟨Finset.subset_univ _, Finset.subset_univ _⟩, ?_, h⟩, rfl, rfl⟩
      intro i hi
      exact Finset.mem_filter.2 ⟨Finset.mem_univ _, (Finset.mem_filter.1 hi).2.1⟩
    · rintro ⟨KH, ⟨-, -, hF⟩, hK, hH⟩
      rw [hK, hH]
      exact hF
  have hdisj : Set.PairwiseDisjoint (↑s) f := by
    intro KH _ KH' _ hne
    refine Set.disjoint_left.2 (fun x hx hx' => hne ?_)
    exact Prod.ext (hx.1.symm.trans hx'.1) (hx.2.symm.trans hx'.2)
  have hmeas : ∀ KH ∈ s, MeasurableSet (f KH) := fun KH _ => (Set.to_countable _).measurableSet
  have hterm : ∀ KH ∈ s, (Measure.pi fun _ : Fin n => ν).real (f KH)
      = w₁₁ ^ KH.2.card * w₁₀ ^ (KH.1.card - KH.2.card) * w₀ ^ (n - KH.1.card) := by
    intro KH hKH
    have hHK : KH.2 ⊆ KH.1 := (Finset.mem_filter.1 hKH).2.1
    rw [hf]
    simp only
    rw [pattern_eq_pi A B KH.1 KH.2 hHK, measureReal_def, Measure.pi_pi, ENNReal.toReal_prod,
      ← prod_pattern KH.1 KH.2 hHK w₁₁ w₁₀ w₀]
    refine Finset.prod_congr rfl (fun i _ => ?_)
    split_ifs <;> rfl
  rw [hcov, measureReal_biUnion_finset hdisj hmeas, Finset.sum_congr rfl hterm, hs,
    Finset.sum_filter, Finset.sum_product]
  have hinner : ∀ K ∈ (univ : Finset (Fin n)).powerset,
      ∑ H ∈ (univ : Finset (Fin n)).powerset, (if H ⊆ K ∧ F H.card K.card then
          w₁₁ ^ H.card * w₁₀ ^ (K.card - H.card) * w₀ ^ (n - K.card) else 0)
        = w₀ ^ (n - K.card) * ∑ h ∈ range (K.card + 1), (K.card.choose h : ℝ)
            * (if F h K.card then w₁₁ ^ h * w₁₀ ^ (K.card - h) else 0) := by
    intro K _
    have hfilt : (univ : Finset (Fin n)).powerset.filter (fun H => H ⊆ K) = K.powerset := by
      ext H; simp
    rw [← Finset.sum_filter_add_sum_filter_not _ (fun H => H ⊆ K)]
    rw [Finset.sum_eq_zero (s := (univ : Finset (Fin n)).powerset.filter (fun H => ¬ H ⊆ K))
      (fun H hH => by rw [if_neg (fun h => (Finset.mem_filter.1 hH).2 h.1)]), add_zero, hfilt]
    have hpw := Finset.sum_powerset_apply_card
      (fun m => if F m K.card then w₁₁ ^ m * w₁₀ ^ (K.card - m) else 0) (x := K)
    rw [Finset.mul_sum]
    simp only [nsmul_eq_mul] at hpw
    rw [Finset.sum_congr rfl (fun H hH => by
      rw [if_congr (and_iff_right (Finset.mem_powerset.1 hH)) rfl rfl]),
      show (∑ H ∈ K.powerset, if F H.card K.card then
          w₁₁ ^ H.card * w₁₀ ^ (K.card - H.card) * w₀ ^ (n - K.card) else 0)
        = w₀ ^ (n - K.card) * ∑ H ∈ K.powerset,
          (if F H.card K.card then w₁₁ ^ H.card * w₁₀ ^ (K.card - H.card) else 0) from by
        rw [Finset.mul_sum]
        refine Finset.sum_congr rfl (fun H _ => ?_)
        split_ifs <;> ring,
      hpw, Finset.mul_sum]
  rw [Finset.sum_congr rfl hinner]
  have hpw := Finset.sum_powerset_apply_card
    (fun k => w₀ ^ (n - k) * ∑ h ∈ range (k + 1), (k.choose h : ℝ)
      * (if F h k then w₁₁ ^ h * w₁₀ ^ (k - h) else 0)) (x := (univ : Finset (Fin n)))
  simp only [nsmul_eq_mul, Finset.card_univ, Fintype.card_fin] at hpw
  rw [hpw]

variable [MeasurableSpace β] [Countable β] [MeasurableSingletonClass β]

open scoped Classical in
/-- The rate interval on the side `A` holds `A`'s rate of `B`, except at its level. -/
theorem pi_rate_miss_le (ν : Measure β) [IsProbabilityMeasure ν] (A B : Set β) (n : ℕ)
    (level : ℝ) (hl : 0 ≤ level) :
    (Measure.pi fun _ : Fin n => ν).real
        {x | ¬ cpCovers (univ.filter (fun i => x i ∈ A)).card
          (univ.filter (fun i => x i ∈ A ∧ x i ∈ B)).card level
          (ν.real (A ∩ B) / ν.real A)} ≤ level := by
  classical
  set m := ν.real A with hm
  set a := ν.real (A ∩ B) / m with ha
  have hmeasB : MeasurableSet B := (Set.to_countable _).measurableSet
  have hmeasA : MeasurableSet A := (Set.to_countable _).measurableSet
  have hm0 : 0 ≤ m := measureReal_nonneg
  have h11 : 0 ≤ ν.real (A ∩ B) := measureReal_nonneg
  have h11m : ν.real (A ∩ B) ≤ m := measureReal_mono Set.inter_subset_left
  have hsum : ν.real (A ∩ B) + ν.real (A \ B) = m := measureReal_inter_add_sdiff hmeasB
  have hcomp : ν.real Aᶜ = 1 - m := by
    rw [measureReal_compl hmeasA, probReal_univ]
  have hw11 : ν.real (A ∩ B) = m * a := by
    rcases eq_or_lt_of_le hm0 with h0 | hpos
    · rw [← h0, zero_mul]; linarith
    · rw [ha]; field_simp
  have hw10 : ν.real (A \ B) = m * (1 - a) := by linarith
  have ha0 : 0 ≤ a := div_nonneg h11 hm0
  have ha1 : a ≤ 1 := by
    rcases eq_or_lt_of_le hm0 with h0 | hpos
    · rw [ha, ← h0, div_zero]; norm_num
    · rw [ha, div_le_one hpos]; exact h11m
  have h := pi_count_eq ν A B n (fun h k => ¬ cpCovers k h level a)
  rw [h, hw11, hw10, hcomp]
  have h1m : 0 ≤ 1 - m := by linarith [measureReal_le_one (μ := ν) (s := A)]
  calc _ ≤ ∑ k ∈ range (n + 1), (n.choose k : ℝ) * ((1 - m) ^ (n - k) * (m ^ k * level)) := by
        refine Finset.sum_le_sum (fun k _ => ?_)
        refine mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left ?_ (pow_nonneg h1m _))
          (Nat.cast_nonneg _)
        calc _ = m ^ k * ∑ h ∈ range (k + 1), (if ¬ cpCovers k h level a
                then (k.choose h : ℝ) * a ^ h * (1 - a) ^ (k - h) else 0) := by
              rw [Finset.mul_sum]
              refine Finset.sum_congr rfl (fun h hh => ?_)
              have hk : h ≤ k := Nat.lt_succ_iff.1 (Finset.mem_range.1 hh)
              split_ifs
              · ring
              · rw [mul_pow, mul_pow, show m ^ k = m ^ h * m ^ (k - h) by
                  rw [← pow_add, Nat.add_sub_cancel' hk]]
                ring
          _ ≤ m ^ k * level :=
              mul_le_mul_of_nonneg_left (sum_not_cpCovers_le k ha0 ha1 level hl) (pow_nonneg hm0 k)
    _ = level * (m + (1 - m)) ^ n := by
        rw [add_pow, Finset.mul_sum]
        refine Finset.sum_congr rfl (fun k _ => ?_)
        ring
    _ = level := by rw [show m + (1 - m) = 1 by ring, one_pow, mul_one]

open scoped Classical in
/-- The mass interval holds the side's mass, except at its level. -/
theorem pi_mass_miss_le (ν : Measure β) [IsProbabilityMeasure ν] (A : Set β) (n : ℕ)
    (level : ℝ) (hl : 0 ≤ level) :
    (Measure.pi fun _ : Fin n => ν).real
        {x | ¬ cpCovers n (univ.filter (fun i => x i ∈ A)).card level (ν.real A)} ≤ level := by
  classical
  set m := ν.real A with hm
  have hmeasA : MeasurableSet A := (Set.to_countable _).measurableSet
  have hm0 : 0 ≤ m := measureReal_nonneg
  have hm1 : m ≤ 1 := measureReal_le_one
  have hsum : ν.real (A ∩ A) + ν.real (A \ A) = m := measureReal_inter_add_sdiff hmeasA
  have hcomp : ν.real Aᶜ = 1 - m := by
    rw [measureReal_compl hmeasA, probReal_univ]
  set w₁₁ := ν.real (A ∩ A)
  set w₁₀ := ν.real (A \ A)
  have h := pi_count_eq ν A A n (fun _ k => ¬ cpCovers n k level m)
  have hset : {x : Fin n → β | ¬ cpCovers n (univ.filter (fun i => x i ∈ A)).card level m}
      = {x | (fun _ k => ¬ cpCovers n k level m)
          (univ.filter (fun i => x i ∈ A ∧ x i ∈ A)).card
          (univ.filter (fun i => x i ∈ A)).card} := rfl
  rw [hset, h, hcomp]
  calc _ = ∑ k ∈ range (n + 1), (if ¬ cpCovers n k level m
          then (n.choose k : ℝ) * m ^ k * (1 - m) ^ (n - k) else 0) := by
        refine Finset.sum_congr rfl (fun k _ => ?_)
        by_cases hc : cpCovers n k level m
        · simp [hc]
        · simp only [hc, not_false_eq_true, if_true]
          rw [show m ^ k = (w₁₁ + w₁₀) ^ k by rw [hsum], add_pow]
          simp only [Finset.mul_sum, Finset.sum_mul]
          exact Finset.sum_congr rfl (fun h _ => by ring)
    _ ≤ level := sum_not_cpCovers_le n hm0 hm1 level hl

end Count

/-! ## Intervals that hold the rates bound the share -/

/-- Hoeffding's bound on the binomial upper tail. -/
theorem binomSfGe_le (n j : ℕ) (θ τ : ℝ) (hθ0 : 0 ≤ θ) (hθ1 : θ ≤ 1) (hτ : 0 ≤ τ)
    (h : (n : ℝ) * (θ + τ) ≤ j) :
    binomSfGe n θ j ≤ Real.exp (-2 * (n : ℝ) * τ ^ 2) := by
  classical
  set p : unitInterval := ⟨θ, hθ0, hθ1⟩ with hp
  have hbin : binomSfGe n θ j = (ProbabilityTheory.binomial n p).real ↑(Finset.Icc j n) := by
    rw [binomial_real_finset]
    rfl
  rw [hbin]
  refine le_trans (measureReal_mono ?_ (measure_ne_top _ _)) (binomial_real_ge_le n p τ hτ)
  intro i hi
  simp only [Finset.coe_Icc, Set.mem_Icc] at hi
  have : (j : ℝ) ≤ (i : ℝ) := by exact_mod_cast hi.1
  exact le_trans h this

/-- The lower tail is the upper tail of the complementary rate. -/
lemma binomCdfLe_eq (n h : ℕ) (hh : h ≤ n) (p : ℝ) :
    binomCdfLe n p h = binomSfGe n (1 - p) (n - h) := by
  unfold binomCdfLe binomSfGe
  refine Finset.sum_nbij' (fun i => n - i) (fun i => n - i) ?_ ?_ ?_ ?_ ?_
  · intro i hi
    simp only [Finset.mem_range, Finset.mem_Icc] at hi ⊢
    omega
  · intro i hi
    simp only [Finset.mem_range, Finset.mem_Icc] at hi ⊢
    omega
  · intro i hi
    simp only [Finset.mem_range] at hi
    omega
  · intro i hi
    simp only [Finset.mem_Icc] at hi
    omega
  · intro i hi
    have hi' : i ≤ n := by have := Finset.mem_range.1 hi; omega
    rw [Nat.choose_symm hi', Nat.sub_sub_self hi', show (1 : ℝ) - (1 - p) = p by ring]
    ring

/-- Hoeffding's bound on the binomial lower tail. -/
theorem binomCdfLe_le (n h : ℕ) (hh : h ≤ n) (p τ : ℝ) (hp0 : 0 ≤ p) (hp1 : p ≤ 1) (hτ : 0 ≤ τ)
    (hle : (h : ℝ) ≤ (n : ℝ) * (p - τ)) :
    binomCdfLe n p h ≤ Real.exp (-2 * (n : ℝ) * τ ^ 2) := by
  rw [binomCdfLe_eq n h hh]
  refine binomSfGe_le n (n - h) (1 - p) τ (by linarith) (by linarith) hτ ?_
  rw [Nat.cast_sub hh]
  linarith

/-- A rate an interval holds lies within the interval's half-width of the count's own rate,
from below. -/
lemma cover_dev_lo {n h : ℕ} (hh : h ≤ n) {level p : ℝ} (hlev : 0 < level) (hp0 : 0 ≤ p)
    (hp1 : p ≤ 1) (hc : cpCovers n h level p) {N t : ℝ} (hn : (n : ℝ) ≤ N) (ht : 0 ≤ t)
    (hL : Real.log (2 / level) ≤ 2 * N * t ^ 2) :
    (h : ℝ) - n * p ≤ N * t := by
  by_contra hcon
  push_neg at hcon
  have hNt : 0 ≤ N * t := by
    have : (0 : ℝ) ≤ n := Nat.cast_nonneg _
    nlinarith
  have hnpos : (0 : ℝ) < n := by
    rcases eq_or_lt_of_le (Nat.cast_nonneg n : (0 : ℝ) ≤ n) with h0 | hpos
    · have : (h : ℝ) ≤ n := by exact_mod_cast hh
      rw [← h0] at hcon this
      linarith
    · exact hpos
  set τ := ((h : ℝ) - n * p) / n with hτ
  have hτ0 : 0 ≤ τ := div_nonneg (by linarith) hnpos.le
  have hsf := binomSfGe_le n h p τ hp0 hp1 hτ0 (by rw [hτ]; field_simp; linarith)
  have hexp : level / 2 ≤ Real.exp (-2 * (n : ℝ) * τ ^ 2) := le_trans hc.1 hsf
  have hlog : 2 * (n : ℝ) * τ ^ 2 ≤ Real.log (2 / level) := by
    have := Real.log_le_log (by positivity) hexp
    rw [Real.log_exp, Real.log_div hlev.ne' (by norm_num)] at this
    rw [Real.log_div (by norm_num) hlev.ne']
    linarith
  have hsq : ((h : ℝ) - n * p) ^ 2 ≤ (N * t) ^ 2 := by
    have e : ((h : ℝ) - n * p) ^ 2 = (n : ℝ) * ((n : ℝ) * τ ^ 2) := by
      rw [hτ]; field_simp
    rw [e]
    have h1 : (n : ℝ) * τ ^ 2 ≤ N * t ^ 2 := by linarith
    have h2 : (n : ℝ) * ((n : ℝ) * τ ^ 2) ≤ N * (N * t ^ 2) :=
      mul_le_mul hn h1 (by positivity) (by linarith)
    nlinarith
  nlinarith

/-- From above. -/
lemma cover_dev_hi {n h : ℕ} (hh : h ≤ n) {level p : ℝ} (hlev : 0 < level) (hp0 : 0 ≤ p)
    (hp1 : p ≤ 1) (hc : cpCovers n h level p) {N t : ℝ} (hn : (n : ℝ) ≤ N) (ht : 0 ≤ t)
    (hL : Real.log (2 / level) ≤ 2 * N * t ^ 2) :
    n * p - (h : ℝ) ≤ N * t := by
  by_contra hcon
  push_neg at hcon
  have hNt : 0 ≤ N * t := by
    have : (0 : ℝ) ≤ n := Nat.cast_nonneg _
    nlinarith
  have hnpos : (0 : ℝ) < n := by
    rcases eq_or_lt_of_le (Nat.cast_nonneg n : (0 : ℝ) ≤ n) with h0 | hpos
    · rw [← h0] at hcon
      have : (0 : ℝ) ≤ h := Nat.cast_nonneg _
      linarith
    · exact hpos
  set τ := ((n : ℝ) * p - h) / n with hτ
  have hτ0 : 0 ≤ τ := div_nonneg (by linarith) hnpos.le
  have hcdf := binomCdfLe_le n h hh p τ hp0 hp1 hτ0 (by rw [hτ]; field_simp; linarith)
  have hexp : level / 2 ≤ Real.exp (-2 * (n : ℝ) * τ ^ 2) := le_trans hc.2 hcdf
  have hlog : 2 * (n : ℝ) * τ ^ 2 ≤ Real.log (2 / level) := by
    have := Real.log_le_log (by positivity) hexp
    rw [Real.log_exp, Real.log_div hlev.ne' (by norm_num)] at this
    rw [Real.log_div (by norm_num) hlev.ne']
    linarith
  have hsq : ((n : ℝ) * p - h) ^ 2 ≤ (N * t) ^ 2 := by
    have e : ((n : ℝ) * p - h) ^ 2 = (n : ℝ) * ((n : ℝ) * τ ^ 2) := by
      rw [hτ]; field_simp
    rw [e]
    have h1 : (n : ℝ) * τ ^ 2 ≤ N * t ^ 2 := by linarith
    have h2 : (n : ℝ) * ((n : ℝ) * τ ^ 2) ≤ N * (N * t ^ 2) :=
      mul_le_mul hn h1 (by positivity) (by linarith)
    nlinarith
  nlinarith

/-- One side's term: a mass at most `k/N + t` reading `x`, with `k x` short of the side's hits
by at most `N t`, misclassifies at most what its counts read as, plus `(p + 1) t`. -/
lemma side_le {N k h q x p t : ℝ} (hN : 0 < N) (hk : 0 ≤ k) (hx0 : 0 ≤ x)
    (hp : 0 ≤ p) (ht : 0 ≤ t) (hq : q ≤ k / N + t) (hdev : h - k * x ≤ N * t) :
    q * max 0 (p - x) ≤ max 0 (p * k - h) / N + (p + 1) * t := by
  have hm0 : 0 ≤ max 0 (p - x) := le_max_left _ _
  have h1 : q * max 0 (p - x) ≤ (k / N + t) * max 0 (p - x) := mul_le_mul_of_nonneg_right hq hm0
  have h2 : k / N * max 0 (p - x) = max 0 (p * k - k * x) / N := by
    rw [div_mul_eq_mul_div, mul_max_of_nonneg _ _ hk]
    congr 1
    rw [mul_zero]; congr 1; ring
  have h3 : max 0 (p * k - k * x) ≤ max 0 (p * k - h) + N * t := by
    rcases le_total 0 (p * k - h) with hc | hc
    · rw [max_eq_right hc]
      exact max_le (by nlinarith) (by linarith)
    · rw [max_eq_left hc]
      exact max_le (by nlinarith) (by linarith)
  have h4 : t * max 0 (p - x) ≤ t * p :=
    mul_le_mul_of_nonneg_left (max_le hp (by linarith)) ht
  have h5 : max 0 (p * k - k * x) / N ≤ max 0 (p * k - h) / N + t := by
    rw [div_le_iff₀ hN, add_mul, div_mul_cancel₀ _ hN.ne']
    linarith
  nlinarith

/-- The gate passes once the counts read as misclassifying little enough that their intervals'
widths still fit under the limit.  `misShare` is linear in each side's deficit, so the intervals
widen it by at most `t` on each side's mass and rate, `N t` in counts. -/
theorem misAdmits_of_point (η₀ level limit t : ℝ) (hA nA hR nR : ℕ) (hη0 : 0 ≤ η₀)
    (hη : η₀ < 1 / 2) (hlev : 0 < level) (hhA : hA ≤ nA) (hhR : hR ≤ nR) (hn : 0 < nA + nR)
    (ht : 0 ≤ t) (hL : Real.log (2 / level) ≤ 2 * ((nA + nR : ℕ) : ℝ) * t ^ 2)
    (hpt : (max 0 ((1 - η₀) * nA - hA) + max 0 (hR - η₀ * nR)) / ((nA + nR : ℕ) : ℝ) + 4 * t
      ≤ (1 - 2 * η₀) * limit) :
    misAdmits η₀ level limit ((hA, nA), (hR, nR)) := by
  intro m a r hm ha hr cm cmR ca cr
  simp only at cm cmR ca cr
  set N : ℝ := ((nA + nR : ℕ) : ℝ) with hNdef
  have hN : 0 < N := by rw [hNdef]; exact_mod_cast hn
  have hNsum : N = (nA : ℝ) + nR := by rw [hNdef]; push_cast; ring
  have hg : 0 < 1 - 2 * η₀ := by linarith
  have hnA : (nA : ℝ) ≤ N := by rw [hNsum]; linarith [(Nat.cast_nonneg nR : (0 : ℝ) ≤ nR)]
  have hnR : (nR : ℝ) ≤ N := by rw [hNsum]; linarith [(Nat.cast_nonneg nA : (0 : ℝ) ≤ nA)]
  -- the mass, from each side's count
  have hmA : m ≤ nA / N + t := by
    have := cover_dev_hi (Nat.le_add_right nA nR) hlev hm.1 hm.2 cm le_rfl ht hL
    rw [← hNdef] at this
    rw [div_add' _ _ _ hN.ne', le_div_iff₀ hN]
    linarith
  have hmR : 1 - m ≤ nR / N + t := by
    have := cover_dev_hi (Nat.le_add_left nR nA) hlev (by linarith [hm.2]) (by linarith [hm.1])
      cmR le_rfl ht hL
    rw [← hNdef] at this
    rw [div_add' _ _ _ hN.ne', le_div_iff₀ hN]
    linarith
  -- the rates, from each side's hits
  have hdA : (hA : ℝ) - nA * a ≤ N * t := cover_dev_lo hhA hlev ha.1 ha.2 ca hnA ht hL
  have hdR : (nR : ℝ) * r - hR ≤ N * t := cover_dev_hi hhR hlev hr.1 hr.2 cr hnR ht hL
  have hsA := side_le (p := 1 - η₀) hN (Nat.cast_nonneg nA) ha.1 (by linarith) ht hmA hdA
  have hhR' : (hR : ℝ) ≤ nR := by exact_mod_cast hhR
  have hsR := side_le (p := 1 - η₀) (h := (nR : ℝ) - hR) (x := 1 - r) hN
    (Nat.cast_nonneg nR) (by linarith [hr.2]) (by linarith) ht hmR (by linarith)
  have eR : max 0 ((1 - η₀) * nR - ((nR : ℝ) - hR)) = max 0 (hR - η₀ * nR) := by
    congr 1; ring
  have eRx : max 0 ((1 - η₀) - (1 - r)) = max 0 (r - η₀) := by congr 1; ring
  rw [eR] at hsR
  rw [eRx] at hsR
  -- clipping only lowers each side's term
  have hclipA : m * max 0 (min 1 ((1 - η₀ - a) / (1 - 2 * η₀)))
      ≤ m * max 0 (1 - η₀ - a) / (1 - 2 * η₀) := by
    rw [mul_div_assoc]
    refine mul_le_mul_of_nonneg_left ?_ hm.1
    rw [le_div_iff₀ hg]
    rcases le_total 0 ((1 - η₀ - a) / (1 - 2 * η₀)) with hc | hc
    · rw [max_eq_right (le_min (by norm_num) hc)]
      calc min 1 ((1 - η₀ - a) / (1 - 2 * η₀)) * (1 - 2 * η₀)
          ≤ (1 - η₀ - a) / (1 - 2 * η₀) * (1 - 2 * η₀) :=
            mul_le_mul_of_nonneg_right (min_le_right _ _) hg.le
        _ = 1 - η₀ - a := div_mul_cancel₀ _ hg.ne'
        _ ≤ max 0 (1 - η₀ - a) := le_max_right _ _
    · rw [max_eq_left (le_trans (min_le_right _ _) hc), zero_mul]
      exact le_max_left _ _
  have hclipR : (1 - m) * max 0 (min 1 ((r - η₀) / (1 - 2 * η₀)))
      ≤ (1 - m) * max 0 (r - η₀) / (1 - 2 * η₀) := by
    rw [mul_div_assoc]
    refine mul_le_mul_of_nonneg_left ?_ (by linarith [hm.2])
    rw [le_div_iff₀ hg]
    rcases le_total 0 ((r - η₀) / (1 - 2 * η₀)) with hc | hc
    · rw [max_eq_right (le_min (by norm_num) hc)]
      calc min 1 ((r - η₀) / (1 - 2 * η₀)) * (1 - 2 * η₀)
          ≤ (r - η₀) / (1 - 2 * η₀) * (1 - 2 * η₀) :=
            mul_le_mul_of_nonneg_right (min_le_right _ _) hg.le
        _ = r - η₀ := div_mul_cancel₀ _ hg.ne'
        _ ≤ max 0 (r - η₀) := le_max_right _ _
    · rw [max_eq_left (le_trans (min_le_right _ _) hc), zero_mul]
      exact le_max_left _ _
  unfold misShare
  have htot : m * max 0 (1 - η₀ - a) + (1 - m) * max 0 (r - η₀)
      ≤ (1 - 2 * η₀) * limit := by
    have hsplit : (max 0 ((1 - η₀) * nA - hA) + max 0 (hR - η₀ * nR)) / N
        = max 0 ((1 - η₀) * nA - hA) / N + max 0 (hR - η₀ * nR) / N := add_div _ _ _
    rw [hsplit] at hpt
    nlinarith
  calc m * max 0 (min 1 ((1 - η₀ - a) / (1 - 2 * η₀)))
        + (1 - m) * max 0 (min 1 ((r - η₀) / (1 - 2 * η₀)))
      ≤ m * max 0 (1 - η₀ - a) / (1 - 2 * η₀)
        + (1 - m) * max 0 (r - η₀) / (1 - 2 * η₀) := add_le_add hclipA hclipR
    _ = (m * max 0 (1 - η₀ - a) + (1 - m) * max 0 (r - η₀)) / (1 - 2 * η₀) := by ring
    _ ≤ limit := by rw [div_le_iff₀ hg]; linarith

end OrthoDFA
