import OrthoDFA.Proofs.Basics
import Mathlib.Probability.Independence.Basic

/-!
# A binomial test is valid on bits that lean its way

`drift_verdict` calls a side drifted when its count's binomial p-value clears a level.  On a
side whose bits are independent and each fire at least as often as the null's rate, that
happens at most the level's share of the time, however few bits the side holds.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

lemma binomCdfLe_zero_n (p : ℝ) (k : ℕ) : binomCdfLe 0 p k = 1 := by
  unfold binomCdfLe
  rw [Finset.sum_eq_single 0 (fun i _ hi => by simp [Nat.choose_eq_zero_of_lt (Nat.pos_of_ne_zero hi)])
    (fun h => absurd (Finset.mem_range.2 (Nat.succ_pos k)) h)]
  simp

lemma binomCdfLe_succ_zero (n : ℕ) (p : ℝ) :
    binomCdfLe (n + 1) p 0 = (1 - p) * binomCdfLe n p 0 := by
  simp [binomCdfLe, pow_succ]
  ring

lemma binomCdfLe_succ_succ (n k : ℕ) (p : ℝ) :
    binomCdfLe (n + 1) p (k + 1)
      = (1 - p) * binomCdfLe n p (k + 1) + p * binomCdfLe n p k := by
  unfold binomCdfLe
  rw [Finset.sum_range_succ' _ (k + 1), Finset.sum_range_succ' _ (k + 1)]
  simp only [Nat.choose_succ_succ, Nat.cast_add, add_mul, Finset.sum_add_distrib,
    Nat.choose_zero_right, Nat.cast_one, pow_zero, one_mul, Nat.sub_zero, mul_one]
  have h1 : ∀ i ∈ Finset.range (k + 1),
      (n.choose (i + 1) : ℝ) * p ^ (i + 1) * (1 - p) ^ (n + 1 - (i + 1))
        = (1 - p) * ((n.choose (i + 1) : ℝ) * p ^ (i + 1) * (1 - p) ^ (n - (i + 1))) := by
    intro i _
    rcases Nat.lt_or_ge n (i + 1) with h | h
    · simp [Nat.choose_eq_zero_of_lt h]
    · rw [show n + 1 - (i + 1) = n - (i + 1) + 1 by omega, pow_succ]
      ring
  have h2 : ∀ i ∈ Finset.range (k + 1),
      (n.choose i : ℝ) * p ^ (i + 1) * (1 - p) ^ (n + 1 - (i + 1))
        = p * ((n.choose i : ℝ) * p ^ i * (1 - p) ^ (n - i)) := by
    intro i _
    rw [show n + 1 - (i + 1) = n - i by omega, pow_succ]
    ring
  rw [Finset.sum_congr rfl h1, Finset.sum_congr rfl h2, ← Finset.mul_sum, ← Finset.mul_sum,
    pow_succ]
  ring

lemma binomCdfLe_nonneg (n k : ℕ) {p : ℝ} (hp0 : 0 ≤ p) (hp1 : p ≤ 1) :
    0 ≤ binomCdfLe n p k :=
  Finset.sum_nonneg (fun i _ => by
    have : 0 ≤ 1 - p := by linarith
    positivity)

lemma binomCdfLe_mono (n : ℕ) {p : ℝ} (hp0 : 0 ≤ p) (hp1 : p ≤ 1) {k k' : ℕ} (hk : k ≤ k') :
    binomCdfLe n p k ≤ binomCdfLe n p k' :=
  Finset.sum_le_sum_of_subset_of_nonneg
    (fun x hx => Finset.mem_range.2 (by have := Finset.mem_range.1 hx; omega))
    (fun i _ _ => by
      have : 0 ≤ 1 - p := by linarith
      positivity)

/-- The binomial lower tail is the upper tail of the complementary count. -/
lemma binomCdfLe_eq (n j : ℕ) (r : ℝ) (hj : j ≤ n) :
    binomCdfLe n r j = binomSfGe n (1 - r) (n - j) := by
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
    simp only [Finset.mem_range] at hi
    rw [Nat.choose_symm (by omega), show n - (n - i) = i by omega, sub_sub_cancel]
    ring

/-- The two tails at a cut partition `Bin(n, p)`. -/
lemma binomCdfLe_add_binomSfGe (n k : ℕ) (p : ℝ) (hk : k ≤ n) :
    binomCdfLe n p k + binomSfGe n p (k + 1) = 1 := by
  have h1 : (∑ i ∈ Finset.range (n + 1), (n.choose i : ℝ) * p ^ i * (1 - p) ^ (n - i)) = 1 := by
    have h := add_pow p (1 - p) n
    rw [add_sub_cancel, one_pow] at h
    calc (∑ i ∈ Finset.range (n + 1), (n.choose i : ℝ) * p ^ i * (1 - p) ^ (n - i))
        = ∑ m ∈ Finset.range (n + 1), p ^ m * (1 - p) ^ (n - m) * (n.choose m : ℝ) :=
          Finset.sum_congr rfl (fun i _ => by ring)
      _ = 1 := h.symm
  have hsplit : binomCdfLe n p k + binomSfGe n p (k + 1)
      = ∑ i ∈ Finset.range (n + 1), (n.choose i : ℝ) * p ^ i * (1 - p) ^ (n - i) := by
    unfold binomCdfLe binomSfGe
    rw [Finset.range_eq_Ico, Finset.range_eq_Ico,
      ← Finset.sum_Ico_consecutive _ (Nat.zero_le (k + 1)) (by omega : k + 1 ≤ n + 1)]
    congr 1
  rw [hsplit, h1]

lemma binomSfGe_anti (n : ℕ) {p : ℝ} (hp0 : 0 ≤ p) (hp1 : p ≤ 1) {j j' : ℕ} (hj : j ≤ j') :
    binomSfGe n p j' ≤ binomSfGe n p j :=
  Finset.sum_le_sum_of_subset_of_nonneg
    (fun x hx => by
      simp only [Finset.mem_Icc] at hx ⊢
      omega)
    (fun i _ _ => by
      have : 0 ≤ 1 - p := by linarith
      positivity)

lemma binomSfGe_zero (n : ℕ) (p : ℝ) : binomSfGe n p 0 = 1 := by
  have h1 : (∑ i ∈ Finset.range (n + 1), (n.choose i : ℝ) * p ^ i * (1 - p) ^ (n - i)) = 1 := by
    have h := add_pow p (1 - p) n
    rw [add_sub_cancel, one_pow] at h
    calc (∑ i ∈ Finset.range (n + 1), (n.choose i : ℝ) * p ^ i * (1 - p) ^ (n - i))
        = ∑ m ∈ Finset.range (n + 1), p ^ m * (1 - p) ^ (n - m) * (n.choose m : ℝ) :=
          Finset.sum_congr rfl (fun i _ => by ring)
      _ = 1 := h.symm
  unfold binomSfGe
  rw [show Finset.Icc 0 n = Finset.range (n + 1) by ext i; simp [Finset.mem_Icc]]
  exact h1

/-- An accept side whose count clears the upper test is far from clearing the lower one. -/
lemma not_lower_of_upper (n k : ℕ) {p α L : ℝ} (hp0 : 0 ≤ p) (hp1 : p ≤ 1) (hk : k ≤ n)
    (hα : α < 1 / 2) (hL : L ≤ α) (h : binomSfGe n p k ≤ α) : ¬ binomCdfLe n p k ≤ L := by
  have e := binomCdfLe_add_binomSfGe n k p hk
  have m := binomSfGe_anti n hp0 hp1 (Nat.le_succ k)
  intro hc
  linarith

/-- A reject side whose count clears the lower test is far from clearing the upper one. -/
lemma not_upper_of_lower (n k : ℕ) {p α L : ℝ} (hp0 : 0 ≤ p) (hp1 : p ≤ 1) (hk : k ≤ n)
    (hα : α < 1 / 2) (hL : L ≤ α) (h : binomCdfLe n p k ≤ α) : ¬ binomSfGe n p k ≤ L := by
  intro hs
  rcases k with _ | k
  · rw [binomSfGe_zero] at hs
    have := binomCdfLe_nonneg n 0 hp0 hp1
    linarith
  · have e := binomCdfLe_add_binomSfGe n k p (by omega)
    have m := binomCdfLe_mono n hp0 hp1 (Nat.le_succ k)
    linarith

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
variable {S : Type*} [Stringlike S]

open scoped Classical in
/-- A count of independent oracle bits, each set with probability at least `p₀`, lies at or
below `k` no more often than `Bin(#A, p₀)` does. -/
theorem count_le_binomCdfLe (O : Oracle μ S) (T : S → Set ℝ) (hT : ∀ p, MeasurableSet (T p))
    {p₀ : ℝ} (hp0 : 0 ≤ p₀) (hp1 : p₀ ≤ 1) (A : Finset S)
    (hA : ∀ p ∈ A, p₀ ≤ μ.real {ω | O.noise p ω ∈ T p}) (k : ℕ) :
    μ.real {ω | (A.filter (fun p => O.noise p ω ∈ T p)).card ≤ k} ≤ binomCdfLe A.card p₀ k := by
  classical
  induction A using Finset.induction_on generalizing k with
  | empty => simp [binomCdfLe_zero_n]
  | insert a A haA ih =>
    have hA' : ∀ p ∈ A, p₀ ≤ μ.real {ω | O.noise p ω ∈ T p} :=
      fun p hp => hA p (Finset.mem_insert_of_mem hp)
    set E : Set Ω := {ω | O.noise a ω ∈ T a} with hE
    set G : ℕ → Set Ω := fun j => {ω | (A.filter (fun p => O.noise p ω ∈ T p)).card ≤ j} with hG
    have hEm : MeasurableSet E := O.noise_meas a (hT a)
    -- the bit at `a` is independent of the count over `A`
    have hind : IndepFun (fun ω (i : ({a} : Finset S)) => O.noise i ω)
        (fun ω (i : A) => O.noise i ω) μ :=
      O.noise_indep.indepFun_finset {a} A (Finset.disjoint_singleton_left.2 haA) O.noise_meas
    have hcnt : Measurable (fun v : A → ℝ =>
        ((Finset.univ : Finset A).filter (fun (i : A) => v i ∈ T (i : S))).card) := by
      simp_rw [Finset.card_filter]
      exact Finset.measurable_sum _ (fun i _ =>
        Measurable.ite ((measurable_pi_apply i) (hT _)) measurable_const measurable_const)
    have hGpre : ∀ j, G j = (fun ω (i : A) => O.noise i ω) ⁻¹'
        {v | ((Finset.univ : Finset A).filter (fun (i : A) => v i ∈ T (i : S))).card ≤ j} := by
      intro j
      ext ω
      simp only [hG, Set.mem_setOf_eq, Set.mem_preimage]
      have hc : ((Finset.univ : Finset A).filter (fun (i : A) => O.noise i ω ∈ T (i : S))).card
          = (A.filter (fun p => O.noise p ω ∈ T p)).card :=
        Finset.card_bij (fun i _ => (i : S))
          (fun i hi => Finset.mem_filter.2 ⟨i.2, (Finset.mem_filter.1 hi).2⟩)
          (fun i₁ _ i₂ _ h => Subtype.ext h)
          (fun b hb => ⟨⟨b, (Finset.mem_filter.1 hb).1⟩,
            Finset.mem_filter.2 ⟨Finset.mem_univ _, (Finset.mem_filter.1 hb).2⟩, rfl⟩)
      rw [hc]
    have hEpre : E = (fun ω (i : ({a} : Finset S)) => O.noise i ω) ⁻¹'
        {v | v ⟨a, Finset.mem_singleton_self a⟩ ∈ T a} := rfl
    have hsetm : ∀ j, MeasurableSet
        {v : A → ℝ | ((Finset.univ : Finset A).filter (fun (i : A) => v i ∈ T (i : S))).card ≤ j} := fun j =>
      hcnt (Set.to_countable {n : ℕ | n ≤ j}).measurableSet
    have hEsetm : MeasurableSet {v : ({a} : Finset S) → ℝ | v ⟨a, Finset.mem_singleton_self a⟩ ∈ T a} :=
      (measurable_pi_apply _) (hT a)
    have hmul : ∀ j, μ.real (E ∩ G j) = μ.real E * μ.real (G j) := by
      intro j
      rw [hEpre, hGpre j]
      simp only [measureReal_def]
      rw [hind.measure_inter_preimage_eq_mul _ _ hEsetm (hsetm j), ENNReal.toReal_mul]
    have hmulc : ∀ j, μ.real (Eᶜ ∩ G j) = (1 - μ.real E) * μ.real (G j) := by
      intro j
      have hGm : MeasurableSet (G j) := by rw [hGpre j]; exact (hsetm j).preimage (measurable_pi_lambda _ (fun i => O.noise_meas _))
      have hsplit : μ.real (G j) = μ.real (E ∩ G j) + μ.real (Eᶜ ∩ G j) := by
        rw [← measureReal_union (Set.disjoint_left.2 (fun ω h1 h2 => h2.1 h1.1))
          (hEm.compl.inter hGm)]
        congr 1
        ext ω; by_cases h : ω ∈ E <;> simp [h]
      rw [hmul j] at hsplit
      linarith
    have hcard : ∀ ω, ((insert a A).filter (fun p => O.noise p ω ∈ T p)).card
        = (A.filter (fun p => O.noise p ω ∈ T p)).card + if O.noise a ω ∈ T a then 1 else 0 := by
      intro ω
      rw [Finset.filter_insert]
      split_ifs with h
      · rw [Finset.card_insert_of_notMem (fun hm => haA (Finset.mem_filter.1 hm).1)]
      · rfl
    have hq : p₀ ≤ μ.real E := hA a (Finset.mem_insert_self a A)
    have hq1 : μ.real E ≤ 1 := measureReal_le_one
    rw [Finset.card_insert_of_notMem haA]
    rcases k with _ | k
    · have hset : {ω | ((insert a A).filter (fun p => O.noise p ω ∈ T p)).card ≤ 0}
          ⊆ Eᶜ ∩ G 0 := by
        intro ω hω
        simp only [Set.mem_setOf_eq, hcard] at hω
        refine ⟨fun hE' => ?_, ?_⟩
        · simp only [hE, Set.mem_setOf_eq] at hE'
          simp [hE'] at hω
        · simp only [hG, Set.mem_setOf_eq]; omega
      refine le_trans (measureReal_mono hset (measure_ne_top _ _)) ?_
      rw [hmulc 0, binomCdfLe_succ_zero]
      have h0 := ih hA' 0
      have hF := binomCdfLe_nonneg A.card 0 hp0 hp1
      nlinarith [measureReal_nonneg (μ := μ) (s := G 0)]
    · have hset : {ω | ((insert a A).filter (fun p => O.noise p ω ∈ T p)).card ≤ k + 1}
          ⊆ (E ∩ G k) ∪ (Eᶜ ∩ G (k + 1)) := by
        intro ω hω
        simp only [Set.mem_setOf_eq, hcard] at hω
        by_cases hE' : ω ∈ E
        · left
          refine ⟨hE', ?_⟩
          simp only [hE, Set.mem_setOf_eq] at hE'
          simp only [hG, Set.mem_setOf_eq]
          simp [hE'] at hω; omega
        · right
          refine ⟨hE', ?_⟩
          simp only [hE, Set.mem_setOf_eq] at hE'
          simp only [hG, Set.mem_setOf_eq]
          simp [hE'] at hω; omega
      refine le_trans (measureReal_mono hset (measure_ne_top _ _))
        (le_trans (measureReal_union_le _ _) ?_)
      rw [hmul k, hmulc (k + 1), binomCdfLe_succ_succ]
      have hk := ih hA' k
      have hk1 := ih hA' (k + 1)
      have hmono := binomCdfLe_mono A.card hp0 hp1 (Nat.le_succ k)
      have hGk := measureReal_nonneg (μ := μ) (s := G k)
      have hGk1 := measureReal_nonneg (μ := μ) (s := G (k + 1))
      have e1 : μ.real E * μ.real (G k) ≤ μ.real E * binomCdfLe A.card p₀ k :=
        mul_le_mul_of_nonneg_left hk measureReal_nonneg
      have e2 : (1 - μ.real E) * μ.real (G (k + 1))
          ≤ (1 - μ.real E) * binomCdfLe A.card p₀ (k + 1) :=
        mul_le_mul_of_nonneg_left hk1 (by linarith)
      have e3 : (μ.real E - p₀) * (binomCdfLe A.card p₀ k - binomCdfLe A.card p₀ (k + 1)) ≤ 0 :=
        mul_nonpos_of_nonneg_of_nonpos (by linarith) (by linarith)
      nlinarith

omit [IsProbabilityMeasure μ] in
/-- The reads that come back `1` are the noise bits in a set fixed by the label. -/
lemma mq_eq_one_iff (O : Oracle μ S) (p : S) (ω : Ω) :
    O.mq p ω = 1 ↔ O.noise p ω ∈ (fun r => O.label p + (1 - 2 * O.label p) * r) ⁻¹' {1} :=
  Iff.rfl

omit [IsProbabilityMeasure μ] in
lemma measurableSet_mqSet (O : Oracle μ S) (p : S) :
    MeasurableSet ((fun r : ℝ => O.label p + (1 - 2 * O.label p) * r) ⁻¹' {1}) :=
  (measurable_const.add (measurable_const.mul measurable_id)) (measurableSet_singleton 1)

open scoped Classical in
/-- `drifted`'s accept-side test is valid: on prefixes each read `1` at least as often as the
null's rate, its lower-tail p-value clears `L` at most `L` of the time. -/
theorem lowerTest_valid (O : Oracle μ S) (A : Finset S) {p₀ L : ℝ} (hL : 0 ≤ L) (hp0 : 0 ≤ p₀)
    (hp1 : p₀ ≤ 1) (hA : ∀ p ∈ A, p₀ ≤ μ.real {ω | O.mq p ω = 1}) :
    μ.real {ω | binomCdfLe A.card p₀ (A.filter (fun p => O.mq p ω = 1)).card ≤ L} ≤ L := by
  classical
  set K := (Finset.range (A.card + 1)).filter (fun j => binomCdfLe A.card p₀ j ≤ L) with hK
  have hin : ∀ ω, binomCdfLe A.card p₀ (A.filter (fun p => O.mq p ω = 1)).card ≤ L →
      (A.filter (fun p => O.mq p ω = 1)).card ∈ K := fun ω h =>
    Finset.mem_filter.2 ⟨Finset.mem_range.2 (Nat.lt_succ_of_le (Finset.card_filter_le _ _)), h⟩
  rcases K.eq_empty_or_nonempty with hKe | hKne
  · have hz : {ω | binomCdfLe A.card p₀ (A.filter (fun p => O.mq p ω = 1)).card ≤ L}
        = (∅ : Set Ω) := by
      ext ω
      simp only [Set.mem_setOf_eq, Set.mem_empty_iff_false, iff_false]
      intro h
      have := hin ω h
      rw [hKe] at this
      simp at this
    rw [hz, measureReal_empty]
    exact hL
  · set kstar := K.max' hKne with hkstar
    have hsub : {ω | binomCdfLe A.card p₀ (A.filter (fun p => O.mq p ω = 1)).card ≤ L}
        ⊆ {ω | (A.filter (fun p => O.noise p ω ∈
            (fun r => O.label p + (1 - 2 * O.label p) * r) ⁻¹' {1})).card ≤ kstar} := by
      intro ω h
      exact K.le_max' _ (hin ω h)
    refine le_trans (measureReal_mono hsub (measure_ne_top _ _)) ?_
    refine le_trans (count_le_binomCdfLe O _ (measurableSet_mqSet O) hp0 hp1 A hA kstar) ?_
    exact (Finset.mem_filter.1 (K.max'_mem hKne)).2

open scoped Classical in
/-- `drifted`'s reject-side test is valid: on prefixes each read `1` at most as often as the
null's rate, its upper-tail p-value clears `L` at most `L` of the time. -/
theorem upperTest_valid (O : Oracle μ S) (R : Finset S) {p₀ L : ℝ} (hL : 0 ≤ L) (hp0 : 0 ≤ p₀)
    (hp1 : p₀ ≤ 1) (hR : ∀ p ∈ R, μ.real {ω | O.mq p ω = 1} ≤ p₀) :
    μ.real {ω | binomSfGe R.card p₀ (R.filter (fun p => O.mq p ω = 1)).card ≤ L} ≤ L := by
  classical
  set T' : S → Set ℝ := fun p => ((fun r => O.label p + (1 - 2 * O.label p) * r) ⁻¹' {1})ᶜ
    with hT'
  have hT'm : ∀ p, MeasurableSet (T' p) := fun p => (measurableSet_mqSet O p).compl
  have hmiss : ∀ p ∈ R, 1 - p₀ ≤ μ.real {ω | O.noise p ω ∈ T' p} := by
    intro p hp
    have hc : {ω | O.noise p ω ∈ T' p} = {ω | O.mq p ω = 1}ᶜ := rfl
    have hmeas : MeasurableSet {ω | O.mq p ω = 1} := (O.noise_meas p) (measurableSet_mqSet O p)
    rw [hc, measureReal_compl hmeas, probReal_univ]
    linarith [hR p hp]
  have hcnt : ∀ ω, (R.filter (fun p => O.noise p ω ∈ T' p)).card
      = R.card - (R.filter (fun p => O.mq p ω = 1)).card := by
    intro ω
    have h := Finset.card_filter_add_card_filter_not (s := R) (fun p => O.mq p ω = 1)
    have : R.filter (fun p => O.noise p ω ∈ T' p) = R.filter (fun p => ¬ O.mq p ω = 1) := rfl
    rw [this]
    omega
  have hsub : {ω | binomSfGe R.card p₀ (R.filter (fun p => O.mq p ω = 1)).card ≤ L}
      ⊆ {ω | binomCdfLe R.card (1 - p₀) (R.filter (fun p => O.noise p ω ∈ T' p)).card ≤ L} := by
    intro ω h
    simp only [Set.mem_setOf_eq] at h ⊢
    have hle : (R.filter (fun p => O.mq p ω = 1)).card ≤ R.card := Finset.card_filter_le _ _
    rw [hcnt ω, binomCdfLe_eq _ _ _ (Nat.sub_le _ _), sub_sub_cancel,
      Nat.sub_sub_self hle]
    exact h
  refine le_trans (measureReal_mono hsub (measure_ne_top _ _)) ?_
  set K := (Finset.range (R.card + 1)).filter (fun j => binomCdfLe R.card (1 - p₀) j ≤ L)
    with hK
  have hin : ∀ ω, binomCdfLe R.card (1 - p₀) (R.filter (fun p => O.noise p ω ∈ T' p)).card ≤ L →
      (R.filter (fun p => O.noise p ω ∈ T' p)).card ∈ K := fun ω h =>
    Finset.mem_filter.2 ⟨Finset.mem_range.2 (Nat.lt_succ_of_le (Finset.card_filter_le _ _)), h⟩
  rcases K.eq_empty_or_nonempty with hKe | hKne
  · have hz : {ω | binomCdfLe R.card (1 - p₀) (R.filter (fun p => O.noise p ω ∈ T' p)).card ≤ L}
        = (∅ : Set Ω) := by
      ext ω
      simp only [Set.mem_setOf_eq, Set.mem_empty_iff_false, iff_false]
      intro h
      have := hin ω h
      rw [hKe] at this
      simp at this
    rw [hz, measureReal_empty]
    exact hL
  · set kstar := K.max' hKne with hkstar
    have hsub2 : {ω | binomCdfLe R.card (1 - p₀) (R.filter (fun p => O.noise p ω ∈ T' p)).card ≤ L}
        ⊆ {ω | (R.filter (fun p => O.noise p ω ∈ T' p)).card ≤ kstar} :=
      fun ω h => K.le_max' _ (hin ω h)
    refine le_trans (measureReal_mono hsub2 (measure_ne_top _ _)) ?_
    have hc := count_le_binomCdfLe O T' hT'm (by linarith) (by linarith) R hmiss kstar
    refine le_trans (le_of_eq ?_) (hc.trans (Finset.mem_filter.1 (K.max'_mem hKne)).2)
    congr!

end OrthoDFA
