import OrthoDFA.Clustering

/-!
# Drawing without replacement, against the draws themselves

`freshDraws T y m` is the first `m` distinct values of `y` outside `T`.  The concentration
arguments read the stream's first draws, repeats and all, so everything here bounds the gap
between the two by the draws the sampler rejected: among the first `n`, that is `n` less the
distinct values outside `T` they hold.
-/

namespace OrthoDFA

variable {S : Type*} [Stringlike S]

/-- The distinct values outside `T` among the first `n` draws. -/
def acceptedAt (T : Finset S) (y : ℕ → S) (n : ℕ) : Finset S :=
  (Finset.range n).image y \ T

lemma card_acceptedAt_le (T : Finset S) (y : ℕ → S) (n : ℕ) :
    (acceptedAt T y n).card ≤ n :=
  le_trans (Finset.card_le_card Finset.sdiff_subset)
    (le_trans Finset.card_image_le (by simp))

lemma acceptedAt_mono (T : Finset S) (y : ℕ → S) {n n' : ℕ} (h : n ≤ n') :
    acceptedAt T y n ⊆ acceptedAt T y n' :=
  Finset.sdiff_subset_sdiff (Finset.image_subset_image (Finset.range_subset_range.2 h))
    le_rfl

lemma acceptedAt_succ (T : Finset S) (y : ℕ → S) (n : ℕ) :
    acceptedAt T y (n + 1) = insert (y n) ((Finset.range n).image y) \ T := by
  rw [acceptedAt, Finset.range_add_one, Finset.image_insert]

lemma card_acceptedAt_succ_le (T : Finset S) (y : ℕ → S) (n : ℕ) :
    (acceptedAt T y (n + 1)).card ≤ (acceptedAt T y n).card + 1 := by
  rw [acceptedAt_succ]
  by_cases hT : y n ∈ T
  · rw [Finset.insert_sdiff_of_mem _ hT]; exact Nat.le_succ _
  · rw [Finset.insert_sdiff_of_notMem _ hT]; exact Finset.card_insert_le _ _

/-- Rejections only accumulate: `n − #accepted` is monotone in `n`. -/
lemma card_acceptedAt_add_le (T : Finset S) (y : ℕ → S) {n n' : ℕ} (h : n ≤ n') :
    (acceptedAt T y n').card + n ≤ (acceptedAt T y n).card + n' := by
  induction n', h using Nat.le_induction with
  | base => exact le_rfl
  | succ k _ ih => have := card_acceptedAt_succ_le T y k; omega

lemma freshDraws_eq (T : Finset S) (y : ℕ → S) (m : ℕ)
    (h : ∃ n, m ≤ (acceptedAt T y n).card) :
    freshDraws T y m = acceptedAt T y (Nat.find h) := by
  classical
  have h' : ∃ n, m ≤ ((Finset.range n).image y \ T).card := h
  unfold freshDraws
  rw [dif_pos h']
  rfl

lemma card_freshDraws (T : Finset S) (y : ℕ → S) (m : ℕ)
    (h : ∃ n, m ≤ (acceptedAt T y n).card) : (freshDraws T y m).card = m := by
  classical
  rw [freshDraws_eq T y m h]
  have hspec := Nat.find_spec h
  refine le_antisymm ?_ hspec
  rcases Nat.eq_zero_or_pos (Nat.find h) with h0 | hpos
  · rw [h0]
    exact le_trans (card_acceptedAt_le T y 0) (Nat.zero_le _)
  · have hmin := Nat.find_min h (show Nat.find h - 1 < Nat.find h by omega)
    have hsucc := card_acceptedAt_succ_le T y (Nat.find h - 1)
    rw [show Nat.find h - 1 + 1 = Nat.find h by omega] at hsucc
    omega

lemma freshDraws_of_not (T : Finset S) (y : ℕ → S) (m : ℕ)
    (h : ¬ ∃ n, m ≤ (acceptedAt T y n).card) : freshDraws T y m = ∅ := by
  classical
  have h' : ¬ ∃ n, m ≤ ((Finset.range n).image y \ T).card := h
  unfold freshDraws
  rw [dif_neg h']

lemma card_freshDraws_le (T : Finset S) (y : ℕ → S) (m : ℕ) : (freshDraws T y m).card ≤ m := by
  classical
  by_cases h : ∃ n, m ≤ (acceptedAt T y n).card
  · exact (card_freshDraws T y m h).le
  · have h' : ¬ ∃ n, m ≤ ((Finset.range n).image y \ T).card := h
    unfold freshDraws
    rw [dif_neg h']
    simp

lemma disjoint_freshDraws (T : Finset S) (y : ℕ → S) (m : ℕ) :
    Disjoint (freshDraws T y m) T := by
  classical
  unfold freshDraws
  split_ifs
  · exact Finset.sdiff_disjoint
  · exact Finset.disjoint_empty_left _

lemma mem_freshDraws {T : Finset S} {y : ℕ → S} {m : ℕ} {v : S} (hv : v ∈ freshDraws T y m) :
    ∃ i, y i = v := by
  classical
  unfold freshDraws at hv
  split_ifs at hv
  · obtain ⟨i, -, hi⟩ := Finset.mem_image.1 (Finset.mem_sdiff.1 hv).1
    exact ⟨i, hi⟩
  · simp at hv

/-- Once the first `n` draws offer `m` fresh values, the sample is among them. -/
lemma freshDraws_subset_acceptedAt (T : Finset S) (y : ℕ → S) {m n : ℕ}
    (hn : m ≤ (acceptedAt T y n).card) : freshDraws T y m ⊆ acceptedAt T y n := by
  classical
  have h : ∃ n, m ≤ (acceptedAt T y n).card := ⟨n, hn⟩
  rw [freshDraws_eq T y m h]
  exact acceptedAt_mono T y (Nat.find_min' h hn)

/-- The sample holds every fresh value among the first `m` draws. -/
lemma acceptedAt_subset_freshDraws (T : Finset S) (y : ℕ → S) (m : ℕ)
    (h : ∃ n, m ≤ (acceptedAt T y n).card) : acceptedAt T y m ⊆ freshDraws T y m := by
  classical
  rw [freshDraws_eq T y m h]
  refine acceptedAt_mono T y ?_
  exact le_trans (Nat.find_spec h) (card_acceptedAt_le T y _)

/-- A bounded score summed over the first `n` draws exceeds its sum over their fresh values by
at most the draws rejected. -/
lemma sum_range_le_sum_acceptedAt (T : Finset S) (y : ℕ → S) (f : S → ℝ)
    (hf0 : ∀ a, 0 ≤ f a) (hf1 : ∀ a, f a ≤ 1) (n : ℕ) :
    ∑ i ∈ Finset.range n, f (y i)
      ≤ ∑ p ∈ acceptedAt T y n, f p + ((n : ℝ) - (acceptedAt T y n).card) := by
  induction n with
  | zero => simp [acceptedAt]
  | succ n ih =>
    rw [Finset.sum_range_succ, acceptedAt_succ]
    have hle1 := hf1 (y n)
    have hge0 := hf0 (y n)
    by_cases hT : y n ∈ T
    · rw [Finset.insert_sdiff_of_mem _ hT]
      push_cast
      have : (Finset.range n).image y \ T = acceptedAt T y n := rfl
      rw [this]
      linarith
    · by_cases hmem : y n ∈ (Finset.range n).image y
      · rw [Finset.insert_eq_of_mem hmem]
        have : (Finset.range n).image y \ T = acceptedAt T y n := rfl
        rw [this]
        push_cast
        linarith
      · rw [Finset.insert_sdiff_of_notMem _ hT]
        have hnot : y n ∉ acceptedAt T y n := fun h => hmem (Finset.mem_sdiff.1 h).1
        have : (Finset.range n).image y \ T = acceptedAt T y n := rfl
        rw [this, Finset.sum_insert hnot, Finset.card_insert_of_notMem hnot]
        push_cast
        linarith

/-- Collapsing an image can only drop terms, never add them. -/
lemma sum_image_le_sum {β : Type*} [DecidableEq β] (g : β → ℝ) (hg : ∀ b, 0 ≤ g b)
    (f : ℕ → β) (s : Finset ℕ) :
    ∑ b ∈ s.image f, g b ≤ ∑ i ∈ s, g (f i) := by
  classical
  refine Finset.induction_on s (by simp) ?_
  intro a t ha ih
  rw [Finset.image_insert, Finset.sum_insert ha]
  by_cases h : f a ∈ t.image f
  · rw [Finset.insert_eq_of_mem h]
    linarith [ih, hg (f a)]
  · rw [Finset.sum_insert h]
    linarith [ih]

/-- The sample against the first `m` draws, once the first `m + r` offer `m` fresh values: it is
`m` strings, all among those draws, and a bounded score summed over it is within `r` of its sum
over the first `m` draws, either way. -/
lemma freshDraws_of_good (T : Finset S) (y : ℕ → S) (m r : ℕ)
    (hgood : m ≤ (acceptedAt T y (m + r)).card) :
    (freshDraws T y m).card = m
      ∧ freshDraws T y m ⊆ acceptedAt T y (m + r)
      ∧ (∀ f : S → ℝ, (∀ a, 0 ≤ f a) → (∀ a, f a ≤ 1) →
          ∑ i ∈ Finset.range m, f (y i) ≤ ∑ p ∈ freshDraws T y m, f p + r
          ∧ ∑ p ∈ freshDraws T y m, f p ≤ ∑ i ∈ Finset.range m, f (y i) + r) := by
  classical
  have h : ∃ n, m ≤ (acceptedAt T y n).card := ⟨m + r, hgood⟩
  have hsub := freshDraws_subset_acceptedAt T y hgood
  refine ⟨card_freshDraws T y m h, hsub, fun f hf0 hf1 => ⟨?_, ?_⟩⟩
  · have h1 := sum_range_le_sum_acceptedAt T y f hf0 hf1 m
    have h2 : ∑ p ∈ acceptedAt T y m, f p ≤ ∑ p ∈ freshDraws T y m, f p :=
      Finset.sum_le_sum_of_subset_of_nonneg (acceptedAt_subset_freshDraws T y m h)
        (fun a _ _ => hf0 a)
    have h3 := card_acceptedAt_add_le T y (show m ≤ m + r by omega)
    have h4 : (m : ℝ) - (acceptedAt T y m).card ≤ r := by
      have : m + m ≤ (acceptedAt T y m).card + (m + r) := le_trans (by omega) h3
      have : (m : ℝ) ≤ (acceptedAt T y m).card + r := by exact_mod_cast (by omega : m ≤ _)
      linarith
    linarith
  · have h1 : ∑ p ∈ freshDraws T y m, f p ≤ ∑ p ∈ (Finset.range (m + r)).image y, f p :=
      Finset.sum_le_sum_of_subset_of_nonneg (hsub.trans Finset.sdiff_subset)
        (fun a _ _ => hf0 a)
    have h2 := sum_image_le_sum f hf0 y (Finset.range (m + r))
    have h3 : ∑ i ∈ Finset.range (m + r), f (y i)
        ≤ ∑ i ∈ Finset.range m, f (y i) + r := by
      rw [Finset.sum_range_add]
      have : ∑ i ∈ Finset.range r, f (y (m + i)) ≤ ∑ _i ∈ Finset.range r, (1 : ℝ) :=
        Finset.sum_le_sum (fun i _ => hf1 _)
      simp only [Finset.sum_const, Finset.card_range, nsmul_eq_mul, mul_one] at this
      linarith
    linarith

/-- `freshDraws_of_good` for a count. -/
lemma card_filter_of_good (T : Finset S) (y : ℕ → S) (m r : ℕ)
    (hgood : m ≤ (acceptedAt T y (m + r)).card) (Q : S → Prop) [DecidablePred Q] :
    (((Finset.range m).filter (fun i => Q (y i))).card : ℝ)
        ≤ ((freshDraws T y m).filter Q).card + r
      ∧ (((freshDraws T y m).filter Q).card : ℝ)
        ≤ ((Finset.range m).filter (fun i => Q (y i))).card + r := by
  obtain ⟨-, -, hsum⟩ := freshDraws_of_good T y m r hgood
  have h := hsum (fun p => if Q p then 1 else 0) (fun a => by split_ifs <;> norm_num)
    (fun a => by split_ifs <;> norm_num)
  simp only [Finset.card_filter, Nat.cast_sum, Nat.cast_ite, Nat.cast_one, Nat.cast_zero]
  exact h

lemma card_filter_univ_fin (M : ℕ) (P : ℕ → Prop) [DecidablePred P] :
    ((Finset.univ : Finset (Fin M)).filter (fun i => P i.val)).card
      = ((Finset.range M).filter P).card := by
  rw [← Finset.card_map Fin.valEmbedding]
  congr 1
  ext i
  simp only [Finset.mem_map, Finset.mem_filter, Finset.mem_univ, true_and,
    Fin.valEmbedding_apply, Finset.mem_range]
  constructor
  · rintro ⟨a, ha, rfl⟩
    exact ⟨a.isLt, ha⟩
  · rintro ⟨hi, hP⟩
    exact ⟨⟨i, hi⟩, hP, rfl⟩

open scoped Classical in
/-- The pairs of draws among the first `n` that repeat a value. -/
noncomputable def repeatPairs (y : ℕ → S) (n : ℕ) : Finset (ℕ × ℕ) :=
  ((Finset.range n) ×ˢ (Finset.range n)).filter (fun q => q.1 < q.2 ∧ y q.1 = y q.2)

lemma repeatPairs_mono (y : ℕ → S) (n : ℕ) : repeatPairs y n ⊆ repeatPairs y (n + 1) := by
  classical
  intro q hq
  simp only [repeatPairs, Finset.mem_filter, Finset.mem_product, Finset.mem_range] at hq ⊢
  exact ⟨⟨by omega, by omega⟩, hq.2⟩

/-- Every rejected draw lands in `T` or repeats an earlier one. -/
lemma le_card_acceptedAt_add (T : Finset S) (y : ℕ → S) (n : ℕ) :
    n ≤ (acceptedAt T y n).card + ((Finset.range n).filter (fun i => y i ∈ T)).card
      + (repeatPairs y n).card := by
  classical
  induction n with
  | zero => simp
  | succ n ih =>
    have hpm := Finset.card_le_card (repeatPairs_mono y n)
    rw [acceptedAt_succ, Finset.range_add_one, Finset.filter_insert]
    by_cases hT : y n ∈ T
    · rw [Finset.insert_sdiff_of_mem _ hT, if_pos hT,
        Finset.card_insert_of_notMem (by simp)]
      have : (Finset.range n).image y \ T = acceptedAt T y n := rfl
      rw [this]
      omega
    · rw [if_neg hT]
      by_cases hmem : y n ∈ (Finset.range n).image y
      · rw [Finset.insert_eq_of_mem hmem]
        have : (Finset.range n).image y \ T = acceptedAt T y n := rfl
        rw [this]
        obtain ⟨k, hk, hkn⟩ := Finset.mem_image.1 hmem
        have hnew : (k, n) ∈ repeatPairs y (n + 1) := by
          simp only [repeatPairs, Finset.mem_filter, Finset.mem_product, Finset.mem_range]
          exact ⟨⟨by simp at hk; omega, by omega⟩, by simp at hk; omega, hkn⟩
        have hold : (k, n) ∉ repeatPairs y n := by
          simp [repeatPairs]
        have hlt : (repeatPairs y n).card < (repeatPairs y (n + 1)).card :=
          Finset.card_lt_card ⟨repeatPairs_mono y n, fun h => hold (h hnew)⟩
        omega
      · rw [Finset.insert_sdiff_of_notMem _ hT]
        have : (Finset.range n).image y \ T = acceptedAt T y n := rfl
        rw [this]
        have hnot : y n ∉ acceptedAt T y n := fun h => hmem (Finset.mem_sdiff.1 h).1
        rw [Finset.card_insert_of_notMem hnot]
        omega

lemma card_repeatPairs_le (y : ℕ → S) (n : ℕ) : 2 * (repeatPairs y n).card ≤ n * n := by
  classical
  set A := (Finset.range n ×ˢ Finset.range n).filter (fun q : ℕ × ℕ => q.1 < q.2) with hA
  set B := (Finset.range n ×ˢ Finset.range n).filter (fun q : ℕ × ℕ => q.2 < q.1) with hB
  have hsub : repeatPairs y n ⊆ A := by
    intro q hq
    simp only [repeatPairs, Finset.mem_filter] at hq
    exact Finset.mem_filter.2 ⟨hq.1, hq.2.1⟩
  have hAB : A.card = B.card := by
    refine Finset.card_bij (fun q _ => (q.2, q.1)) (fun q hq => ?_) (fun a _ b _ h => ?_)
      (fun q hq => ⟨(q.2, q.1), ?_, rfl⟩)
    · simp only [hA, hB, Finset.mem_filter, Finset.mem_product] at hq ⊢
      exact ⟨⟨hq.1.2, hq.1.1⟩, hq.2⟩
    · simp only [Prod.mk.injEq] at h
      exact Prod.ext h.2 h.1
    · simp only [hA, hB, Finset.mem_filter, Finset.mem_product] at hq ⊢
      exact ⟨⟨hq.1.2, hq.1.1⟩, hq.2⟩
  have hdisj : Disjoint A B := by
    refine Finset.disjoint_left.2 (fun q hqA hqB => ?_)
    have h1 := (Finset.mem_filter.1 hqA).2
    have h2 := (Finset.mem_filter.1 hqB).2
    omega
  have hunion : (A ∪ B).card ≤ n * n := by
    have : A ∪ B ⊆ Finset.range n ×ˢ Finset.range n :=
      Finset.union_subset (Finset.filter_subset _ _) (Finset.filter_subset _ _)
    simpa using Finset.card_le_card this
  rw [Finset.card_union_of_disjoint hdisj] at hunion
  have := Finset.card_le_card hsub
  omega

end OrthoDFA
