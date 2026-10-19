import OrthoDFA.Proofs.FreshDraws
import OrthoDFA.Proofs.Thinning

/-!
# A sample and the draws before it

A population's table and certification sample are first-come samples of subsequences of its
draws.  Read against a window of its raw draws, either sample holds the window's share of its
subsequence, or the window holds the whole sample.
-/

namespace OrthoDFA

open Thin

variable {S : Type*} [Stringlike S]

/-- A filled sample and any earlier stretch of the same stream are nested. -/
lemma acceptedAt_comparable (T : Finset S) (y : ℕ → S) (m k : ℕ)
    (hF : (freshDraws T y m).card = m) :
    acceptedAt T y k ⊆ freshDraws T y m ∨ freshDraws T y m ⊆ acceptedAt T y k := by
  classical
  by_cases h : ∃ n, m ≤ (acceptedAt T y n).card
  · rw [freshDraws_eq T y m h]
    rcases le_total k (Nat.find h) with hk | hk
    · exact Or.inl (acceptedAt_mono T y hk)
    · exact Or.inr (acceptedAt_mono T y hk)
  · rw [freshDraws_of_not T y m h] at hF ⊢
    rcases Nat.eq_zero_or_pos m with h0 | h0
    · exact Or.inr (Finset.empty_subset _)
    · simp at hF; omega

/-- A stretch with no more than the sample's count lies inside a filled sample. -/
lemma acceptedAt_subset_of_card_le (T : Finset S) (y : ℕ → S) (m k : ℕ)
    (hF : (freshDraws T y m).card = m) (hk : (acceptedAt T y k).card ≤ m) :
    acceptedAt T y k ⊆ freshDraws T y m := by
  rcases acceptedAt_comparable T y m k hF with h | h
  · exact h
  · have := Finset.eq_of_subset_of_card_le h (by rw [hF]; exact hk)
    rw [this]

/-- What a filled sample holds beyond an earlier stretch: its count less the stretch's. -/
lemma card_sdiff_acceptedAt_le (T : Finset S) (y : ℕ → S) (m k : ℕ)
    (hF : (freshDraws T y m).card = m) :
    (freshDraws T y m \ acceptedAt T y k).card + (acceptedAt T y k).card ≥ m
      ∨ freshDraws T y m ⊆ acceptedAt T y k := by
  rcases acceptedAt_comparable T y m k hF with h | h
  · left
    rw [Finset.card_sdiff_of_subset h, hF]
    omega
  · exact Or.inr h

lemma card_sdiff_le_of_comparable (T : Finset S) (y : ℕ → S) (m k : ℕ)
    (hF : (freshDraws T y m).card = m) :
    ((freshDraws T y m \ acceptedAt T y k).card : ℝ) ≤ max 0 ((m : ℝ) - (acceptedAt T y k).card) := by
  rcases acceptedAt_comparable T y m k hF with h | h
  · have hle : (acceptedAt T y k).card ≤ m := hF ▸ Finset.card_le_card h
    rw [Finset.card_sdiff_of_subset h, hF, Nat.cast_sub hle]
    exact le_max_right _ _
  · rw [Finset.sdiff_eq_empty_iff_subset.2 h]
    simp

open scoped Classical in
/-- A subsequence's stretch up to a raw count is the raw stretch's values in the set. -/
lemma acceptedAt_sub (Q : Set S) (d : S) (w : ℕ → S) (T : Finset S) (n : ℕ) :
    acceptedAt T (nthDrawIn Q d w) (drawsIn Q w n).length
      = ((Finset.range n).image w).filter (fun s => s ∈ Q) \ T := by
  classical
  unfold acceptedAt
  congr 1
  ext s
  simp only [Finset.mem_image, Finset.mem_range, Finset.mem_filter]
  constructor
  · rintro ⟨k, hk, rfl⟩
    rw [sub_eq_getD Q d w hk, List.getD_eq_getElem _ _ hk]
    have hmem : (drawsIn Q w n)[k] ∈ drawsIn Q w n := List.getElem_mem hk
    simp only [drawsIn, List.mem_filter, List.mem_map, List.mem_range, decide_eq_true_eq] at hmem
    obtain ⟨⟨i, hi, he⟩, hQ⟩ := hmem
    exact ⟨⟨i, hi, he⟩, hQ⟩
  · rintro ⟨⟨i, hi, rfl⟩, hQ⟩
    have hmem : w i ∈ drawsIn Q w n := by
      simp only [drawsIn, List.mem_filter, List.mem_map, List.mem_range, decide_eq_true_eq]
      exact ⟨⟨i, hi, rfl⟩, hQ⟩
    obtain ⟨k, hk, he⟩ := List.getElem_of_mem hmem
    refine ⟨k, hk, ?_⟩
    rw [sub_eq_getD Q d w hk, List.getD_eq_getElem _ _ hk, he]

lemma measurable_prependStream (L : List S) : Measurable (prependStream L) := by
  refine measurable_pi_lambda _ (fun k => ?_)
  unfold prependStream
  by_cases hk : k < L.length
  · simp only [dif_pos hk]; exact measurable_const
  · simp only [dif_neg hk]; exact measurable_pi_apply _

/-- The first `|L| + k` drawsIn of `L ++ y`: `L`, and `y`'s first `k` outside it. -/
lemma acceptedAt_prepend (L : List S) (y : ℕ → S) (k : ℕ) :
    acceptedAt ∅ (prependStream L y) (L.length + k) = L.toFinset ∪ acceptedAt L.toFinset y k := by
  classical
  unfold acceptedAt
  ext s
  simp only [Finset.sdiff_empty, Finset.mem_image, Finset.mem_range, Finset.mem_union,
    List.mem_toFinset, Finset.mem_sdiff]
  constructor
  · rintro ⟨i, hi, rfl⟩
    by_cases hL : i < L.length
    · left
      simp only [prependStream, dif_pos hL]
      exact List.get_mem _ _
    · by_cases hmem : prependStream L y i ∈ L
      · exact Or.inl hmem
      · right
        refine ⟨⟨i - L.length, by omega, ?_⟩, ?_⟩
        · simp only [prependStream, dif_neg hL]
        · exact hmem
  · rintro (h | ⟨⟨i, hi, rfl⟩, -⟩)
    · obtain ⟨⟨i, hi⟩, rfl⟩ := List.get_of_mem h
      exact ⟨i, by omega, by simp [prependStream, hi]⟩
    · exact ⟨L.length + i, by omega, by simp [prependStream]⟩

/-- A stretch with exactly the sample's count is the sample. -/
lemma freshDraws_eq_of_card (T : Finset S) (y : ℕ → S) (m k : ℕ)
    (hk : (acceptedAt T y k).card = m) : freshDraws T y m = acceptedAt T y k := by
  classical
  have h : ∃ n, m ≤ (acceptedAt T y n).card := ⟨k, hk.ge⟩
  have hF : (freshDraws T y m).card = m := card_freshDraws T y m h
  rcases acceptedAt_comparable T y m k hF with h1 | h1
  · exact (Finset.eq_of_subset_of_card_le h1 (by rw [hF, hk])).symm
  · exact Finset.eq_of_subset_of_card_le h1 (by rw [hF, hk])

open scoped Classical in
/-- The distinct draws among the first `n` that fall in `A`. -/
noncomputable def rawIn (A : Set S) (w : ℕ → S) (n : ℕ) : Finset S :=
  ((Finset.range n).image w).filter (fun s => s ∈ A)

lemma rawIn_mono (A : Set S) (w : ℕ → S) {n n' : ℕ} (h : n ≤ n') :
    rawIn A w n ⊆ rawIn A w n' := by
  classical
  exact Finset.filter_subset_filter _ (Finset.image_subset_image (Finset.range_subset_range.2 h))

lemma card_rawIn_succ_le (A : Set S) (w : ℕ → S) (n : ℕ) :
    (rawIn A w (n + 1)).card ≤ (rawIn A w n).card + 1 := by
  classical
  unfold rawIn
  rw [Finset.range_add_one, Finset.image_insert, Finset.filter_insert]
  split_ifs
  · exact Finset.card_insert_le _ _
  · omega

/-- A count the raw stretches reach, they reach exactly. -/
lemma exists_card_rawIn_eq (A : Set S) (w : ℕ → S) {N m : ℕ} (hN : m ≤ (rawIn A w N).card) :
    ∃ n ≤ N, (rawIn A w n).card = m := by
  classical
  induction N generalizing m with
  | zero =>
    refine ⟨0, le_rfl, ?_⟩
    have : rawIn A w 0 = ∅ := by simp [rawIn]
    rw [this] at hN ⊢
    simp at hN ⊢; omega
  | succ N ih =>
    by_cases h : m ≤ (rawIn A w N).card
    · obtain ⟨n, hn, he⟩ := ih h
      exact ⟨n, by omega, he⟩
    · have := card_rawIn_succ_le A w N
      exact ⟨N + 1, le_rfl, by omega⟩

lemma acceptedAt_sub_eq_rawIn (Q : Set S) (d : S) (w : ℕ → S) (T : Finset S) (n : ℕ) :
    acceptedAt T (nthDrawIn Q d w) (drawsIn Q w n).length = rawIn Q w n \ T := by
  rw [acceptedAt_sub]
  rfl

/-- A filled table is its harvested strings and a raw stretch of its side. -/
lemma table_eq_rawIn (L : List S) (hL : L.Nodup) (A : Set S) (hLA : ∀ s ∈ L, s ∉ A)
    (pad : S) (w : ℕ → S) (m n : ℕ) (hn : (rawIn A w n).card + L.length = m) :
    freshDraws ∅ (prependStream L (nthDrawIn A pad w)) m = L.toFinset ∪ rawIn A w n := by
  classical
  have hdisj : Disjoint L.toFinset (rawIn A w n) := by
    rw [Finset.disjoint_left]
    intro s hs hs'
    exact hLA s (List.mem_toFinset.1 hs) (Finset.mem_filter.1 hs').2
  have hsd : rawIn A w n \ L.toFinset = rawIn A w n := Finset.sdiff_eq_self_of_disjoint hdisj.symm
  have hk : acceptedAt ∅ (prependStream L (nthDrawIn A pad w)) (L.length + (drawsIn A w n).length)
      = L.toFinset ∪ rawIn A w n := by
    rw [acceptedAt_prepend, acceptedAt_sub_eq_rawIn, hsd]
  rw [freshDraws_eq_of_card _ _ m _ (by
    rw [hk, Finset.card_union_of_disjoint hdisj, List.toFinset_card_of_nodup hL]; omega), hk]

/-- A filled certification sample is a raw stretch of its side, outside the tables. -/
lemma cert_eq_rawIn (R : Set S) (pad : S) (w : ℕ → S) (T : Finset S) (m n : ℕ)
    (hn : (rawIn R w n \ T).card = m) :
    freshDraws T (nthDrawIn R pad w) m = rawIn R w n \ T := by
  rw [freshDraws_eq_of_card _ _ m _ (by rw [acceptedAt_sub_eq_rawIn]; exact hn),
    acceptedAt_sub_eq_rawIn]

end OrthoDFA
