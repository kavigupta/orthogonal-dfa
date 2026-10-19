import OrthoDFA.Proofs.SideMeas
import OrthoDFA.Proofs.CoinWindow

/-!
# A window of a population's draws

The window is the first `K` distinct draws in a set, or all of them when there are fewer.  It is
read off the draws alone, so the coins on it are independent of it, and a table or certification
sample read against it either lies inside it or holds the window's share of its side.
-/

namespace OrthoDFA

open MeasureTheory Thin

variable {S : Type*} [Stringlike S]

lemma acceptedAt_congr (T : Finset S) {y y' : ℕ → S} {n : ℕ} (h : ∀ k < n, y k = y' k) :
    acceptedAt T y n = acceptedAt T y' n := by
  unfold acceptedAt
  congr 1
  exact Finset.image_congr (fun k hk => h k (Finset.mem_range.1 hk))

/-- Two streams agreeing on a stretch that already offers `m` fresh values sample alike. -/
lemma freshDraws_congr (T : Finset S) {y y' : ℕ → S} {n m : ℕ} (h : ∀ k < n, y k = y' k)
    (hm : m ≤ (acceptedAt T y n).card) : freshDraws T y m = freshDraws T y' m := by
  classical
  have hagree : ∀ k ≤ n, acceptedAt T y k = acceptedAt T y' k :=
    fun k hk => acceptedAt_congr T (fun i hi => h i (by omega))
  have hy : ∃ k, m ≤ (acceptedAt T y k).card := ⟨n, hm⟩
  have hy' : ∃ k, m ≤ (acceptedAt T y' k).card := ⟨n, by rw [← hagree n le_rfl]; exact hm⟩
  have h1 : Nat.find hy ≤ n := Nat.find_min' hy hm
  have h2 : Nat.find hy' ≤ Nat.find hy :=
    Nat.find_min' hy' (by rw [← hagree _ h1]; exact Nat.find_spec hy)
  have h3 : Nat.find hy ≤ Nat.find hy' :=
    Nat.find_min' hy (by rw [hagree _ (le_trans h2 h1)]; exact Nat.find_spec hy')
  rw [freshDraws_eq T y m hy, freshDraws_eq T y' m hy', le_antisymm h3 h2]
  exact hagree _ (le_trans h2 h1)

/-- A stream of one value never fills a sample of two or more. -/
lemma freshDraws_const (T : Finset S) (c : S) {y : ℕ → S} (hy : ∀ k, y k = c) {m : ℕ}
    (hm : 2 ≤ m) : freshDraws T y m = ∅ := by
  classical
  refine freshDraws_of_not T y m (fun ⟨n, hn⟩ => ?_)
  have hsub : acceptedAt T y n ⊆ {c} := by
    intro s hs
    obtain ⟨k, -, rfl⟩ := Finset.mem_image.1 (Finset.mem_sdiff.1 hs).1
    rw [hy k]
    exact Finset.mem_singleton_self c
  have := Finset.card_le_card hsub
  rw [Finset.card_singleton] at this
  omega

/-- An entry of a subsequence is a draw in its set, or the spare. -/
lemma nthDrawIn_spec (A : Set S) (d : S) (w : ℕ → S) (k : ℕ) :
    nthDrawIn A d w k = d ∨ ∃ i, w i ∈ A ∧ w i = nthDrawIn A d w k := by
  classical
  by_cases h : ∃ n, k < (drawsIn A w n).length
  · right
    obtain ⟨n, hn⟩ := h
    rw [sub_eq_getD A d w hn, List.getD_eq_getElem _ _ hn]
    have hmem : (drawsIn A w n)[k] ∈ drawsIn A w n := List.getElem_mem hn
    simp only [drawsIn, List.mem_filter, List.mem_map, List.mem_range, decide_eq_true_eq] at hmem
    obtain ⟨⟨i, -, he⟩, hA⟩ := hmem
    exact ⟨i, he ▸ hA, he⟩
  · left
    rw [nthDrawIn, dif_neg h]

lemma nthDrawIn_zero_mem (A : Set S) (d : S) (w : ℕ → S) {i : ℕ} (hi : w i ∈ A) :
    ∃ i', w i' ∈ A ∧ w i' = nthDrawIn A d w 0 := by
  classical
  rcases nthDrawIn_spec A d w 0 with h | h
  · have hlen : 0 < (drawsIn A w (i + 1)).length := by
      rw [entries_succ, if_pos hi]
      simp
    have := sub_eq_getD A d w hlen
    rw [List.getD_eq_getElem _ _ hlen] at this
    have hmem : (drawsIn A w (i + 1))[0] ∈ drawsIn A w (i + 1) := List.getElem_mem hlen
    simp only [drawsIn, List.mem_filter, List.mem_map, List.mem_range,
      decide_eq_true_eq] at hmem
    obtain ⟨⟨i', -, he⟩, hA⟩ := hmem
    exact ⟨i', he ▸ hA, he.trans this.symm⟩
  · exact h

/-- Every entry of a certification sequence is a reserve-side draw, unless there is none. -/
lemma reserveSeqOf_spec (R : Set S) (w z : ℕ → S) :
    (∀ k, reserveSeqOf R w z k = z 1) ∨ ∀ k, ∃ i, w i ∈ R ∧ w i = reserveSeqOf R w z k := by
  classical
  by_cases hex : ∃ i, w i ∈ R
  · right
    obtain ⟨i, hi⟩ := hex
    obtain ⟨i₀, hi₀, he₀⟩ := nthDrawIn_zero_mem R (z 1) w hi
    intro k
    unfold reserveSeqOf
    rcases nthDrawIn_spec R (nthDrawIn R (z 1) w 0) w k with h | h
    · exact ⟨i₀, hi₀, he₀.trans h.symm⟩
    · exact h
  · left
    push Not at hex
    intro k
    unfold reserveSeqOf
    rw [sub_eq_const_of_none R w hex, sub_eq_const_of_none R w hex]

/-- Every entry of a table sequence is a harvested string or a table-side draw, unless the
sequence is constant. -/
lemma tableSeqOf_spec (L : List S) (A : Set S) (w z : ℕ → S) :
    (∀ k, tableSeqOf L A w z k = z 0)
      ∨ ∀ k, tableSeqOf L A w z k ∈ L ∨ ∃ i, w i ∈ A ∧ w i = tableSeqOf L A w z k := by
  classical
  set pad := L.headD (nthDrawIn A (z 0) w 0) with hpad
  have hseq : ∀ k, tableSeqOf L A w z k ∈ L ∨ tableSeqOf L A w z k = pad
      ∨ ∃ i, w i ∈ A ∧ w i = tableSeqOf L A w z k := by
    intro k
    unfold tableSeqOf
    unfold prependStream
    split_ifs with hk
    · exact Or.inl (List.get_mem _ _)
    · rcases nthDrawIn_spec A pad w (k - L.length) with h | h
      · exact Or.inr (Or.inl h)
      · exact Or.inr (Or.inr h)
  rcases L with _ | ⟨a, L'⟩
  · by_cases hex : ∃ i, w i ∈ A
    · right
      obtain ⟨i, hi⟩ := hex
      obtain ⟨i₀, hi₀, he₀⟩ := nthDrawIn_zero_mem A (z 0) w hi
      intro k
      rcases hseq k with h | h | h
      · exact Or.inl h
      · right
        refine ⟨i₀, hi₀, ?_⟩
        rw [h, hpad]
        exact he₀
      · exact Or.inr h
    · left
      push Not at hex
      intro k
      unfold tableSeqOf prependStream
      simp only [List.length_nil, Nat.not_lt_zero, dif_neg, not_false_eq_true,
        List.headD_nil, Nat.sub_zero]
      rw [sub_eq_const_of_none A w hex, sub_eq_const_of_none A w hex]
  · right
    intro k
    rcases hseq k with h | h | h
    · exact Or.inl h
    · left
      rw [h, hpad]
      exact List.mem_cons_self
    · exact Or.inr h

/-- A filled table holds only harvested strings and table-side draws. -/
lemma mem_table_of_two (L : List S) (A : Set S) (w z : ℕ → S) {m : ℕ} (hm : 2 ≤ m)
    {s : S} (hs : s ∈ freshDraws ∅ (tableSeqOf L A w z) m) : s ∈ L ∨ s ∈ A := by
  rcases tableSeqOf_spec L A w z with h | h
  · rw [freshDraws_const ∅ (z 0) h hm] at hs
    simp at hs
  · obtain ⟨k, rfl⟩ := mem_freshDraws hs
    rcases h k with h' | ⟨i, hi, he⟩
    · exact Or.inl h'
    · exact Or.inr (he ▸ hi)

/-- A filled certification sample holds only reserve-side draws. -/
lemma mem_cert_of_two (R : Set S) (w z : ℕ → S) (T : Finset S) {m : ℕ} (hm : 2 ≤ m)
    {s : S} (hs : s ∈ freshDraws T (reserveSeqOf R w z) m) : ∃ i, w i ∈ R ∧ w i = s := by
  rcases reserveSeqOf_spec R w z with h | h
  · rw [freshDraws_const T (z 1) h hm] at hs
    simp at hs
  · obtain ⟨k, rfl⟩ := mem_freshDraws hs
    exact h k

/-! ### The window -/

open scoped Classical in
lemma rawIn_filter {A Q : Set S} (hAQ : A ⊆ Q) (w : ℕ → S) (n : ℕ) :
    rawIn A w n = (rawIn Q w n).filter (fun s => s ∈ A) := by
  classical
  unfold rawIn
  rw [Finset.filter_filter]
  exact Finset.filter_congr (fun s _ => ⟨fun h => ⟨hAQ h, h⟩, fun h => h.2⟩)

lemma mem_rawIn {A : Set S} {w : ℕ → S} {i n : ℕ} (hi : i < n) (hA : w i ∈ A) :
    w i ∈ rawIn A w n := by
  classical
  exact Finset.mem_filter.2 ⟨Finset.mem_image.2 ⟨i, Finset.mem_range.2 hi, rfl⟩, hA⟩

lemma mem_of_mem_rawIn {A : Set S} {w : ℕ → S} {n : ℕ} {s : S} (h : s ∈ rawIn A w n) :
    s ∈ A := by
  classical
  unfold rawIn at h
  exact (Finset.mem_filter.1 h).2

lemma rawIn_ssubset_max (Q : Set S) (w : ℕ → S) {n n' : ℕ}
    (h : ¬ rawIn Q w n' ⊆ rawIn Q w n) : rawIn Q w n ⊂ rawIn Q w (max n n') :=
  Finset.ssubset_iff_subset_ne.2 ⟨rawIn_mono Q w (le_max_left _ _),
    fun he => h (by rw [he]; exact rawIn_mono Q w (le_max_right _ _))⟩

lemma exists_winIdx (Q : Set S) (K : ℕ) (w : ℕ → S) :
    ∃ n, K ≤ (rawIn Q w n).card ∨ ∀ n', rawIn Q w n' ⊆ rawIn Q w n := by
  by_contra hc
  push Not at hc
  have key : ∀ k, ∃ n, k ≤ (rawIn Q w n).card := by
    intro k
    induction k with
    | zero => exact ⟨0, Nat.zero_le _⟩
    | succ k ih =>
      obtain ⟨n, hn⟩ := ih
      obtain ⟨n', hn'⟩ := (hc n).2
      exact ⟨max n n', lt_of_le_of_lt hn (Finset.card_lt_card (rawIn_ssubset_max Q w hn'))⟩
  obtain ⟨n, hn⟩ := key K
  exact absurd hn (not_le.2 (hc n).1)

open scoped Classical in
/-- How many draws the window reads. -/
noncomputable def winIdx (Q : Set S) (K : ℕ) (w : ℕ → S) : ℕ := Nat.find (exists_winIdx Q K w)

/-- The first `K` distinct draws of `w` in `Q`, or all of them when there are fewer. -/
noncomputable def window (Q : Set S) (K : ℕ) (w : ℕ → S) : Finset S :=
  rawIn Q w (winIdx Q K w)

lemma window_card_le (Q : Set S) (K : ℕ) (w : ℕ → S) : (window Q K w).card ≤ K := by
  classical
  show (rawIn Q w (Nat.find (exists_winIdx Q K w))).card ≤ K
  set n := Nat.find (exists_winIdx Q K w) with hn
  rcases Nat.eq_zero_or_pos n with h0 | hpos
  · rw [h0]
    simp [rawIn]
  · have hmin := Nat.find_min (exists_winIdx Q K w) (show n - 1 < n by omega)
    push Not at hmin
    have hs := card_rawIn_succ_le Q w (n - 1)
    rw [show n - 1 + 1 = n by omega] at hs
    have := hmin.1
    omega

lemma window_full_or_all (Q : Set S) (K : ℕ) (w : ℕ → S) :
    (window Q K w).card = K ∨ ∀ n, rawIn Q w n ⊆ window Q K w := by
  classical
  rcases Nat.find_spec (exists_winIdx Q K w) with h | h
  · exact Or.inl (le_antisymm (window_card_le Q K w) h)
  · exact Or.inr h

/-- When the window is not full, every draw in `Q` is in it. -/
lemma mem_window_of_not_full {Q : Set S} {K : ℕ} {w : ℕ → S}
    (h : (window Q K w).card ≠ K) {i : ℕ} (hi : w i ∈ Q) : w i ∈ window Q K w := by
  rcases window_full_or_all Q K w with h' | h'
  · exact absurd h' h
  · exact h' (i + 1) (mem_rawIn (Nat.lt_succ_self i) hi)

/-- Every event about a stream's first `n` draws is measurable. -/
lemma measurableSet_of_firstN (n : ℕ) (P : (ℕ → S) → Prop)
    (hP : ∀ w w', (∀ k < n, w k = w' k) → (P w ↔ P w')) : MeasurableSet {w : ℕ → S | P w} := by
  set ext : (Fin n → S) → ℕ → S := fun t k => if h : k < n then t ⟨k, h⟩ else 1 with hext
  have hrw : {w : ℕ → S | P w} = {w | (fun t : Fin n → S => P (ext t)) (fun i => w i.val)} := by
    ext w
    simp only [Set.mem_ofPred_eq]
    exact hP _ _ (fun k hk => by simp [hext, hk])
  rw [hrw]
  exact measurableSet_firstN (X := S) n (fun t => P (ext t))

lemma measurableSet_rawIn (Q : Set S) (n : ℕ) (P : Finset S → Prop) :
    MeasurableSet {w : ℕ → S | P (rawIn Q w n)} := by
  classical
  refine measurableSet_of_firstN n _ (fun w w' h => ?_)
  have : rawIn Q w n = rawIn Q w' n := by
    unfold rawIn
    congr 1
    exact Finset.image_congr (fun k hk => h k (Finset.mem_range.1 hk))
  rw [this]

open scoped Classical in
lemma measurableSet_winIdx_eq (Q : Set S) (K n : ℕ) :
    MeasurableSet {w : ℕ → S | winIdx Q K w = n} := by
  classical
  have hP : ∀ m, MeasurableSet {w : ℕ → S |
      K ≤ (rawIn Q w m).card ∨ ∀ n', rawIn Q w n' ⊆ rawIn Q w m} := by
    intro m
    have : {w : ℕ → S | K ≤ (rawIn Q w m).card ∨ ∀ n', rawIn Q w n' ⊆ rawIn Q w m}
        = {w | K ≤ (rawIn Q w m).card} ∪ ⋂ n', {w | rawIn Q w n' ⊆ rawIn Q w m} := by
      ext w; simp
    rw [this]
    refine (measurableSet_rawIn Q m (fun z => K ≤ z.card)).union
      (MeasurableSet.iInter (fun n' : ℕ => ?_))
    refine measurableSet_of_firstN (max n' m) _ (fun w w' h => ?_)
    have h1 : rawIn Q w n' = rawIn Q w' n' := by
      unfold rawIn; congr 1
      exact Finset.image_congr (fun k hk => h k (by
        have := Finset.mem_range.1 hk; omega))
    have h2 : rawIn Q w m = rawIn Q w' m := by
      unfold rawIn; congr 1
      exact Finset.image_congr (fun k hk => h k (by
        have := Finset.mem_range.1 hk; omega))
    rw [h1, h2]
  have hrw : {w : ℕ → S | winIdx Q K w = n}
      = {w | K ≤ (rawIn Q w n).card ∨ ∀ n', rawIn Q w n' ⊆ rawIn Q w n}
        ∩ ⋂ m ∈ Finset.range n,
          {w | K ≤ (rawIn Q w m).card ∨ ∀ n', rawIn Q w n' ⊆ rawIn Q w m}ᶜ := by
    ext w
    simp only [Set.mem_ofPred_eq, Set.mem_inter_iff, Set.mem_iInter, Finset.mem_range,
      Set.mem_compl_iff, winIdx]
    rw [Nat.find_eq_iff]
  rw [hrw]
  exact (hP n).inter (Finset.measurableSet_biInter _ (fun m _ => (hP m).compl))

lemma measurableSet_window_eq (Q : Set S) (K : ℕ) (P : Finset S → Prop) :
    MeasurableSet {w : ℕ → S | P (window Q K w)} := by
  have hrw : {w : ℕ → S | P (window Q K w)}
      = ⋃ n : ℕ, {w | winIdx Q K w = n} ∩ {w | P (rawIn Q w n)} := by
    ext w
    simp only [Set.mem_ofPred_eq, Set.mem_iUnion, Set.mem_inter_iff]
    exact ⟨fun h => ⟨_, rfl, h⟩, fun ⟨n, hn, h⟩ => by unfold window; rw [hn]; exact h⟩
  rw [hrw]
  exact MeasurableSet.iUnion (fun n => (measurableSet_winIdx_eq Q K n).inter
    (measurableSet_rawIn Q n P))

/-! ### Events about the coins on a window -/

section Coins

variable {Ω : Type*} [MeasurableSpace Ω] {J : Type*}

/-- An event about the coins on a finite set the draws pick is measurable. -/
lemma measurableSet_coinEvent {X : Type*} [Countable X] (F : Run Ω S J → X)
    (hF : ∀ a, MeasurableSet {r | F r = a}) (U : X → Finset S) (Φ : X → (S → Bool) → Prop)
    (hΦ : ∀ a g g', (∀ s ∈ U a, g s = g' s) → (Φ a g ↔ Φ a g')) :
    MeasurableSet {x : Draws Ω S J | Φ (F x.1) x.2} := by
  have hrw : {x : Draws Ω S J | Φ (F x.1) x.2}
      = ⋃ a : X, {r | F r = a} ×ˢ {g : S → Bool | Φ a g} := by
    ext x
    simp only [Set.mem_ofPred_eq, Set.mem_iUnion, Set.mem_prod]
    exact ⟨fun h => ⟨_, rfl, h⟩, fun ⟨a, ha, h⟩ => ha ▸ h⟩
  rw [hrw]
  refine MeasurableSet.iUnion (fun a => (hF a).prod ?_)
  have hm : Measurable (fun (g : S → Bool) (s : U a) => g s) := by fun_prop
  have hset : {g : S → Bool | Φ a g}
      = (fun (g : S → Bool) (s : U a) => g s) ⁻¹'
        {γ | ∃ g : S → Bool, (∀ s : U a, g s = γ s) ∧ Φ a g} := by
    ext g
    simp only [Set.mem_ofPred_eq, Set.mem_preimage]
    refine ⟨fun h => ⟨g, fun _ => rfl, h⟩, fun ⟨g', hg', h⟩ => ?_⟩
    exact (hΦ a g' g (fun s hs => hg' ⟨s, hs⟩)).1 h
  rw [hset]
  exact hm (Set.to_countable _).measurableSet

end Coins

end OrthoDFA
