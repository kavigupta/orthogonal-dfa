import OrthoDFA.Proofs.RawRun
import OrthoDFA.Proofs.RawWindow

/-!
# A named population's sample, against its table

A named population's table and certification sample are the two sides of one stream.  On a
window of its draws the hash splits every member's flips evenly, so a family whose members
rarely flip on the tables rarely flips on the certification sample either: the sample's heavy
prefixes are the members' table flips, the hash's deviation, and what the sample holds beyond
the window.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory Thin
open scoped ENNReal

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
variable {S : Type*} [Stringlike S]
variable {J : Type*} [Fintype J]

section Count

open scoped Classical in
/-- The family's flips summed over a sample are its members' flip counts on the sample. -/
lemma sum_flipCount_eq (O : Oracle μ S) (F C : Finset S) :
    ∑ p ∈ C, flipCount O F p = ∑ v ∈ F, ((C.filter (fun p => O.flip v p = 1)).card : ℝ) := by
  unfold flipCount
  simp only [Finset.card_filter, Nat.cast_sum, Nat.cast_ite, Nat.cast_one, Nat.cast_zero]
  exact Finset.sum_comm

open scoped Classical in
/-- A sample's heavy prefixes in a class `Q`: on the window, the members' reserve-side flips on the
class, which the hash holds to their table-side flips on it and these, summed over the family, to
its flips on the class on the tables; beyond the window, every prefix. -/
theorem heavy_count_le (O : Oracle μ S) (F C Z P : Finset S) (g : S → Bool) (Q : S → Prop)
    {f φ L EC : ℝ} (hf : 0 < f) (hL : 0 ≤ L) (hφ0 : 0 ≤ φ)
    (hflips : ∑ v ∈ F, ((P.filter (fun p => O.flip v p = 1 ∧ Q p)).card : ℝ) ≤ (F.card : ℝ) * φ)
    (hZA : Z.filter (fun s => g s = true) ⊆ P)
    (hbal : ∀ v ∈ F,
      (((Z.filter (fun p => O.flip v p = 1 ∧ Q p)).filter (fun s => g s = false)).card : ℝ)
        - (((Z.filter (fun p => O.flip v p = 1 ∧ Q p)).filter (fun s => g s = true)).card : ℝ)
      ≤ Real.sqrt (2 * ((Z.filter (fun p => O.flip v p = 1 ∧ Q p)).card : ℝ) * L))
    (hCZ : ∀ p ∈ C, p ∈ Z → g p = false) (hEC : ((C \ Z).card : ℝ) ≤ EC) :
    ((C.filter (fun p => Q p ∧ (F.card : ℝ) * f < flipCount O F p)).card : ℝ)
      ≤ (φ + 2 * Real.sqrt (L * φ) + 2 * L) / f + EC := by
  set CZ := (C.filter (fun p => p ∈ Z)).filter Q with hCZdef
  set T : S → ℝ := fun v => ((P.filter (fun p => O.flip v p = 1 ∧ Q p)).card : ℝ) with hT
  have hT0 : ∀ v, 0 ≤ T v := fun v => Nat.cast_nonneg _
  have hX : ∀ v ∈ F, ((CZ.filter (fun p => O.flip v p = 1)).card : ℝ)
      ≤ T v + 2 * Real.sqrt (L * T v) + 2 * L := by
    intro v hv
    set Fv := Z.filter (fun p => O.flip v p = 1 ∧ Q p) with hFv
    set X : ℝ := ((Fv.filter (fun s => g s = false)).card : ℝ) with hXdef
    set Y : ℝ := ((Fv.filter (fun s => g s = true)).card : ℝ) with hYdef
    have hXY : (Fv.card : ℝ) = X + Y := by
      have h := Finset.card_filter_add_card_filter_not (s := Fv) (fun s => g s = false)
      have e : Fv.filter (fun s => ¬ g s = false) = Fv.filter (fun s => g s = true) :=
        Finset.filter_congr (fun s _ => by simp)
      rw [e] at h
      rw [hXdef, hYdef]
      exact_mod_cast h.symm
    have hYT : Y ≤ T v := by
      have hsub : Fv.filter (fun s => g s = true) ⊆ P.filter (fun p => O.flip v p = 1 ∧ Q p) := by
        intro s hs
        simp only [hFv, Finset.mem_filter] at hs
        exact Finset.mem_filter.2 ⟨hZA (Finset.mem_filter.2 ⟨hs.1.1, hs.2⟩), hs.1.2⟩
      have hc := Finset.card_le_card hsub
      show ((Fv.filter (fun s => g s = true)).card : ℝ)
        ≤ ((P.filter (fun p => O.flip v p = 1 ∧ Q p)).card : ℝ)
      exact_mod_cast hc
    have hb := le_of_balance (X := X) (Y := Y) (L := L) (Nat.cast_nonneg _) hL (Nat.cast_nonneg _)
      (by rw [← hXY]; exact hbal v hv)
    have hsub2 : CZ.filter (fun p => O.flip v p = 1) ⊆ Fv.filter (fun s => g s = false) := by
      intro p hp
      simp only [hCZdef, Finset.mem_filter] at hp
      exact Finset.mem_filter.2 ⟨Finset.mem_filter.2 ⟨hp.1.1.2, hp.2, hp.1.2⟩,
        hCZ p hp.1.1.1 hp.1.1.2⟩
    have h2 : ((CZ.filter (fun p => O.flip v p = 1)).card : ℝ) ≤ X := by
      rw [hXdef]; exact_mod_cast Finset.card_le_card hsub2
    have h3 : Real.sqrt (L * Y) ≤ Real.sqrt (L * T v) :=
      Real.sqrt_le_sqrt (mul_le_mul_of_nonneg_left hYT hL)
    linarith
  -- summed over the family: Cauchy–Schwarz on the square roots
  have hroot : ∑ v ∈ F, Real.sqrt (L * T v) ≤ (F.card : ℝ) * Real.sqrt (L * φ) := by
    have hcs := sq_sum_le_card_mul_sum_sq (s := F) (f := fun v => Real.sqrt (L * T v))
    have e : ∑ v ∈ F, Real.sqrt (L * T v) ^ 2 = L * ∑ v ∈ F, T v := by
      rw [Finset.mul_sum]
      exact Finset.sum_congr rfl (fun v _ => Real.sq_sqrt (mul_nonneg hL (hT0 v)))
    rw [e] at hcs
    have hLT : L * ∑ v ∈ F, T v ≤ L * ((F.card : ℝ) * φ) := mul_le_mul_of_nonneg_left hflips hL
    have hsq : (∑ v ∈ F, Real.sqrt (L * T v)) ^ 2 ≤ ((F.card : ℝ) * Real.sqrt (L * φ)) ^ 2 := by
      rw [mul_pow, Real.sq_sqrt (mul_nonneg hL hφ0)]
      have hc0 : (0 : ℝ) ≤ F.card := Nat.cast_nonneg _
      nlinarith [mul_le_mul_of_nonneg_left hLT hc0]
    exact le_of_pow_le_pow_left₀ two_ne_zero (by positivity) hsq
  have hK0 : 0 ≤ φ + 2 * Real.sqrt (L * φ) + 2 * L := by positivity
  set hvy : ℝ := ((CZ.filter (fun p => (F.card : ℝ) * f < flipCount O F p)).card : ℝ)
    with hhvy
  have hheavyZ : hvy ≤ (φ + 2 * Real.sqrt (L * φ) + 2 * L) / f := by
    rcases Nat.eq_zero_or_pos F.card with h0 | hpos
    · have hF : F = ∅ := Finset.card_eq_zero.1 h0
      have hE : CZ.filter (fun p => (F.card : ℝ) * f < flipCount O F p) = ∅ := by
        refine Finset.filter_false_of_mem (fun p _ => ?_)
        simp [flipCount, hF]
      rw [hhvy, hE, Finset.card_empty, Nat.cast_zero]
      positivity
    · have hFR : (0 : ℝ) < F.card := by exact_mod_cast hpos
      have hle : hvy * ((F.card : ℝ) * f) ≤ ∑ p ∈ CZ, flipCount O F p := by
        calc hvy * ((F.card : ℝ) * f)
            = ∑ _p ∈ CZ.filter (fun p => (F.card : ℝ) * f < flipCount O F p),
                (F.card : ℝ) * f := by rw [Finset.sum_const, nsmul_eq_mul]
          _ ≤ ∑ p ∈ CZ.filter (fun p => (F.card : ℝ) * f < flipCount O F p),
                flipCount O F p :=
              Finset.sum_le_sum (fun p hp => (Finset.mem_filter.1 hp).2.le)
          _ ≤ ∑ p ∈ CZ, flipCount O F p :=
              Finset.sum_le_sum_of_subset_of_nonneg (Finset.filter_subset _ _)
                (fun p _ _ => by unfold flipCount; positivity)
      have hsumle : ∑ p ∈ CZ, flipCount O F p
          ≤ (F.card : ℝ) * (φ + 2 * Real.sqrt (L * φ) + 2 * L) := by
        rw [sum_flipCount_eq]
        refine le_trans (Finset.sum_le_sum hX) ?_
        rw [Finset.sum_add_distrib, Finset.sum_add_distrib, ← Finset.mul_sum, Finset.sum_const,
          nsmul_eq_mul]
        have h1 : ∑ v ∈ F, T v ≤ (F.card : ℝ) * φ := hflips
        nlinarith [hroot]
      have hcomb := le_trans hle hsumle
      rw [le_div_iff₀ hf]
      have h3 : (hvy * f) * (F.card : ℝ)
          ≤ (φ + 2 * Real.sqrt (L * φ) + 2 * L) * (F.card : ℝ) := by linarith
      exact le_of_mul_le_mul_right h3 hFR
  have hsplit : C.filter (fun p => Q p ∧ (F.card : ℝ) * f < flipCount O F p)
      ⊆ CZ.filter (fun p => (F.card : ℝ) * f < flipCount O F p) ∪ (C \ Z) := by
    intro p hp
    obtain ⟨hpC, hq, hph⟩ := Finset.mem_filter.1 hp
    by_cases hpZ : p ∈ Z
    · exact Finset.mem_union_left _ (Finset.mem_filter.2 ⟨Finset.mem_filter.2
        ⟨Finset.mem_filter.2 ⟨hpC, hpZ⟩, hq⟩, hph⟩)
    · exact Finset.mem_union_right _ (Finset.mem_sdiff.2 ⟨hpC, hpZ⟩)
  have hc := Finset.card_le_card hsplit
  have hc' := Finset.card_union_le (CZ.filter (fun p => (F.card : ℝ) * f < flipCount O F p))
    (C \ Z)
  have hcR : ((C.filter (fun p => Q p ∧ (F.card : ℝ) * f < flipCount O F p)).card : ℝ)
      ≤ hvy + ((C \ Z).card : ℝ) := by rw [hhvy]; exact_mod_cast le_trans hc hc'
  linarith

end Count

section Geometry

open scoped Classical in
/-- The window's table side lies in the filled table when it is no larger than the table's
share of the draws. -/
lemma window_table_subset (L : List S) (hL : L.Nodup) (Hs : Finset S) (hLH : ∀ s ∈ L, s ∈ Hs)
    (g : S → Bool) (w z : ℕ → S) (K N : ℕ)
    (hT : (freshDraws ∅ (tableSeqOf L (onSide Hs true g) w z) N).card = N)
    (hA : ((window (↑Hs)ᶜ K w).filter (fun s => g s = true)).card + L.length ≤ N) :
    (window (↑Hs)ᶜ K w).filter (fun s => g s = true)
      ⊆ freshDraws ∅ (tableSeqOf L (onSide Hs true g) w z) N := by
  set A := onSide Hs true g with hAdef
  set n₀ := winIdx (↑Hs)ᶜ K w with hn₀
  have hAQ : A ⊆ (↑Hs)ᶜ := fun s hs => hs.2
  have hZA : (window (↑Hs)ᶜ K w).filter (fun s => g s = true) = rawIn A w n₀ := by
    rw [rawIn_filter hAQ w n₀]
    refine Finset.filter_congr (fun s hs => ?_)
    have hsQ : s ∉ Hs := by
      have := mem_of_mem_rawIn hs
      simpa using this
    simp only [hAdef, onSide, Set.mem_ofPred_eq, hsQ, not_false_eq_true, and_true]
  set pad := L.headD (nthDrawIn A (z 0) w 0) with hpad
  have hseq : tableSeqOf L A w z = prependStream L (nthDrawIn A pad w) := rfl
  have hdisj : Disjoint L.toFinset (rawIn A w n₀) := by
    rw [Finset.disjoint_left]
    intro s hs hs'
    exact (Finset.mem_filter.1 hs').2.2 (hLH s (List.mem_toFinset.1 hs))
  have hacc : acceptedAt ∅ (tableSeqOf L A w z) (L.length + (drawsIn A w n₀).length)
      = L.toFinset ∪ rawIn A w n₀ := by
    rw [hseq, acceptedAt_prepend, acceptedAt_sub_eq_rawIn,
      Finset.sdiff_eq_self_of_disjoint hdisj.symm]
  have hcard : (acceptedAt ∅ (tableSeqOf L A w z) (L.length + (drawsIn A w n₀).length)).card
      ≤ N := by
    rw [hacc]
    refine le_trans (Finset.card_union_le _ _) ?_
    rw [List.toFinset_card_of_nodup hL, ← hZA]
    omega
  have hsub := acceptedAt_subset_of_card_le ∅ _ N _ hT hcard
  rw [hacc] at hsub
  rw [hZA]
  exact fun s hs => hsub (Finset.mem_union_right _ hs)

open scoped Classical in
/-- A filled certification sample holds no more beyond the window's reserve side outside the
tables than the sample's count less that side's. -/
lemma card_cert_sdiff_le (Hs : Finset S) (g : S → Bool) (w z : ℕ → S) (K N : ℕ)
    (T : Finset S) (hC : (freshDraws T (reserveSeqOf (onSide Hs false g) w z) N).card = N) :
    ((freshDraws T (reserveSeqOf (onSide Hs false g) w z) N
        \ ((window (↑Hs)ᶜ K w).filter (fun s => g s = false) \ T)).card : ℝ)
      ≤ max 0 ((N : ℝ) - (((window (↑Hs)ᶜ K w).filter (fun s => g s = false) \ T).card : ℝ)) := by
  set R := onSide Hs false g with hRdef
  set n₀ := winIdx (↑Hs)ᶜ K w with hn₀
  have hRQ : R ⊆ (↑Hs)ᶜ := fun s hs => hs.2
  have hZR : (window (↑Hs)ᶜ K w).filter (fun s => g s = false) = rawIn R w n₀ := by
    rw [rawIn_filter hRQ w n₀]
    refine Finset.filter_congr (fun s hs => ?_)
    have hsQ : s ∉ Hs := by
      have := mem_of_mem_rawIn hs
      simpa using this
    simp only [hRdef, onSide, Set.mem_ofPred_eq, hsQ, not_false_eq_true, and_true]
  have hseq : reserveSeqOf R w z = nthDrawIn R (nthDrawIn R (z 1) w 0) w := rfl
  have hacc : acceptedAt T (reserveSeqOf R w z) (drawsIn R w n₀).length = rawIn R w n₀ \ T := by
    rw [hseq, acceptedAt_sub_eq_rawIn]
  rw [hZR, ← hacc]
  exact card_sdiff_le_of_comparable T _ N _ hC

/-- When the window is not full, a filled certification sample lies inside it. -/
lemma cert_subset_window (Hs : Finset S) (g : S → Bool) (w z : ℕ → S) (K N : ℕ) (hN : 2 ≤ N)
    (T : Finset S) (hfull : (window (↑Hs)ᶜ K w).card ≠ K) :
    freshDraws T (reserveSeqOf (onSide Hs false g) w z) N ⊆ window (↑Hs)ᶜ K w := by
  intro s hs
  obtain ⟨i, hi, rfl⟩ := mem_cert_of_two _ w z T hN hs
  exact mem_window_of_not_full hfull hi.2

end Geometry

end OrthoDFA
