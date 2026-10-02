import Mathlib.Probability.ProductMeasure

namespace OrthoDFA

open MeasureTheory

variable {S : Type*} [MeasurableSpace S] {J : Type*} [Fintype J] [DecidableEq J]

/-- Certification draws at distinct coordinates land in their sets independently. -/
lemma certCoords_eq (D : J → Measure S) [∀ j, IsProbabilityMeasure (D j)]
    {ι : Type*} [DecidableEq ι] (s : Finset ι) (c : ι → J × ℕ) (hc : Set.InjOn c s)
    (W : J → Set S) (hW : ∀ j, MeasurableSet (W j)) :
    (Measure.pi fun j : J => Measure.infinitePi fun _ : ℕ => D j)
        {x | ∀ r ∈ s, x (c r).1 (c r).2 ∈ W (c r).1}
      = ∏ r ∈ s, D (c r).1 (W (c r).1) := by
  classical
  set K : J → Finset ℕ := fun j => (s.filter (fun r => (c r).1 = j)).image (fun r => (c r).2)
    with hK
  have hset : {x : J → ℕ → S | ∀ r ∈ s, x (c r).1 (c r).2 ∈ W (c r).1}
      = Set.univ.pi (fun j => Set.pi (↑(K j) : Set ℕ) (fun _ => W j)) := by
    ext x
    simp only [Set.mem_setOf_eq, Set.mem_pi, Finset.mem_coe]
    constructor
    · intro h j _ k hk
      obtain ⟨r, hr, hrk⟩ := Finset.mem_image.1 hk
      obtain ⟨hr, hrj⟩ := Finset.mem_filter.1 hr
      rw [← hrk, ← hrj]
      exact h r hr
    · intro h r hr
      exact h (c r).1 (Set.mem_univ _) (c r).2 (Finset.mem_image.2 ⟨r, Finset.mem_filter.2 ⟨hr, rfl⟩, rfl⟩)
  rw [hset, Measure.pi_pi]
  have hinj : ∀ j, Set.InjOn (fun r => (c r).2) ↑(s.filter (fun r => (c r).1 = j)) := by
    intro j r hr r' hr' h
    simp only [Finset.coe_filter, Set.mem_setOf_eq] at hr hr'
    exact hc hr.1 hr'.1 (Prod.ext (hr.2.trans hr'.2.symm) h)
  calc ∏ j, Measure.infinitePi (fun _ : ℕ => D j) (Set.pi (↑(K j) : Set ℕ) (fun _ => W j))
      = ∏ j, ∏ r ∈ s.filter (fun r => (c r).1 = j), D j (W j) := by
        refine Finset.prod_congr rfl (fun j _ => ?_)
        rw [Measure.infinitePi_pi (μ := fun _ : ℕ => D j) (s := K j) (t := fun _ => W j)
          (fun _ _ => hW j)]
        simp only [hK]
        rw [Finset.prod_image (hinj j)]
    _ = ∏ j, ∏ r ∈ s.filter (fun r => (c r).1 = j), D (c r).1 (W (c r).1) := by
        refine Finset.prod_congr rfl (fun j _ => Finset.prod_congr rfl (fun r hr => ?_))
        rw [(Finset.mem_filter.1 hr).2]
    _ = ∏ r ∈ s, D (c r).1 (W (c r).1) :=
        Finset.prod_fiberwise s (fun r => (c r).1) (fun r => D (c r).1 (W (c r).1))

lemma block_index_eq {v r r' i i' : ℕ} (hi : i < v) (hi' : i' < v)
    (h : r * v + i = r' * v + i') : r = r' := by
  rcases lt_trichotomy r r' with hlt | heq | hgt
  · have : r * v + v ≤ r' * v := by rw [← Nat.succ_mul]; exact Nat.mul_le_mul_right v hlt
    omega
  · exact heq
  · have : r' * v + v ≤ r * v := by rw [← Nat.succ_mul]; exact Nat.mul_le_mul_right v hgt
    omega

open scoped Classical in
/-- Of `R` blocks of `v` certification draws each, at least `m` holding a draw that lands in its
population's set: at most `C(R, m)·h^m`, `h` the chance one block does. -/
lemma manyBlocks_le (D : J → Measure S) [∀ j, IsProbabilityMeasure (D j)]
    (Jv : Finset J) (W : J → Set S) (hW : ∀ j, MeasurableSet (W j)) (n v R m : ℕ) :
    (Measure.pi fun j : J => Measure.infinitePi fun _ : ℕ => D j)
        {x | m ≤ ((Finset.range R).filter (fun r =>
          ∃ j ∈ Jv, ∃ i < v, x j (n + r * v + i) ∈ W j)).card}
      ≤ (R.choose m : ENNReal) * (∑ j ∈ Jv, (v : ENNReal) * D j (W j)) ^ m := by
  classical
  set ν := Measure.pi fun j : J => Measure.infinitePi fun _ : ℕ => D j with hν
  set H : ℕ → Set (J → ℕ → S) := fun r => {x | ∃ j ∈ Jv, ∃ i < v, x j (n + r * v + i) ∈ W j}
    with hH
  set P : Finset (J × ℕ) := Jv ×ˢ Finset.range v with hP
  have hsub : {x : J → ℕ → S | m ≤ ((Finset.range R).filter (fun r =>
        ∃ j ∈ Jv, ∃ i < v, x j (n + r * v + i) ∈ W j)).card}
      ⊆ ⋃ T ∈ (Finset.range R).powersetCard m, ⋂ r ∈ T, H r := by
    intro x hx
    obtain ⟨T, hT, hTc⟩ := Finset.exists_subset_card_eq hx
    refine Set.mem_biUnion (Finset.mem_powersetCard.2 ⟨fun r hr => ?_, hTc⟩) ?_
    · exact (Finset.mem_filter.1 (hT hr)).1
    · exact Set.mem_biInter (fun r hr => (Finset.mem_filter.1 (hT hr)).2)
  have hblock : ∀ T ∈ (Finset.range R).powersetCard m,
      ν (⋂ r ∈ T, H r) ≤ (∑ j ∈ Jv, (v : ENNReal) * D j (W j)) ^ m := by
    intro T hT
    have hTc := (Finset.mem_powersetCard.1 hT).2
    have hcov : (⋂ r ∈ T, H r) ⊆ ⋃ g ∈ T.pi (fun _ => P),
        {x | ∀ r ∈ T.attach, x (g r.1 r.2).1 (n + r.1 * v + (g r.1 r.2).2)
          ∈ W (g r.1 r.2).1} := by
      intro x hx
      have hx' : ∀ r ∈ T, ∃ q ∈ P, x q.1 (n + r * v + q.2) ∈ W q.1 := by
        intro r hr
        obtain ⟨j, hj, i, hi, hxi⟩ := Set.mem_iInter₂.1 hx r hr
        exact ⟨(j, i), Finset.mem_product.2 ⟨hj, Finset.mem_range.2 hi⟩, hxi⟩
      choose g hgP hg using hx'
      refine Set.mem_biUnion (Finset.mem_pi.2 (fun r hr => hgP r hr)) ?_
      exact fun r _ => hg r.1 r.2
    refine le_trans (measure_mono hcov) (le_trans (measure_biUnion_finset_le _ _) ?_)
    have heach : ∀ g ∈ T.pi (fun _ => P),
        ν {x | ∀ r ∈ T.attach, x (g r.1 r.2).1 (n + r.1 * v + (g r.1 r.2).2) ∈ W (g r.1 r.2).1}
          = ∏ r ∈ T.attach, D (g r.1 r.2).1 (W (g r.1 r.2).1) := by
      intro g hg
      refine certCoords_eq D T.attach (fun r => ((g r.1 r.2).1, n + r.1 * v + (g r.1 r.2).2))
        (fun r _ r' _ h => ?_) W hW
      simp only [Prod.mk.injEq] at h
      have hi := Finset.mem_range.1 (Finset.mem_product.1 (Finset.mem_pi.1 hg r.1 r.2)).2
      have hi' := Finset.mem_range.1 (Finset.mem_product.1 (Finset.mem_pi.1 hg r'.1 r'.2)).2
      exact Subtype.ext (block_index_eq hi hi' (by omega))
    rw [Finset.sum_congr rfl heach,
      ← Finset.prod_sum (s := T) (t := fun _ => P) (f := fun _ q => D q.1 (W q.1)),
      Finset.prod_const, hTc]
    refine le_of_eq (congrArg (· ^ m) ?_)
    rw [hP, Finset.sum_product]
    refine Finset.sum_congr rfl (fun j _ => ?_)
    simp only
    rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul]
  calc ν _ ≤ ν (⋃ T ∈ (Finset.range R).powersetCard m, ⋂ r ∈ T, H r) := measure_mono hsub
    _ ≤ ∑ T ∈ (Finset.range R).powersetCard m, ν (⋂ r ∈ T, H r) := measure_biUnion_finset_le _ _
    _ ≤ ∑ T ∈ (Finset.range R).powersetCard m, (∑ j ∈ Jv, (v : ENNReal) * D j (W j)) ^ m :=
        Finset.sum_le_sum hblock
    _ = (R.choose m : ENNReal) * (∑ j ∈ Jv, (v : ENNReal) * D j (W j)) ^ m := by
        rw [Finset.sum_const, Finset.card_powersetCard, Finset.card_range, nsmul_eq_mul]

end OrthoDFA
