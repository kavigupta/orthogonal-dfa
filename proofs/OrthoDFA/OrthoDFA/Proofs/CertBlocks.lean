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
/-- Of `R` blocks of `v` certification draws each, at least `m` in which some population's draws
land in its set at `h + 1` places or more: at most `C(R, m)·(Σⱼ C(v, h + 1)·Dⱼ(Wⱼ)^(h+1))^m`. -/
lemma manyCrowded_le (D : J → Measure S) [∀ j, IsProbabilityMeasure (D j)]
    (Jv : Finset J) (W : J → Set S) (hW : ∀ j, MeasurableSet (W j)) (n v R m h : ℕ) :
    (Measure.pi fun j : J => Measure.infinitePi fun _ : ℕ => D j)
        {x | m ≤ ((Finset.range R).filter (fun r =>
          ∃ j ∈ Jv, ∃ T ∈ (Finset.range v).powersetCard (h + 1),
            ∀ i ∈ T, x j (n + r * v + i) ∈ W j)).card}
      ≤ (R.choose m : ENNReal)
        * (∑ j ∈ Jv, (v.choose (h + 1) : ENNReal) * D j (W j) ^ (h + 1)) ^ m := by
  classical
  set ν := Measure.pi fun j : J => Measure.infinitePi fun _ : ℕ => D j with hν
  set P : Finset (J × Finset ℕ) := Jv ×ˢ (Finset.range v).powersetCard (h + 1) with hP
  set H : ℕ → Set (J → ℕ → S) := fun r => {x | ∃ q ∈ P, ∀ i ∈ q.2, x q.1 (n + r * v + i) ∈ W q.1}
    with hH
  have hsub : {x : J → ℕ → S | m ≤ ((Finset.range R).filter (fun r =>
        ∃ j ∈ Jv, ∃ T ∈ (Finset.range v).powersetCard (h + 1),
          ∀ i ∈ T, x j (n + r * v + i) ∈ W j)).card}
      ⊆ ⋃ T ∈ (Finset.range R).powersetCard m, ⋂ r ∈ T, H r := by
    intro x hx
    obtain ⟨T, hT, hTc⟩ := Finset.exists_subset_card_eq hx
    refine Set.mem_biUnion (Finset.mem_powersetCard.2 ⟨fun r hr => ?_, hTc⟩) ?_
    · exact (Finset.mem_filter.1 (hT hr)).1
    · refine Set.mem_biInter (fun r hr => ?_)
      obtain ⟨j, hj, U, hU, hxU⟩ := (Finset.mem_filter.1 (hT hr)).2
      exact ⟨(j, U), Finset.mem_product.2 ⟨hj, hU⟩, hxU⟩
  have hblock : ∀ T ∈ (Finset.range R).powersetCard m,
      ν (⋂ r ∈ T, H r)
        ≤ (∑ j ∈ Jv, (v.choose (h + 1) : ENNReal) * D j (W j) ^ (h + 1)) ^ m := by
    intro T hT
    have hTc := (Finset.mem_powersetCard.1 hT).2
    have hcov : (⋂ r ∈ T, H r) ⊆ ⋃ g ∈ T.pi (fun _ => P),
        {x | ∀ z ∈ T.attach.sigma (fun r => (g r.1 r.2).2),
          x (g z.1.1 z.1.2).1 (n + z.1.1 * v + z.2) ∈ W (g z.1.1 z.1.2).1} := by
      intro x hx
      have hx' : ∀ r ∈ T, ∃ q ∈ P, ∀ i ∈ q.2, x q.1 (n + r * v + i) ∈ W q.1 :=
        fun r hr => Set.mem_iInter₂.1 hx r hr
      choose g hgP hg using hx'
      refine Set.mem_biUnion (Finset.mem_pi.2 (fun r hr => hgP r hr)) ?_
      intro z hz
      exact hg z.1.1 z.1.2 z.2 (Finset.mem_sigma.1 hz).2
    refine le_trans (measure_mono hcov) (le_trans (measure_biUnion_finset_le _ _) ?_)
    have heach : ∀ g ∈ T.pi (fun _ => P),
        ν {x | ∀ z ∈ T.attach.sigma (fun r => (g r.1 r.2).2),
          x (g z.1.1 z.1.2).1 (n + z.1.1 * v + z.2) ∈ W (g z.1.1 z.1.2).1}
          = ∏ r ∈ T.attach, D (g r.1 r.2).1 (W (g r.1 r.2).1) ^ (g r.1 r.2).2.card := by
      intro g hg
      have hgv : ∀ r (hr : r ∈ T), (g r hr).2 ⊆ Finset.range v := fun r hr =>
        Finset.mem_powersetCard.1 (Finset.mem_product.1 (Finset.mem_pi.1 hg r hr)).2 |>.1
      rw [certCoords_eq D (T.attach.sigma (fun r => (g r.1 r.2).2))
        (fun z => ((g z.1.1 z.1.2).1, n + z.1.1 * v + z.2)) ?_ W hW, Finset.prod_sigma]
      · refine Finset.prod_congr rfl (fun r _ => ?_)
        dsimp only
        exact Finset.prod_const (s := (g r.1 r.2).2) (D (g r.1 r.2).1 (W (g r.1 r.2).1))
      · intro z hz z' hz' he
        simp only [Prod.mk.injEq] at he
        have hi := Finset.mem_range.1 (hgv _ _ (Finset.mem_sigma.1 hz).2)
        have hi' := Finset.mem_range.1 (hgv _ _ (Finset.mem_sigma.1 hz').2)
        have hr : z.1.1 = z'.1.1 := block_index_eq hi hi' (by omega)
        have hr' : z.1 = z'.1 := Subtype.ext hr
        obtain ⟨⟨r, hrT⟩, i⟩ := z
        obtain ⟨⟨r', hrT'⟩, i'⟩ := z'
        simp only at hr hr' he hi hi'
        subst hr
        have : i = i' := by omega
        subst this
        rfl
    rw [Finset.sum_congr rfl heach,
      ← Finset.prod_sum (s := T) (t := fun _ => P) (f := fun _ q => D q.1 (W q.1) ^ q.2.card),
      Finset.prod_const, hTc]
    refine le_of_eq (congrArg (· ^ m) ?_)
    rw [hP, Finset.sum_product]
    refine Finset.sum_congr rfl (fun j _ => ?_)
    dsimp only
    rw [Finset.sum_congr rfl (g := fun _ => D j (W j) ^ (h + 1))
      (fun U hU => by rw [(Finset.mem_powersetCard.1 hU).2])]
    simp
  calc ν _ ≤ ν (⋃ T ∈ (Finset.range R).powersetCard m, ⋂ r ∈ T, H r) := measure_mono hsub
    _ ≤ ∑ T ∈ (Finset.range R).powersetCard m, ν (⋂ r ∈ T, H r) := measure_biUnion_finset_le _ _
    _ ≤ ∑ T ∈ (Finset.range R).powersetCard m,
          (∑ j ∈ Jv, (v.choose (h + 1) : ENNReal) * D j (W j) ^ (h + 1)) ^ m :=
        Finset.sum_le_sum hblock
    _ = _ := by rw [Finset.sum_const, Finset.card_powersetCard, Finset.card_range, nsmul_eq_mul]

end OrthoDFA
