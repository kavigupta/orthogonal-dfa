import RLStar.Proofs.DrawFrom
import RLStar.Proofs.RandomRisk
import RLStar.Proofs.RandomHarvest
import RLStar.Proofs.PadCanon
import RLStar.Proofs.Hoeffding

/-!
# A round's trie from fresh draws

A string harvested from a probe walked from `X` extends the probe's first prefix outside `X`, so
its atom is at most that prefix's mass (`lazyHarvestLaw_atom_le`). The round's trie holds every
prefix shorter than `K` that at least `m` of `Nh` fresh draws share. Those of one length share
disjoint draws, so there are at most `K Nh / m` (`card_heavyOf_mul_le`), and at
`m ≤ θ Nh / 2 + 1` the trie misses a prefix of mass above `θ` with chance at most
`K/θ · exp(−θ Nh / 8)` (`heavy_miss_le`).
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory ProbabilityTheory
open Random (prefixMax)

variable {α : Type*} [Fintype α]

namespace Random

/-- A string harvested from `x ≥ K` long, padded from `X`, has its pads dropped `x`'s walk start
and its first prefix. -/
theorem walkStartIn_unpad_of_harvest {read : FreeMonoid α → ARU} {K : ℕ}
    {X : Finset (FreeMonoid α)} (hX : ∀ c ∈ X, c.toList.length < K)
    {H : HypothesisDFA (Option α)} {s : Level3Site} {x : FreeMonoid α}
    (hx : K ≤ x.toList.length) {w : FreeMonoid (Option α)}
    (hw : w ∈ siteHarvest (padRead read) K H s (padIn K X x)) :
    walkStartIn X (unpad w) = walkStartIn X x
      ∧ prefixOf (unpad w) (walkStartIn X x) = prefixOf x (walkStartIn X x) := by
  classical
  obtain ⟨hpot, -⟩ := siteHarvest_pot (padRead read) K H s (padIn K X x) w hw
  unfold pot at hpot
  obtain ⟨p, hp, hz⟩ := Finset.mem_biUnion.1 hpot
  obtain ⟨hKp, -⟩ := Finset.mem_Icc.1 hp
  obtain ⟨m, -, rfl⟩ := Finset.mem_image.1 hz
  have hnK := walkStartIn_le_of_short hX x
  rw [unpad_mul, unpad_prefixOf_padIn]
  exact walkStartIn_prefixOf_mul hX hx (by omega) _

end Random

theorem real_cyl_le_prefixMax (D : Measure (FreeMonoid α)) [IsFiniteMeasure D] (k : ℕ)
    (u : FreeMonoid α) : D.real {x | prefixOf x k = u} ≤ prefixMax D k :=
  le_ciSup (f := fun u => D.real {x | prefixOf x k = u}) ⟨D.real Set.univ, by
    rintro _ ⟨v, rfl⟩; exact measureReal_mono (Set.subset_univ _)⟩ u

/-- Where every prefix shorter than `K` and outside `X` holds at most `θ` of `D`, and `D`'s strings
are at least `K` long, no string's atom in the harvest law passes `max θ (prefixMax D K)`. -/
theorem lazyHarvestLaw_atom_le (read : FreeMonoid α → ARU) (D : Measure (FreeMonoid α))
    [IsFiniteMeasure D] (K : ℕ) (X : Finset (FreeMonoid α)) (H : HypothesisDFA (Option α))
    (s : Random.Level3Site) (θ : ℝ) (hX : ∀ c ∈ X, c.toList.length < K)
    (hlen : ∀ᵐ x ∂D, K ≤ x.toList.length)
    (hlight : ∀ c : FreeMonoid α, c.toList.length < K → c ∉ X →
      D.real {x | prefixOf x c.toList.length = c} ≤ θ) (t : FreeMonoid α) :
    (Random.lazyHarvestLaw read D K X H s).real {t} ≤ max θ (prefixMax D K) := by
  classical
  set c := prefixOf t (Random.walkStartIn X t) with hc
  set S : Set (FreeMonoid α) :=
    {x | prefixOf x c.toList.length = c ∧ c ∉ X ∧ c.toList.length ≤ K} with hS
  have h1 : Random.lazyHarvestLaw read D K X H s {t} ≤ D S := by
    unfold Random.lazyHarvestLaw
    rw [Measure.map_apply (measurable_of_countable _) (measurableSet_singleton t),
      Random.harvestLaw_apply, lintegral_map (measurable_of_countable _)
        (measurable_of_countable _), ← lintegral_indicator_one (Set.to_countable S).measurableSet]
    refine lintegral_mono_ae (hlen.mono fun x hx => ?_)
    set l := Random.siteHarvest (Random.padRead read) K H s (Random.padIn K X x)
    by_cases hxS : x ∈ S
    · rw [Set.indicator_of_mem hxS, Pi.one_apply]
      exact Nat.cast_le_one.2
        ((List.length_filter_le _ _).trans (Random.siteHarvest_length _ K H s _))
    · rw [Set.indicator_of_notMem hxS]
      have : l.filter (fun w => decide (w ∈ Random.unpad ⁻¹' {t})) = [] := by
        refine List.filter_eq_nil_iff.2 fun w hw hwt => hxS ?_
        have hwt' : Random.unpad w = t := by simpa using hwt
        obtain ⟨hws, hpre⟩ := Random.walkStartIn_unpad_of_harvest hX hx hw
        rw [hwt'] at hws hpre
        have hcx : c = prefixOf x (Random.walkStartIn X x) := by rw [hc, hws, hpre]
        have hlc : c.toList.length = Random.walkStartIn X x := by
          rw [hcx, Random.length_prefixOf']
          exact min_eq_left (Random.walkStartIn_le X x)
        refine ⟨by rw [hlc, hcx], hcx ▸ Random.prefixOf_walkStartIn_notMem hX hx, ?_⟩
        rw [hlc]
        exact Random.walkStartIn_le_of_short hX x
      rw [this]
      simp
  have h2 : (Random.lazyHarvestLaw read D K X H s).real {t} ≤ D.real S :=
    ENNReal.toReal_mono (measure_ne_top D S) h1
  exact h2.trans (Random.real_walkCell_le D hlight (real_cyl_le_prefixMax D K) c)

omit [Fintype α] in
private lemma prefixOf_prefixOf_le (x : FreeMonoid α) {j n : ℕ} (h : j ≤ n) :
    prefixOf (prefixOf x n) j = prefixOf x j := by
  simp [prefixOf, List.take_take, min_eq_left h]

theorem mem_prefixTrie_iff {K : ℕ} {P : Finset (FreeMonoid α)} {c : FreeMonoid α} :
    c ∈ Random.prefixTrie K P ↔ c.toList.length < K ∧ ∃ p ∈ P, prefixOf p c.toList.length = c := by
  unfold Random.prefixTrie
  rw [Finset.mem_biUnion]
  constructor
  · rintro ⟨p, hp, hc⟩
    obtain ⟨i, hi, rfl⟩ := Finset.mem_image.1 hc
    rw [Finset.mem_range, lt_min_iff] at hi
    have hl : (prefixOf p i).toList.length = i := by rw [Random.length_prefixOf']; omega
    rw [hl]
    exact ⟨hi.1, p, hp, rfl⟩
  · rintro ⟨hcK, p, hp, hpc⟩
    have hl := congrArg (fun z : FreeMonoid α => z.toList.length) hpc
    simp only [Random.length_prefixOf'] at hl
    exact ⟨p, hp, Finset.mem_image.2 ⟨c.toList.length,
      Finset.mem_range.2 (lt_min_iff.2 ⟨hcK, by omega⟩), hpc⟩⟩

theorem mem_heavyOf {m K Nh : ℕ} {v : Fin Nh → FreeMonoid α} {c : FreeMonoid α} :
    c ∈ heavyOf m K v ↔ (c.toList.length < K ∧ ∃ i, prefixOf (v i) c.toList.length = c)
      ∧ m ≤ (Finset.univ.filter fun i => prefixOf (v i) c.toList.length = c).card := by
  unfold heavyOf
  rw [Finset.mem_filter, mem_prefixTrie_iff]
  constructor
  · rintro ⟨⟨hK, p, hp, hpc⟩, hm⟩
    obtain ⟨i, -, rfl⟩ := Finset.mem_image.1 hp
    exact ⟨⟨hK, i, hpc⟩, hm⟩
  · rintro ⟨⟨hK, i, hi⟩, hm⟩
    exact ⟨⟨hK, v i, Finset.mem_image_of_mem v (Finset.mem_univ i), hi⟩, hm⟩

theorem prefixOf_mem_heavyOf {m K Nh : ℕ} {v : Fin Nh → FreeMonoid α} {c : FreeMonoid α}
    (hc : c ∈ heavyOf m K v) {j : ℕ} (hj : j ≤ c.toList.length) :
    prefixOf c j ∈ heavyOf m K v := by
  obtain ⟨⟨hK, i, hi⟩, hm⟩ := mem_heavyOf.1 hc
  have hl : (prefixOf c j).toList.length = j := by rw [Random.length_prefixOf']; omega
  have hpre : ∀ i, prefixOf (v i) c.toList.length = c → prefixOf (v i) j = prefixOf c j :=
    fun i h => by rw [← h, prefixOf_prefixOf_le _ hj]
  refine mem_heavyOf.2 ⟨⟨by omega, i, by rw [hl]; exact hpre i hi⟩,
    hm.trans (Finset.card_le_card fun i' hi' => ?_)⟩
  rw [Finset.mem_filter] at hi' ⊢
  exact ⟨hi'.1, by rw [hl]; exact hpre i' hi'.2⟩

theorem prefixTrie_heavyOf_subset {m K Nh : ℕ} (v : Fin Nh → FreeMonoid α) :
    Random.prefixTrie K (heavyOf m K v) ⊆ heavyOf m K v := by
  intro c hc
  unfold Random.prefixTrie at hc
  obtain ⟨p, hp, hc⟩ := Finset.mem_biUnion.1 hc
  obtain ⟨j, hj, rfl⟩ := Finset.mem_image.1 hc
  rw [Finset.mem_range, lt_min_iff] at hj
  exact prefixOf_mem_heavyOf hp (by omega)

theorem prefixTrie_union_heavyOf_subset {m K Nh : ℕ} (P : Finset (FreeMonoid α))
    (v : Fin Nh → FreeMonoid α) :
    Random.prefixTrie K (P ∪ heavyOf m K v) ⊆ Random.prefixTrie K P ∪ heavyOf m K v := by
  intro c hc
  unfold Random.prefixTrie at hc
  rw [Finset.union_biUnion, Finset.mem_union] at hc
  rcases hc with hc | hc
  · exact Finset.mem_union_left _ hc
  · exact Finset.mem_union_right _ (prefixTrie_heavyOf_subset v hc)

/-- At each length below `K` the members have disjoint sets of at least `m` of the strings. -/
theorem card_heavyOf_mul_le (m K Nh : ℕ) (v : Fin Nh → FreeMonoid α) :
    (heavyOf m K v).card * m ≤ K * Nh := by
  set H := heavyOf m K v
  have hlen : Set.MapsTo (fun c : FreeMonoid α => c.toList.length) (H : Set (FreeMonoid α))
      (Finset.range K) := fun c hc =>
    Finset.mem_range.2 (mem_heavyOf.1 (Finset.mem_coe.1 hc)).1.1
  rw [Finset.card_eq_sum_card_fiberwise hlen, Finset.sum_mul]
  calc ∑ l ∈ Finset.range K, (H.filter fun c => c.toList.length = l).card * m
      ≤ ∑ _l ∈ Finset.range K, Nh := Finset.sum_le_sum fun l _ => ?_
    _ = K * Nh := by rw [Finset.sum_const, Finset.card_range, smul_eq_mul]
  set Hl := H.filter fun c => c.toList.length = l
  set s := Finset.univ.filter fun i => prefixOf (v i) l ∈ Hl
  have h := Finset.mul_card_image_le_card_of_maps_to (s := s) (t := Hl)
    (f := fun i => prefixOf (v i) l) (fun i hi => (Finset.mem_filter.1 hi).2) m fun c hc => ?_
  · rw [mul_comm]
    exact h.trans ((Finset.card_le_univ s).trans (by rw [Fintype.card_fin]))
  · obtain ⟨hcH, hcl⟩ := Finset.mem_filter.1 hc
    refine (mem_heavyOf.1 hcH).2.trans (Finset.card_le_card fun i hi => ?_)
    rw [Finset.mem_filter] at hi
    rw [hcl] at hi
    exact Finset.mem_filter.2 ⟨Finset.mem_filter.2 ⟨Finset.mem_univ _, hi.2 ▸ hc⟩, hi.2⟩

theorem card_heavyOf_le {m K Nh : ℕ} (v : Fin Nh → FreeMonoid α) {θ : ℝ} (hθ : 0 < θ)
    (hm : θ * Nh / 2 ≤ m) : ((heavyOf m K v).card : ℝ) ≤ 2 * K / θ := by
  rcases Nat.eq_zero_or_pos Nh with rfl | hN
  · have he : heavyOf m K v = ∅ := Finset.eq_empty_of_forall_notMem fun c hc => by
      obtain ⟨⟨-, i, -⟩, -⟩ := mem_heavyOf.1 hc
      exact i.elim0
    rw [he, Finset.card_empty, Nat.cast_zero]
    positivity
  have hN' : (0 : ℝ) < Nh := by exact_mod_cast hN
  have h : ((heavyOf m K v).card : ℝ) * m ≤ K * Nh := by
    exact_mod_cast card_heavyOf_mul_le m K Nh v
  have h2 : ((heavyOf m K v).card : ℝ) * (θ * Nh / 2) ≤ (heavyOf m K v).card * m :=
    mul_le_mul_of_nonneg_left hm (Nat.cast_nonneg _)
  have h3 : (heavyOf m K v).card * θ * Nh ≤ 2 * K * Nh := by linarith
  rw [le_div_iff₀ hθ]
  exact le_of_mul_le_mul_right h3 hN'

/-- Some prefix shorter than `K` holding more than `θ` of `D` is shared by fewer than `m` of `Nh`
draws, `m ≤ θ Nh / 2 + 1`, with chance at most `K/θ · exp(−θ Nh / 8)`. -/
theorem heavy_miss_le (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (m K Nh : ℕ) {θ : ℝ}
    (hθ : 0 < θ) (hm : (m : ℝ) ≤ θ * Nh / 2 + 1) :
    (Measure.pi fun _ : Fin Nh => (volume : Measure unitInterval)).real
      {u | ∃ c : FreeMonoid α, c.toList.length < K
        ∧ θ < D.real {x | prefixOf x c.toList.length = c}
        ∧ c ∉ Random.prefixTrie K (heavySet D m K u)} ≤ K / θ * Real.exp (-(θ * Nh / 8)) := by
  classical
  set ρ := Measure.pi fun _ : Fin Nh => (volume : Measure unitInterval)
  set cyl : FreeMonoid α → Set (FreeMonoid α) := fun c => {x | prefixOf x c.toList.length = c}
    with hcyl
  set Hs := (Random.completeTrie K).filter fun c : FreeMonoid α => θ < D.real (cyl c) with hHs
  set X : FreeMonoid α → Fin Nh → (Fin Nh → unitInterval) → ℝ :=
    fun c i u => (drawFrom D ⁻¹' cyl c).indicator 1 (u i) with hX
  set E : FreeMonoid α → Set (Fin Nh → unitInterval) := fun c =>
    {u | ∑ i ∈ Finset.univ, X c i u ≤ D.real (cyl c) * Nh - D.real (cyl c) * Nh / 2}
  have hcount : ∀ c u, ∑ i ∈ Finset.univ, X c i u
      = ((Finset.univ.filter fun i => prefixOf (drawFrom D (u i)) c.toList.length = c).card
        : ℝ) := by
    intro c u
    rw [Finset.natCast_card_filter]
    refine Finset.sum_congr rfl fun i _ => ?_
    by_cases h : prefixOf (drawFrom D (u i)) c.toList.length = c
    · simp [hX, hcyl, h]
    · simp [hX, hcyl, h]
  have hsub : {u : Fin Nh → unitInterval | ∃ c : FreeMonoid α, c.toList.length < K
        ∧ θ < D.real {x | prefixOf x c.toList.length = c}
        ∧ c ∉ Random.prefixTrie K (heavySet D m K u)} ⊆ ⋃ c ∈ Hs, E c := by
    rintro u ⟨c, hcK, hθc, hc⟩
    refine Set.mem_biUnion (Finset.mem_coe.2
      (Finset.mem_filter.2 ⟨Random.mem_completeTrie.2 hcK, hθc⟩)) ?_
    change ∑ i ∈ Finset.univ, X c i u ≤ _
    rw [hcount]
    rw [heavySet_eq_heavyOf] at hc
    have hcH : c ∉ heavyOf m K fun i => drawFrom D (u i) := fun h =>
      hc (mem_prefixTrie_iff.2 ⟨hcK, c, h, by simp [prefixOf]⟩)
    have hn : ((Finset.univ.filter fun i => prefixOf (drawFrom D (u i)) c.toList.length = c).card
        : ℝ) ≤ θ * Nh / 2 := by
      rcases Nat.eq_zero_or_pos
        (Finset.univ.filter fun i => prefixOf (drawFrom D (u i)) c.toList.length = c).card
        with h0 | hpos
      · rw [h0, Nat.cast_zero]
        positivity
      obtain ⟨i, hi⟩ := Finset.card_pos.1 hpos
      have hlt : (Finset.univ.filter fun i => prefixOf (drawFrom D (u i)) c.toList.length = c).card
          < m := by
        by_contra hge
        exact hcH (mem_heavyOf.2 ⟨⟨hcK, i, (Finset.mem_filter.1 hi).2⟩, not_lt.1 hge⟩)
      have : ((Finset.univ.filter fun i => prefixOf (drawFrom D (u i)) c.toList.length = c).card
          : ℝ) + 1 ≤ m := by exact_mod_cast hlt
      linarith
    have hp : θ * Nh ≤ D.real (cyl c) * Nh :=
      mul_le_mul_of_nonneg_right hθc.le (Nat.cast_nonneg _)
    linarith
  have hEc : ∀ c ∈ Hs, ρ.real (E c) ≤ Real.exp (-(θ * Nh / 8)) := by
    intro c hc
    have hθc : θ < D.real (cyl c) := (Finset.mem_filter.1 hc).2
    rcases Nat.eq_zero_or_pos Nh with h0 | hN
    · have hN0 : (Nh : ℝ) = 0 := by exact_mod_cast h0
      rw [hN0, mul_zero, zero_div, neg_zero, Real.exp_zero]
      exact measureReal_le_one
    have hcm : MeasurableSet (drawFrom D ⁻¹' cyl c) :=
      measurable_drawFrom D (Set.to_countable _).measurableSet
    have hmeas : ∀ i, Measurable (X c i) := fun i =>
      (measurable_const.indicator hcm).comp (measurable_pi_apply i)
    have hind : iIndepFun (X c) ρ :=
      iIndepFun_pi (X := fun _ (x : unitInterval) => (drawFrom D ⁻¹' cyl c).indicator
        (1 : unitInterval → ℝ) x) fun _ => (measurable_const.indicator hcm).aemeasurable
    have hIcc : ∀ i, ∀ᵐ u ∂ρ, X c i u ∈ Set.Icc (0 : ℝ) 1 := fun i => ae_of_all _ fun u => by
      simp only [hX, Set.indicator_apply, Pi.one_apply]
      split_ifs <;> norm_num
    have hmean : ∀ i, ρ[X c i] = D.real (cyl c) := by
      intro i
      have e : X c i
          = ((fun u : Fin Nh → unitInterval => u i) ⁻¹' (drawFrom D ⁻¹' cyl c)).indicator 1 := by
        ext u
        simp only [hX, Set.indicator_apply, Set.mem_preimage, Pi.one_apply]
      rw [e, integral_indicator_one ((measurable_pi_apply i) hcm), measureReal_def,
        ← Measure.map_apply (measurable_pi_apply i) hcm, (measurePreserving_eval _ i).map_eq,
        ← Measure.map_apply (measurable_drawFrom D) (Set.to_countable _).measurableSet,
        map_drawFrom, ← measureReal_def]
    have hν : 0 < D.real (cyl c) * Nh := mul_pos (hθ.trans hθc) (by exact_mod_cast hN)
    have hsum : D.real (cyl c) * Nh ≤ ∑ i ∈ Finset.univ, ρ[X c i] := by
      rw [Finset.sum_congr rfl fun i _ => hmean i, Finset.sum_const, Finset.card_univ,
        Fintype.card_fin, nsmul_eq_mul, mul_comm]
    have h := sumLower_rel (μ := ρ) (X c) Finset.univ (D.real (cyl c) * Nh)
      (D.real (cyl c) * Nh / 2) hmeas hind hIcc hν hsum (by positivity)
    refine h.trans (Real.exp_le_exp.2 ?_)
    have e : (D.real (cyl c) * Nh / 2) ^ 2 / (2 * (D.real (cyl c) * Nh))
        = D.real (cyl c) * Nh / 8 := by
      field_simp
      ring
    rw [e]
    have : θ * Nh ≤ D.real (cyl c) * Nh := mul_le_mul_of_nonneg_right hθc.le (Nat.cast_nonneg _)
    linarith
  have hsum : ∑ c ∈ Hs, D.real (cyl c) ≤ K := by
    rw [← Finset.sum_fiberwise_of_maps_to (s := Hs) (t := Finset.range K)
      (g := fun c : FreeMonoid α => c.toList.length) fun c hc =>
        Finset.mem_range.2 (Random.mem_completeTrie.1 (Finset.mem_filter.1 hc).1)]
    calc ∑ l ∈ Finset.range K, ∑ c ∈ Hs with c.toList.length = l, D.real (cyl c)
        ≤ ∑ _l ∈ Finset.range K, (1 : ℝ) := by
          refine Finset.sum_le_sum fun l _ => ?_
          rw [← measureReal_biUnion_finset (fun c hc c' hc' hne => ?_)
            fun c _ => (Set.to_countable _).measurableSet]
          · exact measureReal_le_one
          · rw [Finset.coe_filter] at hc hc'
            refine Set.disjoint_left.2 fun x hx hx' => hne ?_
            have h1 : prefixOf x c.toList.length = c := hx
            have h2 : prefixOf x c'.toList.length = c' := hx'
            rw [← h1, ← h2, hc.2, hc'.2]
      _ = K := by simp
  have hcard : (Hs.card : ℝ) ≤ K / θ := by
    rw [le_div_iff₀ hθ]
    calc (Hs.card : ℝ) * θ = ∑ _c ∈ Hs, θ := by rw [Finset.sum_const, nsmul_eq_mul]
      _ ≤ ∑ c ∈ Hs, D.real (cyl c) :=
          Finset.sum_le_sum fun c hc => (Finset.mem_filter.1 hc).2.le
      _ ≤ K := hsum
  calc _ ≤ ρ.real (⋃ c ∈ Hs, E c) := measureReal_mono hsub (measure_ne_top _ _)
    _ ≤ ∑ c ∈ Hs, ρ.real (E c) := measureReal_biUnion_finset_le Hs _
    _ ≤ ∑ _c ∈ Hs, Real.exp (-(θ * Nh / 8)) := Finset.sum_le_sum hEc
    _ = Hs.card * Real.exp (-(θ * Nh / 8)) := by rw [Finset.sum_const, nsmul_eq_mul]
    _ ≤ K / θ * Real.exp (-(θ * Nh / 8)) :=
        mul_le_mul_of_nonneg_right hcard (Real.exp_pos _).le

theorem measurableSet_heavyMiss (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    (m K Nh : ℕ) (θ : ℝ) :
    MeasurableSet {u : Fin Nh → unitInterval | ∃ c : FreeMonoid α, c.toList.length < K
        ∧ θ < D.real {x | prefixOf x c.toList.length = c}
        ∧ c ∉ Random.prefixTrie K (heavySet D m K u)} := by
  classical
  have e : {u : Fin Nh → unitInterval | ∃ c : FreeMonoid α, c.toList.length < K
        ∧ θ < D.real {x | prefixOf x c.toList.length = c}
        ∧ c ∉ Random.prefixTrie K (heavySet D m K u)}
      = (fun u : Fin Nh → unitInterval => fun i => drawFrom D (u i)) ⁻¹'
        {v : Fin Nh → FreeMonoid α | ∃ c : FreeMonoid α, c.toList.length < K
          ∧ θ < D.real {x | prefixOf x c.toList.length = c}
          ∧ c ∉ Random.prefixTrie K (heavyOf m K v)} := rfl
  rw [e]
  exact (measurable_pi_lambda _ fun i => (measurable_drawFrom D).comp (measurable_pi_apply i))
    (Set.to_countable _).measurableSet

theorem prefixTrie_mono {K : ℕ} {P P' : Finset (FreeMonoid α)} (h : P ⊆ P') :
    Random.prefixTrie K P ⊆ Random.prefixTrie K P' :=
  Finset.biUnion_subset_biUnion_of_subset_left _ h

theorem card_prefixTrie_le (K : ℕ) (P : Finset (FreeMonoid α)) :
    (Random.prefixTrie K P).card ≤ K * P.card := by
  unfold Random.prefixTrie
  refine Finset.card_biUnion_le.trans ?_
  calc ∑ p ∈ P, ((Finset.range (min K (p.toList.length + 1))).image fun i => prefixOf p i).card
      ≤ ∑ _p ∈ P, K := Finset.sum_le_sum fun p _ => Finset.card_image_le.trans (by simp)
    _ = K * P.card := by rw [Finset.sum_const, smul_eq_mul, mul_comm]

end OrthoDFA

end
