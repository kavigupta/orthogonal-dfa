import RLStar.Proofs.DrawFrom
import RLStar.Proofs.RandomRisk
import RLStar.Proofs.RandomHarvest
import RLStar.Proofs.PadCanon

/-!
# A round's trie from fresh draws

A string harvested from a probe walked from `X` extends the probe's first prefix outside `X`, so
its atom is at most that prefix's mass (`lazyHarvestLaw_atom_le`). The round's trie holds every
prefix shorter than `K` of `Nh` fresh draws, so it misses a prefix of mass above `θ` with chance at
most `K/θ · (1 − θ)^Nh` (`heavy_miss_le`).
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory
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

/-- Some prefix shorter than `K` holding more than `θ` of `D` is a prefix of none of `Nh` draws
with chance at most `K/θ · (1 − θ)^Nh`. -/
theorem heavy_miss_le (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (K Nh : ℕ) {θ : ℝ}
    (hθ : 0 < θ) (hθ1 : θ ≤ 1) :
    (Measure.pi fun _ : Fin Nh => (volume : Measure unitInterval)).real
      {u | ∃ c : FreeMonoid α, c.toList.length < K
        ∧ θ < D.real {x | prefixOf x c.toList.length = c}
        ∧ c ∉ Random.prefixTrie K (heavySet D u)} ≤ K / θ * (1 - θ) ^ Nh := by
  classical
  set cyl : FreeMonoid α → Set (FreeMonoid α) := fun c => {x | prefixOf x c.toList.length = c}
    with hcyl
  set Hs := (Random.completeTrie K).filter fun c : FreeMonoid α => θ < D.real (cyl c) with hHs
  have hsub : {u : Fin Nh → unitInterval | ∃ c : FreeMonoid α, c.toList.length < K
        ∧ θ < D.real {x | prefixOf x c.toList.length = c}
        ∧ c ∉ Random.prefixTrie K (heavySet D u)}
      ⊆ ⋃ c ∈ Hs, Set.pi Set.univ fun _ => drawFrom D ⁻¹' (cyl c)ᶜ := by
    rintro u ⟨c, hcK, hθc, hc⟩
    refine Set.mem_biUnion (Finset.mem_coe.2
      (Finset.mem_filter.2 ⟨Random.mem_completeTrie.2 hcK, hθc⟩)) ?_
    intro i _ hi
    apply hc
    have hi' : prefixOf (drawFrom D (u i)) c.toList.length = c := hi
    have hl : (prefixOf (drawFrom D (u i)) c.toList.length).toList.length = c.toList.length := by
      rw [hi']
    rw [Random.length_prefixOf'] at hl
    unfold Random.prefixTrie heavySet
    refine Finset.mem_biUnion.2 ⟨drawFrom D (u i), Finset.mem_image_of_mem _ (Finset.mem_univ i),
      Finset.mem_image.2 ⟨c.toList.length, Finset.mem_range.2 ?_, hi'⟩⟩
    omega
  have hpt : ∀ c, (Measure.pi fun _ : Fin Nh => (volume : Measure unitInterval)).real
      (Set.pi Set.univ fun _ => drawFrom D ⁻¹' (cyl c)ᶜ) = (1 - D.real (cyl c)) ^ Nh := by
    intro c
    rw [measureReal_def, Measure.pi_pi, Finset.prod_const, Finset.card_univ, Fintype.card_fin,
      ENNReal.toReal_pow, ← Measure.map_apply (measurable_drawFrom D)
        (Set.to_countable _).measurableSet, map_drawFrom, ← measureReal_def,
      probReal_compl_eq_one_sub (Set.to_countable _).measurableSet]
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
  calc _ ≤ (Measure.pi fun _ : Fin Nh => (volume : Measure unitInterval)).real
        (⋃ c ∈ Hs, Set.pi Set.univ fun _ => drawFrom D ⁻¹' (cyl c)ᶜ) :=
          measureReal_mono hsub (measure_ne_top _ _)
    _ ≤ ∑ c ∈ Hs, (Measure.pi fun _ : Fin Nh => (volume : Measure unitInterval)).real
        (Set.pi Set.univ fun _ => drawFrom D ⁻¹' (cyl c)ᶜ) := measureReal_biUnion_finset_le Hs _
    _ = ∑ c ∈ Hs, (1 - D.real (cyl c)) ^ Nh := Finset.sum_congr rfl fun c _ => hpt c
    _ ≤ ∑ _c ∈ Hs, (1 - θ) ^ Nh := by
        refine Finset.sum_le_sum fun c hc => pow_le_pow_left₀ ?_ ?_ Nh
        · linarith [measureReal_le_one (μ := D) (s := cyl c)]
        · linarith [(Finset.mem_filter.1 hc).2]
    _ = Hs.card * (1 - θ) ^ Nh := by rw [Finset.sum_const, nsmul_eq_mul]
    _ ≤ K / θ * (1 - θ) ^ Nh :=
        mul_le_mul_of_nonneg_right hcard (pow_nonneg (by linarith) _)

theorem measurableSet_heavyMiss (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    (K Nh : ℕ) (θ : ℝ) :
    MeasurableSet {u : Fin Nh → unitInterval | ∃ c : FreeMonoid α, c.toList.length < K
        ∧ θ < D.real {x | prefixOf x c.toList.length = c}
        ∧ c ∉ Random.prefixTrie K (heavySet D u)} := by
  classical
  have e : {u : Fin Nh → unitInterval | ∃ c : FreeMonoid α, c.toList.length < K
        ∧ θ < D.real {x | prefixOf x c.toList.length = c}
        ∧ c ∉ Random.prefixTrie K (heavySet D u)}
      = (fun u : Fin Nh → unitInterval => fun i => drawFrom D (u i)) ⁻¹'
        {v : Fin Nh → FreeMonoid α | ∃ c : FreeMonoid α, c.toList.length < K
          ∧ θ < D.real {x | prefixOf x c.toList.length = c}
          ∧ c ∉ Random.prefixTrie K (Finset.univ.image v)} := rfl
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
