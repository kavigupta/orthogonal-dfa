import OrthoDFA.Proofs.Level5Firsts

/-!
# A harvest's sampler

Every string a probe harvests extends the probe's length-`k` prefix, and a probe harvests no
string twice, so the sampler finds a string at most as often as draws share its length-`k` prefix
(`harvestLaw_spread`). Over draws of bounded length its mass on a set is the mean count of a
probe's harvested strings in it (`harvestLaw_real`), and `SiteGood` is the claim's harvest case
(`endsWell_of_siteGood`).
-/

namespace OrthoDFA

open MeasureTheory
open scoped ENNReal

variable {α : Type*} [Fintype α] [DecidableEq α]

section Fresh

variable (cut : FreeMonoid α → Option Bool) (T : DTree α) (E : Edges α) (k : ℕ)
  (x : FreeMonoid α)

theorem middleHarvestBy_cases :
    middleHarvestBy cut T E k x = [] ∨ ∃ b, middleHarvestBy cut T E k x = [b] := by
  unfold middleHarvestBy
  split
  · rename_i zs h
    obtain ⟨-, b, -, -, -, rfl⟩ := probe_middle _ T E k x h
    exact .inr ⟨b, rfl⟩
  · exact .inl rfl

theorem startHarvestBy_fresh : (startHarvestBy cut T k x).Nodup
    ∧ ∀ z ∈ startHarvestBy cut T k x, prefixOf z k = prefixOf x k := by
  unfold startHarvestBy
  split_ifs with hkx
  · simp
  · push Not at hkx
    rcases hs : T.sift (ofCut cut) (prefixOf x k) with p | b
    · simp
    · simp only [Sum.elim_inr, List.nodup_singleton, List.mem_singleton, forall_eq, true_and]
      exact probeStrings_prefix (route_mem_probeStrings hkx le_rfl (sift_inr_reads cut T _ b hs).1)

theorem edgeHarvestBy_fresh (e : List Bool × α) : (edgeHarvestBy cut T E k x e).Nodup
    ∧ ∀ z ∈ edgeHarvestBy cut T E k x e, prefixOf z k = prefixOf x k := by
  by_cases hkx : x.toList.length < k
  · simp [edgeHarvestBy_short cut T E hkx]
  push Not at hkx
  refine ⟨(List.Sublist.map Prod.snd List.filter_sublist).nodup (firstsBy_nodup cut T E k x), ?_⟩
  intro z hz
  unfold edgeHarvestBy at hz
  obtain ⟨q, hq, rfl⟩ := List.mem_map.1 hz
  exact probeStrings_prefix (firstsBy_mem_probeStrings cut T E k x hkx (List.mem_of_mem_filter hq))

theorem middleHarvestBy_fresh : (middleHarvestBy cut T E k x).Nodup
    ∧ ∀ z ∈ middleHarvestBy cut T E k x, prefixOf z k = prefixOf x k := by
  by_cases hkx : x.toList.length < k
  · simp [middleHarvestBy_short cut T E hkx]
  push Not at hkx
  have key : ∀ j b, k ≤ j → T.sift (ofCut cut) (prefixOf x j) = .inr b →
      prefixOf b k = prefixOf x k := fun j b hj hb =>
    probeStrings_prefix (route_mem_probeStrings hkx hj (sift_inr_reads cut T _ b hb).1)
  unfold middleHarvestBy
  split
  · rename_i zs h
    obtain ⟨j, b, hj, -, hs, rfl⟩ := probe_middle _ T E k x h
    simpa using key j b hj.le hs
  · simp

theorem siteHarvest_fresh (site : Level5Site α) (v : DTree α × Edges α) :
    (siteHarvest cut k site v x).Nodup
      ∧ ∀ z ∈ siteHarvest cut k site v x, prefixOf z k = prefixOf x k := by
  cases site with
  | start => exact startHarvestBy_fresh cut v.1 k x
  | edge e => exact edgeHarvestBy_fresh cut v.1 v.2 k x e
  | middles => exact middleHarvestBy_fresh cut v.1 v.2 k x

theorem siteHarvest_length_le (site : Level5Site α) (v : DTree α × Edges α) :
    (siteHarvest cut k site v x).length ≤ x.toList.length + 1 := by
  cases site with
  | start =>
    show (startHarvestBy cut v.1 k x).length ≤ _
    unfold startHarvestBy
    split_ifs
    · simp
    · rcases v.1.sift (ofCut cut) (prefixOf x k) with _ | _ <;> simp
  | edge e => exact edgeHarvestBy_length_le cut v.1 v.2 k x e
  | middles =>
    show (middleHarvestBy cut v.1 v.2 k x).length ≤ _
    rcases middleHarvestBy_cases cut v.1 v.2 k x with h | ⟨b, h⟩ <;> rw [h] <;> simp

end Fresh

section Law

variable (D : Measure (FreeMonoid α)) (cut : FreeMonoid α → Option Bool) (k : ℕ)

theorem list_dirac_apply (l : List (FreeMonoid α)) (A : Set (FreeMonoid α))
    [DecidablePred (· ∈ A)] :
    (l.map Measure.dirac).sum A = ((l.filter (· ∈ A)).length : ℝ≥0∞) := by
  induction l with
  | nil => simp
  | cons z l ih =>
    rw [List.map_cons, List.sum_cons, Measure.add_apply, ih,
      Measure.dirac_apply' _ (measurableSet_of_countable A)]
    by_cases hz : z ∈ A
    · simp [hz, add_comm]
    · simp [hz]

theorem harvestLaw_apply (H : HypothesisDFA α) (site : Level5Site α) (A : Set (FreeMonoid α))
    [DecidablePred (· ∈ A)] :
    harvestLaw D cut k H site A = ∫⁻ x,
      (((siteHarvest cut k site (H.tree, H.edges) x).filter (· ∈ A)).length
        : ℝ≥0∞) ∂D := by
  unfold harvestLaw
  rw [Measure.bind_apply (measurableSet_of_countable A) (measurable_of_countable _).aemeasurable]
  simp only [list_dirac_apply]

/-- The sampler finds a string at most as often as draws share its length-`k` prefix. -/
theorem harvestLaw_spread [IsFiniteMeasure D] (H : HypothesisDFA α) (site : Level5Site α) :
    Spread D k (harvestLaw D cut k H site) := by
  intro t
  set v := (H.tree, H.edges)
  set B := {p : FreeMonoid α | p.toList.take k = t.toList.take k}
  have hle : ∫⁻ x, (((siteHarvest cut k site v x).filter (· ∈ ({t} : Set _))).length
      : ℝ≥0∞) ∂D ≤ D B := by
    rw [← lintegral_indicator_one (measurableSet_of_countable B)]
    refine lintegral_mono fun x => ?_
    have hf := siteHarvest_fresh cut k x site v
    by_cases ht : t ∈ siteHarvest cut k site v x
    · have hlen : ((siteHarvest cut k site v x).filter (· ∈ ({t} : Set _))).length ≤ 1 := by
        have hnd := hf.1.filter (fun a => decide (a ∈ ({t} : Set (FreeMonoid α))))
        have hsub : ((siteHarvest cut k site v x).filter
            (· ∈ ({t} : Set _))).toFinset ⊆ {t} := by
          intro a ha
          simp only [List.mem_toFinset, List.mem_filter, Set.mem_singleton_iff,
            decide_eq_true_eq] at ha
          simp [ha.2]
        rw [← List.toFinset_card_of_nodup hnd]
        exact (Finset.card_le_card hsub).trans (by simp)
      have hx : x ∈ B := by
        have h' := congrArg FreeMonoid.toList (hf.2 t ht)
        simpa [prefixOf, B] using h'.symm
      rw [Set.indicator_of_mem hx, Pi.one_apply]
      calc _ ≤ ((1 : ℕ) : ℝ≥0∞) := Nat.cast_le.2 hlen
        _ = 1 := Nat.cast_one
    · have : (siteHarvest cut k site v x).filter (· ∈ ({t} : Set _)) = [] := by
        refine List.filter_eq_nil_iff.2 fun a ha => ?_
        simp only [Set.mem_singleton_iff, decide_eq_true_eq]
        rintro rfl
        exact ht ha
      rw [this]
      simp
  have hlaw : harvestLaw D cut k H site {t} ≤ D B := by
    rw [harvestLaw_apply]
    exact hle
  exact ENNReal.toReal_mono (measure_ne_top D B) hlaw

theorem integrable_siteCount [IsFiniteMeasure D] {L : ℕ} (hlen : ∀ᵐ x ∂D, x.toList.length ≤ L)
    (site : Level5Site α) (v : DTree α × Edges α) (P : FreeMonoid α → Bool) :
    Integrable (fun x => (((siteHarvest cut k site v x).filter P).length : ℝ)) D :=
  Integrable.of_bound (measurable_of_countable _).aestronglyMeasurable (L + 1) (by
    filter_upwards [hlen] with x hx
    rw [Real.norm_eq_abs, abs_of_nonneg (Nat.cast_nonneg _)]
    have h1 := List.length_filter_le P (siteHarvest cut k site v x)
    have h2 := siteHarvest_length_le cut k x site v
    exact_mod_cast (by omega : ((siteHarvest cut k site v x).filter P).length ≤ L + 1))

/-- Over draws of bounded length, the sampler's mass on `A` is a probe's mean count of harvested
strings in `A`. -/
theorem harvestLaw_real [IsFiniteMeasure D] {L : ℕ} (hlen : ∀ᵐ x ∂D, x.toList.length ≤ L)
    (H : HypothesisDFA α) (site : Level5Site α) (A : Set (FreeMonoid α)) [DecidablePred (· ∈ A)] :
    (harvestLaw D cut k H site).real A = ∫ x,
      (((siteHarvest cut k site (H.tree, H.edges) x).filter (· ∈ A)).length : ℝ)
        ∂D := by
  rw [measureReal_def, harvestLaw_apply]
  have := ofReal_integral_eq_lintegral_ofReal (integrable_siteCount D cut k hlen site
    (H.tree, H.edges) (fun x => decide (x ∈ A)))
    (ae_of_all _ fun _ => Nat.cast_nonneg _)
  simp only [ENNReal.ofReal_natCast] at this
  rw [← this, ENNReal.toReal_ofReal (integral_nonneg fun _ => Nat.cast_nonneg _)]

/-- `SiteGood` at the final hypothesis makes the harvest end well, its sampler spread from `k`. -/
theorem endsWell_of_siteGood {σ : Type*} (G : ReadModel α σ) [IsProbabilityMeasure D] {L : ℕ}
    (hlen : ∀ᵐ x ∂D, x.toList.length ≤ L) (rd : FreeMonoid α → ARU) (ε : ℝ)
    (dis : Set (FreeMonoid α)) (y : ℝ) (H : HypothesisDFA α) (site : Level5Site α)
    (hm : SiteGood G D rd k y site (H.tree, H.edges)) :
    EndsWell G.M (fun q => ¬ G.Good q) D ε dis k (harvestLaw D (fun z => (rd z).cut) k H) y
      (.harvest site) := by
  classical
  set v := (H.tree, H.edges)
  have hsplit : siteTotal D rd k site v
      = siteMean G D rd k site G.Good v + siteMean G D rd k site (fun q => ¬ G.Good q) v := by
    unfold siteMean siteTotal
    rw [← integral_add (integrable_siteCount D _ k hlen site v _)
      (integrable_siteCount D _ k hlen site v _)]
    refine integral_congr_ae (ae_of_all _ fun x => ?_)
    have key : ∀ (l : List (FreeMonoid α)) (p q : FreeMonoid α → Bool), (∀ b, q b = !p b) →
        l.length = (l.filter p).length + (l.filter q).length := by
      intro l p q hq
      rw [List.length_eq_length_filter_add (l := l) p]
      congr 2
      exact List.filter_congr fun b _ => (hq b).symm
    dsimp only
    norm_cast
    apply key
    intro b
    exact Bool.eq_iff_iff.2 (by simp)
  have huniv : (harvestLaw D (fun z => (rd z).cut) k H site).real Set.univ
      = siteTotal D rd k site v := by
    rw [harvestLaw_real D _ k hlen]
    unfold siteTotal
    congr 1
    funext x
    rw [List.filter_eq_self.2 (by simp)]
  have hbad : (harvestLaw D (fun z => (rd z).cut) k H site).real
      {z | ¬ G.Good (G.M.eval z.toList)} = siteMean G D rd k site (fun q => ¬ G.Good q) v := by
    rw [harvestLaw_real D _ k hlen]
    unfold siteMean
    congr 1
    funext x
    congr 2
    exact List.filter_congr fun b _ => Bool.eq_iff_iff.2 (by simp)
  refine ⟨?_, harvestLaw_spread D _ k H site, ?_⟩
  · rw [huniv, hbad]
    linarith [hm.1]
  · show y ≤ (harvestLaw D (fun z => (rd z).cut) k H site).real Set.univ
    rw [huniv]
    exact hm.2

omit [DecidableEq α] in
/-- An ending that is well at a yield is well at any smaller one. -/
theorem EndsWell.mono_yield {σ H : Type*} {M : _root_.DFA α σ} {bad : σ → Prop}
    {D : Measure (FreeMonoid α)} {ε : ℝ} {dis : Set (FreeMonoid α)} {k : ℕ}
    {law : H → Measure (FreeMonoid α)} {y y' : ℝ} (hy : y ≤ y') {e : RoundEnd H}
    (h : EndsWell M bad D ε dis k law y' e) : EndsWell M bad D ε dis k law y e := by
  cases e with
  | consistent => exact h
  | harvest h' => exact ⟨h.1, h.2.1, hy.trans h.2.2⟩
  | failed => exact h

/-- Over draws of bounded length the sampler's law is finite. -/
theorem harvestLaw_univ_ne_top {L : ℕ} (hlen : ∀ᵐ x ∂D, x.toList.length ≤ L) [IsFiniteMeasure D]
    (H : HypothesisDFA α) (site : Level5Site α) : harvestLaw D cut k H site Set.univ ≠ ⊤ := by
  classical
  rw [harvestLaw_apply]
  refine ne_top_of_le_ne_top (b := ((L + 1 : ℕ) : ℝ≥0∞) * D Set.univ) ?_ ?_
  · simp [ENNReal.mul_eq_top, measure_ne_top]
  rw [← lintegral_const]
  refine lintegral_mono_ae ?_
  filter_upwards [hlen] with x hx
  have h1 := List.length_filter_le (· ∈ Set.univ)
    (siteHarvest cut k site (H.tree, H.edges) x)
  have h2 := siteHarvest_length_le cut k x site (H.tree, H.edges)
  exact_mod_cast (by omega : ((siteHarvest cut k site (H.tree, H.edges)
    x).filter (· ∈ Set.univ)).length ≤ L + 1)

end Law

end OrthoDFA
