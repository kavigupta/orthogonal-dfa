import OrthoDFA.Proofs.LoopE1

/-!
# Level 4's probes avoid the clustering's prefixes

The probes are drawn from `D` conditioned on their first `k` letters avoiding those of the
clustering's prefixes: each draw avoids them (`drawFrom_avoids`), the conditioning removes at most
`|K₀| · prefixMax` of `D` (`real_compl_avoiding_le`), and raises its prefix mass by at most the
factor `1/(1 − m)` (`prefixMax_avoidLaw_le`).
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α] {K : ℕ}

lemma clusterKeys_eq {Ω : Type*} [MeasurableSpace Ω] (sched : Finset State) (pops : Finset (Option (Fin K))) (k : ℕ)
    (x : Run Ω (FreeMonoid α) (Option (Fin K))) :
    clusterKeys sched pops k x = (schedPrefs sched pops x).image fun p => prefixOf p k := by
  classical
  unfold clusterKeys schedPrefs
  congr 1
  ext p
  simp only [Finset.mem_biUnion, Finset.mem_union]
  constructor
  · rintro ⟨B, hB, hp | ⟨j, hj, hp⟩⟩
    · exact Or.inl (prefixesAt_mono pops (Finset.le_sup (f := State.npref) hB) x hp)
    · exact Or.inr ⟨j, hj, certOf_mono j (by
        have := Finset.le_sup (f := State.npref) hB; unfold schedNP; omega) x hp⟩
  · intro h
    rcases sched.eq_empty_or_nonempty with hs | hs
    · subst hs
      simp only [schedNP, Finset.sup_empty, Nat.bot_eq_zero, mul_zero] at h
      rcases h with h | ⟨j, -, h⟩
      · obtain ⟨j, -, h⟩ := Finset.mem_biUnion.1 h
        simp [prefixesOf] at h
      · simp [certOf] at h
    · obtain ⟨B, hB, hBmax⟩ := Finset.exists_mem_eq_sup sched hs State.npref
      refine ⟨B, hB, ?_⟩
      unfold schedNP at h
      rw [hBmax] at h
      rcases h with h | ⟨j, hj, h⟩
      · exact Or.inl h
      · exact Or.inr ⟨j, hj, h⟩

lemma drawFrom_pos_or_one (ν : Measure (FreeMonoid α)) [IsFiniteMeasure ν] (u : unitInterval) :
    0 < ν.real {drawFrom ν u} ∨ drawFrom ν u = 1 := by
  rcases (drawFrom_eq_iff ν u (drawFrom ν u)).1 rfl with ⟨h1, h2⟩ | ⟨h, -⟩
  · left
    rw [massBelow_succ] at h2
    have : {z : FreeMonoid α | strCode z = strCode (drawFrom ν u)} = {drawFrom ν u} := by
      ext z; simp [strCode_injective.eq_iff]
    rw [this] at h2
    linarith
  · exact Or.inr h

lemma avoidLaw_singleton (D : Measure (FreeMonoid α)) {k : ℕ} (K₀ : Finset (FreeMonoid α))
    {z : FreeMonoid α} (hz : ¬ Avoids k ↑K₀ z) (h1 : z ≠ 1) : avoidLaw D k K₀ {z} = 0 := by
  unfold avoidLaw popLaw
  split_ifs
  · rw [Measure.dirac_apply' _ (measurableSet_singleton _)]
    simp only [Set.indicator, Set.mem_singleton_iff]
    rw [if_neg (Ne.symm h1)]
  · rw [Measure.smul_apply, Measure.restrict_apply (measurableSet_singleton _)]
    have : {z} ∩ {x : FreeMonoid α | k ≤ x.toList.length → prefixOf x k ∉ K₀} = ∅ := by
      ext z'
      simp only [Set.mem_inter_iff, Set.mem_singleton_iff, Set.mem_ofPred_eq,
        Set.mem_empty_iff_false, iff_false, not_and]
      rintro rfl
      exact hz
    rw [this, measure_empty, smul_zero]

lemma one_avoids {k : ℕ} (hk : 1 ≤ k) (K₀ : Set (FreeMonoid α)) : Avoids k K₀ 1 := by
  intro hk'
  simp at hk'
  omega

lemma drawFrom_avoids (D : Measure (FreeMonoid α)) {k : ℕ} (hk : 1 ≤ k) (K₀ : Finset (FreeMonoid α))
    (u : unitInterval) : Avoids k ↑K₀ (drawFrom (avoidLaw D k K₀) u) := by
  rcases drawFrom_pos_or_one (avoidLaw D k K₀) u with h | h
  · by_contra hn
    have h1 : drawFrom (avoidLaw D k K₀) u ≠ 1 := fun h1 => hn (h1 ▸ one_avoids hk _)
    rw [measureReal_def, avoidLaw_singleton D K₀ hn h1, ENNReal.toReal_zero] at h
    exact lt_irrefl _ h
  · rw [h]
    exact one_avoids hk _

lemma real_compl_avoiding_le (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (k : ℕ)
    (K₀ : Finset (FreeMonoid α)) :
    D.real (avoiding k ↑K₀)ᶜ ≤ K₀.card * prefixMax D k := by
  classical
  have hsub : (avoiding (α := α) k ↑K₀)ᶜ ⊆ ⋃ w ∈ K₀, {x | prefixOf x k = w} := by
    intro x hx
    have hx' : ¬ (k ≤ x.toList.length → prefixOf x k ∉ (K₀ : Set (FreeMonoid α))) := hx
    by_contra hno
    apply hx'
    intro _ hmem
    exact hno (Set.mem_iUnion₂.2 ⟨_, hmem, rfl⟩)
  calc D.real (avoiding k ↑K₀)ᶜ ≤ D.real (⋃ w ∈ K₀, {x | prefixOf x k = w}) :=
        measureReal_mono hsub (measure_ne_top _ _)
    _ ≤ ∑ w ∈ K₀, D.real {x | prefixOf x k = w} := measureReal_biUnion_finset_le _ _
    _ ≤ ∑ _w ∈ K₀, prefixMax D k := Finset.sum_le_sum fun w _ => le_prefixMax D k w
    _ = K₀.card * prefixMax D k := by rw [Finset.sum_const, nsmul_eq_mul]

lemma avoidLaw_eq (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (k : ℕ)
    (K₀ : Finset (FreeMonoid α)) : avoidLaw D k K₀ = popLaw (D.restrict (avoiding k ↑K₀)) := rfl

lemma prefixMax_avoidLaw_le (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (k : ℕ)
    (K₀ : Finset (FreeMonoid α)) (hm : D.real (avoiding k ↑K₀)ᶜ < 1) :
    prefixMax (avoidLaw D k K₀) k ≤ prefixMax D k / (1 - D.real (avoiding k ↑K₀)ᶜ) := by
  set A := avoiding (α := α) k ↑K₀
  have hA : D.real A = 1 - D.real Aᶜ := by
    rw [probReal_compl_eq_one_sub (measurableSet_of_countable A)]; ring
  have hApos : 0 < D.real A := by rw [hA]; linarith
  have h0 : D.restrict A Set.univ ≠ 0 := by
    rw [Measure.restrict_apply_univ]
    intro h0
    rw [measureReal_def, h0, ENNReal.toReal_zero] at hApos
    exact lt_irrefl _ hApos
  refine ciSup_le fun w => ?_
  rw [avoidLaw_eq, popLaw_real h0 (measure_ne_top _ _), measureReal_restrict_apply_univ,
    ← hA, measureReal_restrict_apply (measurableSet_of_countable _)]
  exact div_le_div_of_nonneg_right ((measureReal_mono Set.inter_subset_left
    (measure_ne_top _ _)).trans (le_prefixMax D k w)) hApos.le

omit [DecidableEq α] in
lemma QuarterEndsWell.mono {σ H : Type*} {M : _root_.DFA α σ} {bad : σ → Prop}
    {D : Measure (FreeMonoid α)} {ε ε' : ℝ} {dis : Set (FreeMonoid α)} {k : ℕ}
    {law : H → Measure (FreeMonoid α)} {y y' : ℝ} (hε : ε ≤ ε') (hy : y' ≤ y) {e : RoundEnd H}
    (h : QuarterEndsWell M bad D ε dis k law y e) : QuarterEndsWell M bad D ε' dis k law y' e := by
  cases e with
  | consistent => exact le_trans h hε
  | harvest h' => exact ⟨h.1, h.2.1, hy.trans h.2.2⟩
  | failed => exact h

end OrthoDFA
