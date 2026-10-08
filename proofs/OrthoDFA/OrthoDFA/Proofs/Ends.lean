import OrthoDFA.Ends

namespace OrthoDFA

open MeasureTheory

theorem start_root_covered : StartRootCovered := by
  intro α _ _ Ω _ μ _ O B F D _ k b hb
  have hint : ∀ ν : Measure (FreeMonoid α), IsProbabilityMeasure ν →
      ∀ g : FreeMonoid α → Set Ω, Integrable (fun p => μ.real (g p)) ν := fun ν _ g =>
    Integrable.of_bound measurable_from_top.aestronglyMeasurable 1
      (Filter.Eventually.of_forall fun p => by
        rw [Real.norm_of_nonneg measureReal_nonneg]; exact measureReal_le_one)
  refine le_trans ?_ hb
  rw [startPopulation, integral_map measurable_from_top.aemeasurable
    measurable_from_top.aestronglyMeasurable]
  refine integral_mono (hint D inferInstance _) (hint D inferInstance _) fun x => ?_
  refine measureReal_mono (fun ω hω => ?_)
  simp only [Set.mem_ofPred_eq, mul_one, CutReads.cut, readsAt, decided, voteCount] at hω ⊢
  unfold acceptsOn at hω
  intro hd
  split_ifs at hω with h1 h2
  rcases hd with hd | hd
  · exact h1 (by convert Nat.lt_of_succ_lt hd using 2)
  · exact h2 (by convert hd using 2)

variable {α : Type*} [Fintype α] [DecidableEq α]

theorem route_undecided_mid (cut : FreeMonoid α → Option Bool) (x : FreeMonoid α) :
    ∀ t : DTree α, (t.route cut x).2.isRight →
      ∃ m ∈ DTree.midfixes t, cut (x * m) = none
  | .leaf, h => by simp [DTree.route] at h
  | .node m r a, h => by
    simp only [DTree.route] at h
    split at h
    · exact ⟨m, by simp [DTree.midfixes], by assumption⟩
    · obtain ⟨m', hm', hc⟩ := route_undecided_mid cut x a (by simpa using h)
      exact ⟨m', by simp [DTree.midfixes, hm'], hc⟩
    · obtain ⟨m', hm', hc⟩ := route_undecided_mid cut x r (by simpa using h)
      exact ⟨m', by simp [DTree.midfixes, hm'], hc⟩

theorem deep_undecided (R : CutReads α) (t : DTree α) (x : FreeMonoid α)
    (h : DeepUndecided R t x) : ∃ m ∈ t.belowRoot, R.cut (x * m) = none := by
  obtain ⟨hr, hl⟩ := h
  cases t with
  | leaf => simp [DTree.route] at hl
  | node m r a =>
    simp only [DTree.sift, DTree.route] at hr hl
    split at hr
    · simp_all
    · obtain ⟨m', hm', hc⟩ := route_undecided_mid R.cut x a (by simpa using hr)
      exact ⟨m', by simp [DTree.belowRoot, hm'], hc⟩
    · obtain ⟨m', hm', hc⟩ := route_undecided_mid R.cut x r (by simpa using hr)
      exact ⟨m', by simp [DTree.belowRoot, hm'], hc⟩

theorem ends_covered : EndsCovered := by
  intro α _ _ R t X _
  calc X.real {x | DeepUndecided R t x}
      ≤ X.real (⋃ m ∈ t.belowRoot, (· * m) ⁻¹' {p | R.cut p = none}) := by
        refine measureReal_mono (fun x hx => ?_) (measure_ne_top _ _)
        obtain ⟨m, hm, hc⟩ := deep_undecided R t x hx
        exact Set.mem_biUnion hm hc
    _ ≤ ∑ m ∈ t.belowRoot, X.real ((· * m) ⁻¹' {p | R.cut p = none}) :=
        measureReal_biUnion_finset_le _ _
    _ = ∑ m ∈ t.belowRoot, (endPopulation X m).real {p | R.cut p = none} := by
        refine Finset.sum_congr rfl fun m _ => ?_
        rw [endPopulation,
          map_measureReal_apply measurable_from_top (MeasurableSpace.measurableSet_top)]

theorem bad_share : BadShare := by
  intro α _ _ Ω _ μ _ Q A O B F P _ f umax hf hmax
  set u := fun p => stateIndecision A O B F (A.state p)
  have hle : ∀ p, undecidedProb O B.lo B.hi F p ≤ u p :=
    fun p => le_csSup ⟨1, by rintro _ ⟨s, -, rfl⟩; exact measureReal_le_one⟩ ⟨p, rfl, rfl⟩
  have hpt : ∀ p, undecidedProb O B.lo B.hi F p
      ≤ f + (umax - f) * {p | f < u p}.indicator (fun _ => (1 : ℝ)) p := by
    intro p
    simp only [Set.indicator, Set.mem_ofPred_eq]
    split_ifs with h
    · linarith [hle p, hmax (A.state p)]
    · linarith [hle p, not_lt.1 h]
  have hb : ∀ g : FreeMonoid α → ℝ, (∀ p, |g p| ≤ |f| + |umax - f|) → Integrable g P :=
    fun g hg => Integrable.of_bound measurable_from_top.aestronglyMeasurable _
      (Filter.Eventually.of_forall fun p => by rw [Real.norm_eq_abs]; exact hg p)
  have hf' := le_abs_self f
  have hf'' := neg_abs_le f
  have hd : umax - f = |umax - f| := (abs_of_nonneg (by linarith)).symm
  calc ∫ p, undecidedProb O B.lo B.hi F p ∂P
      ≤ ∫ p, f + (umax - f) * {p | f < u p}.indicator (fun _ => (1 : ℝ)) p ∂P := by
        refine integral_mono (hb _ fun p => ?_) (hb _ fun p => ?_) hpt
        · have h0 : 0 ≤ undecidedProb O B.lo B.hi F p := measureReal_nonneg
          exact abs_le.2 ⟨by linarith [abs_nonneg f, abs_nonneg (umax - f)],
            by linarith [hle p, hmax (A.state p)]⟩
        · simp only [Set.indicator, Set.mem_ofPred_eq]
          split_ifs
          · exact abs_le.2 ⟨by linarith, by linarith⟩
          · exact abs_le.2 ⟨by linarith [abs_nonneg (umax - f)], by linarith⟩
    _ = f + (umax - f) * P.real {p | f < u p} := by
        rw [integral_add (integrable_const _) ((integrable_const _).indicator
            MeasurableSpace.measurableSet_top |>.const_mul _), integral_const, integral_const_mul,
          show (fun _ => (1 : ℝ)) = (1 : FreeMonoid α → ℝ) from rfl,
          integral_indicator_one MeasurableSpace.measurableSet_top]
        simp

end OrthoDFA
