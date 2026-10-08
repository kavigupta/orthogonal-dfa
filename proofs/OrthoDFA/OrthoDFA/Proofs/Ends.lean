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

end OrthoDFA
