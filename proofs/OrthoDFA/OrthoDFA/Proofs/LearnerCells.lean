import OrthoDFA.Proofs.ReturnAccuracy

/-!
# One step of the learner, one pinned cell at a time

`loop_round_le` with what the step conditions on left abstract: a countable draw `a` fixes,
together with the noise at the strings read so far, whatever the step is handed.  The rest of
the step's draws are independent of both.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory
open scoped ENNReal

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
variable {S : Type*} [Stringlike S]

namespace LearnerProof

theorem cell_split_le {A W C : Type*} [MeasurableSpace A] [Countable A]
    [MeasurableSingletonClass A] [MeasurableSpace W] (O : Oracle μ S) (π : Measure A)
    [IsProbabilityMeasure π] (ζ : Measure W) [IsProbabilityMeasure ζ] {T : ℕ}
    (stage : Ω → A → C) (queried : Ω → A → Finset S)
    (hreads : ∀ a, ReadsOnly O (fun ω => queried ω a) (fun ω => stage ω a))
    (hT : ∀ ω a, (queried ω a).card ≤ T) (F : C → Set (Ω × W)) (B : ℝ≥0∞)
    (hF : ∀ c U b, U.card ≤ T →
      (μ.prod ζ) (F c ∩ {q | q.1 ∈ pinned O U b}) ≤ μ (pinned O U b) * B) :
    (μ.prod (π.prod ζ)) {z | (z.1, z.2.2) ∈ F (stage z.1 z.2.1)} ≤ B := by
  classical
  set L := μ.prod (π.prod ζ)
  set bits : Finset S → S → ℝ := fun t s => if s ∈ t then 1 else 0
  set cell : A → Finset S → Finset S → Set Ω := fun a U t =>
    {ω | queried ω a = U ∧ (∀ s ∈ U, O.noise s ω = 0 ∨ O.noise s ω = 1)
      ∧ U.filter (fun s => O.noise s ω = 1) = t}
  have hcell : ∀ a U t ω0, ω0 ∈ cell a U t →
      cell a U t = pinned O U (bits t) ∧ ∀ ω ∈ cell a U t, stage ω a = stage ω0 a := by
    intro a U t ω0 h0
    have hbits : ∀ ω ∈ cell a U t, ∀ s ∈ U, O.noise s ω = bits t s := by
      rintro ω ⟨_, hb, ht⟩ s hs
      simp only [bits]
      rcases hb s hs with h | h
      · rw [if_neg, h]
        intro hst
        rw [← ht] at hst
        have := (Finset.mem_filter.1 hst).2
        rw [h] at this
        norm_num at this
      · rw [if_pos, h]
        rw [← ht]
        exact Finset.mem_filter.2 ⟨hs, h⟩
    have hsame : ∀ ω, (∀ s ∈ U, O.noise s ω = bits t s) →
        queried ω a = U ∧ stage ω a = stage ω0 a := by
      intro ω hω
      have := hreads a ω0 ω (fun s hs => by
        have hs' : s ∈ U := by rw [← h0.1]; exact hs
        rw [hbits ω0 h0 s hs', hω s hs'])
      exact ⟨this.1.trans h0.1, this.2⟩
    have htU : t ⊆ U := by rw [← h0.2.2]; exact Finset.filter_subset _ _
    refine ⟨Set.ext fun ω => ⟨fun hω => hbits ω hω, fun hω => ?_⟩,
      fun ω hω => (hsame ω (hbits ω hω)).2⟩
    refine ⟨(hsame ω hω).1, fun s hs => ?_, ?_⟩
    · rw [hω s hs]
      simp only [bits]
      split_ifs <;> simp
    · ext s
      simp only [Finset.mem_filter]
      constructor
      · rintro ⟨hs, h1⟩
        rw [hω s hs] at h1
        simp only [bits] at h1
        split_ifs at h1 with hst
        · exact hst
        · norm_num at h1
      · intro hst
        exact ⟨htU hst, by rw [hω s (htU hst)]; simp [bits, hst]⟩
  have hcellm : ∀ a U t, MeasurableSet (cell a U t) := by
    intro a U t
    by_cases hne : (cell a U t).Nonempty
    · obtain ⟨ω0, h0⟩ := hne
      rw [(hcell a U t ω0 h0).1]
      exact measurableSet_pinned O U _
    · rw [Set.not_nonempty_iff_eq_empty.1 hne]
      exact MeasurableSet.empty
  have hcelld : ∀ a, Pairwise (Function.onFun Disjoint
      fun Ut : Finset S × Finset S => cell a Ut.1 Ut.2) := by
    rintro a ⟨U, t⟩ ⟨U', t'⟩ hne
    rw [Function.onFun, Set.disjoint_left]
    rintro ω ⟨hU, _, ht⟩ ⟨hU', _, ht'⟩
    have hUU : U = U' := hU.symm.trans hU'
    subst hUU
    exact hne (Prod.ext rfl (ht.symm.trans ht'))
  set Φ : Ω × A × W → A × (Ω × W) := fun z => (z.2.1, (z.1, z.2.2))
  have hΦ : MeasurePreserving Φ L (π.prod (μ.prod ζ)) := by
    have e2 := MeasurePreserving.symm _ (measurePreserving_prodAssoc μ π ζ)
    have e3 := (Measure.measurePreserving_swap (μ := μ) (ν := π)).prod
      (MeasurePreserving.id ζ)
    have e4 := measurePreserving_prodAssoc π μ ζ
    exact e4.comp (e3.comp e2)
  set piece : A → Finset S × Finset S → Set (Ω × A × W) :=
    fun a Ut => {z | z.2.1 = a ∧ z.1 ∈ cell a Ut.1 Ut.2 ∧ (z.1, z.2.2) ∈ F (stage z.1 z.2.1)}
  have hpiece : ∀ a Ut, L (piece a Ut) ≤ π {a} * (μ (cell a Ut.1 Ut.2) * B) := by
    rintro a ⟨U, t⟩
    by_cases hne : (cell a U t).Nonempty
    · obtain ⟨ω0, h0⟩ := hne
      obtain ⟨hceq, hst⟩ := hcell a U t ω0 h0
      have hU : U.card ≤ T := by rw [← h0.1]; exact hT ω0 a
      have hsub : piece a (U, t)
          ⊆ Φ ⁻¹' ({a} ×ˢ (F (stage ω0 a) ∩ {q | q.1 ∈ pinned O U (bits t)})) := by
        rintro z ⟨ha, hz, hFz⟩
        have hst' : stage z.1 z.2.1 = stage ω0 a := by rw [ha]; exact hst z.1 hz
        refine ⟨ha, ?_, ?_⟩
        · rw [← hst']; exact hFz
        · change z.1 ∈ pinned O U (bits t)
          rw [← hceq]; exact hz
      calc L (piece a (U, t))
          ≤ L (Φ ⁻¹' ({a} ×ˢ (F (stage ω0 a) ∩ {q | q.1 ∈ pinned O U (bits t)}))) :=
            measure_mono hsub
        _ ≤ (L.map Φ) ({a} ×ˢ (F (stage ω0 a) ∩ {q | q.1 ∈ pinned O U (bits t)})) :=
            Measure.le_map_apply hΦ.measurable.aemeasurable _
        _ = π {a} * (μ.prod ζ) (F (stage ω0 a) ∩ {q | q.1 ∈ pinned O U (bits t)}) := by
            rw [hΦ.map_eq, Measure.prod_prod]
        _ ≤ π {a} * (μ (pinned O U (bits t)) * B) := by
            gcongr
            exact hF _ U (bits t) hU
        _ = π {a} * (μ (cell a U t) * B) := by rw [← hceq]
    · have : piece a (U, t) = ∅ := by
        ext z
        simp only [Set.mem_empty_iff_false, iff_false]
        rintro ⟨_, hz, _⟩
        exact hne ⟨z.1, hz⟩
      rw [this, measure_empty]
      exact zero_le
  set N0 := {ω : Ω | ∃ s, ¬ (O.noise s ω = 0 ∨ O.noise s ω = 1)}
  have hN0 : μ N0 = 0 := by
    have : N0 = ⋃ s, {ω | ¬ (O.noise s ω = 0 ∨ O.noise s ω = 1)} := by
      ext ω; simp only [N0, Set.mem_ofPred_eq, Set.mem_iUnion]
    rw [this]
    exact measure_iUnion_null fun s => ae_iff.1 (O.noise_bit s)
  have hcover : {z : Ω × A × W | (z.1, z.2.2) ∈ F (stage z.1 z.2.1)}
      ⊆ N0 ×ˢ Set.univ ∪ ⋃ a, ⋃ Ut, piece a Ut := by
    intro z hz
    by_cases h0 : z.1 ∈ N0
    · exact Or.inl ⟨h0, trivial⟩
    · right
      have h0' : ∀ s, O.noise s z.1 = 0 ∨ O.noise s z.1 = 1 := fun s => by
        by_contra hc; exact h0 ⟨s, hc⟩
      refine Set.mem_iUnion.2 ⟨z.2.1, Set.mem_iUnion.2 ⟨(queried z.1 z.2.1,
        (queried z.1 z.2.1).filter fun s => O.noise s z.1 = 1), rfl,
        ⟨rfl, fun s _ => h0' s, rfl⟩, hz⟩⟩
  calc L {z | (z.1, z.2.2) ∈ F (stage z.1 z.2.1)}
      ≤ L (N0 ×ˢ Set.univ ∪ ⋃ a, ⋃ Ut, piece a Ut) := measure_mono hcover
    _ ≤ L (N0 ×ˢ Set.univ) + L (⋃ a, ⋃ Ut, piece a Ut) := measure_union_le _ _
    _ ≤ 0 + ∑' a, ∑' Ut, L (piece a Ut) := by
        gcongr
        · rw [Measure.prod_prod, hN0, zero_mul]
        · exact (measure_iUnion_le _).trans (ENNReal.tsum_le_tsum fun a => measure_iUnion_le _)
    _ ≤ ∑' a, ∑' Ut : Finset S × Finset S, π {a} * (μ (cell a Ut.1 Ut.2) * B) := by
        rw [zero_add]
        exact ENNReal.tsum_le_tsum fun a => ENNReal.tsum_le_tsum fun Ut => hpiece a Ut
    _ = ∑' a, π {a} * ((∑' Ut : Finset S × Finset S, μ (cell a Ut.1 Ut.2)) * B) := by
        refine tsum_congr fun a => ?_
        rw [ENNReal.tsum_mul_left, ENNReal.tsum_mul_right]
    _ ≤ ∑' a, π {a} * (1 * B) := by
        gcongr with a
        rw [← measure_iUnion (hcelld a) fun Ut => hcellm a Ut.1 Ut.2]
        exact prob_le_one
    _ = (∑' a, π {a}) * B := by rw [← ENNReal.tsum_mul_right]; simp only [one_mul]
    _ ≤ 1 * B := by
        gcongr
        rw [← measure_iUnion (fun a a' hne => Set.disjoint_singleton.2 hne)
          fun a => measurableSet_singleton a]
        exact prob_le_one
    _ = B := one_mul B

/-- Round `r`'s draws split into what its step conditions on and the rest: the earlier rounds'
draws with this round's first part, against this round's second part and its streams. -/
theorem measurePreserving_roundSplit {K : ℕ} {X X₁ X₂ Y : Type*} [MeasurableSpace X]
    [MeasurableSpace X₁] [MeasurableSpace X₂] [MeasurableSpace Y] (ν : Measure X)
    [IsProbabilityMeasure ν] (ν₁ : Measure X₁) [IsProbabilityMeasure ν₁] (ν₂ : Measure X₂)
    [IsProbabilityMeasure ν₂] (ξ : Measure Y) [IsProbabilityMeasure ξ] (σ : X → X₁ × X₂)
    (hσ : MeasurePreserving σ ν (ν₁.prod ν₂)) (r : Fin K) :
    MeasurePreserving (fun z : Ω × (Fin K → X) × (Fin K → Y) =>
        (z.1, (((fun i : {i : Fin K // i < r} => z.2.1 i), (σ (z.2.1 r)).1),
          ((σ (z.2.1 r)).2, z.2.2 r))))
      (μ.prod ((Measure.pi fun _ : Fin K => ν).prod (Measure.pi fun _ : Fin K => ξ)))
      (μ.prod (((Measure.pi fun _ : {i : Fin K // i < r} => ν).prod ν₁).prod (ν₂.prod ξ))) := by
  set π := Measure.pi fun _ : {i : Fin K // i < r} => ν
  have hsplit : MeasurePreserving
      (fun x : Fin K → X => ((fun i : {i : Fin K // i < r} => x i), x r))
      (Measure.pi fun _ : Fin K => ν) (π.prod ν) :=
    ((MeasurePreserving.id π).prod
      (measurePreserving_eval (fun _ : {i : Fin K // ¬ i < r} => ν) ⟨r, lt_irrefl r⟩)).comp
      (measurePreserving_piEquivPiSubtypeProd (fun _ : Fin K => ν) (fun i : Fin K => i < r))
  have hstep := hsplit.prod (measurePreserving_eval (fun _ : Fin K => ξ) r)
  have hσ' := ((MeasurePreserving.id π).prod hσ).prod (MeasurePreserving.id ξ)
  have e1 := (MeasurePreserving.symm _ (measurePreserving_prodAssoc π ν₁ ν₂)).prod
    (MeasurePreserving.id ξ)
  have e2 := measurePreserving_prodAssoc (π.prod ν₁) ν₂ ξ
  exact (MeasurePreserving.id μ).prod (e2.comp (e1.comp (hσ'.comp hstep)))

/-- Round `r`'s step, one pinned cell at a time, conditioning on the earlier rounds' draws and
this round's first part. -/
theorem round_split_le {K : ℕ} {X X₁ X₂ Y C : Type*} [MeasurableSpace X] [Countable X]
    [MeasurableSingletonClass X] [MeasurableSpace X₁] [Countable X₁] [MeasurableSingletonClass X₁]
    [MeasurableSpace X₂] [MeasurableSpace Y] (O : Oracle μ S) (ν : Measure X)
    [IsProbabilityMeasure ν] (ν₁ : Measure X₁) [IsProbabilityMeasure ν₁] (ν₂ : Measure X₂)
    [IsProbabilityMeasure ν₂] (ξ : Measure Y) [IsProbabilityMeasure ξ] (σ : X → X₁ × X₂)
    (hσ : MeasurePreserving σ ν (ν₁.prod ν₂)) (r : Fin K) {T : ℕ}
    (stage : Ω → ({i : Fin K // i < r} → X) × X₁ → C)
    (queried : Ω → ({i : Fin K // i < r} → X) × X₁ → Finset S)
    (hreads : ∀ a, ReadsOnly O (fun ω => queried ω a) (fun ω => stage ω a))
    (hT : ∀ ω a, (queried ω a).card ≤ T) (F : C → Set (Ω × (X₂ × Y))) (B : ℝ≥0∞)
    (hF : ∀ c U b, U.card ≤ T →
      (μ.prod (ν₂.prod ξ)) (F c ∩ {q | q.1 ∈ pinned O U b}) ≤ μ (pinned O U b) * B) :
    (μ.prod ((Measure.pi fun _ : Fin K => ν).prod (Measure.pi fun _ : Fin K => ξ)))
      {z | (z.1, ((σ (z.2.1 r)).2, z.2.2 r))
        ∈ F (stage z.1 ((fun i : {i : Fin K // i < r} => z.2.1 i), (σ (z.2.1 r)).1))} ≤ B := by
  have hΨ := measurePreserving_roundSplit (μ := μ) ν ν₁ ν₂ ξ σ hσ r
  refine (Measure.le_map_apply hΨ.measurable.aemeasurable
    {w | (w.1, w.2.2) ∈ F (stage w.1 w.2.1)}).trans ?_
  rw [hΨ.map_eq]
  exact cell_split_le O _ _ stage queried hreads hT F B hF

end LearnerProof

end OrthoDFA
