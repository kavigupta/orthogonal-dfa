import OrthoDFA.Proofs.HarvestClasses

/-!
# The classes' quality

`harvest_holds_le` for each of the five classes, off the union of their noise sets.
-/

namespace OrthoDFA

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

theorem integral_le_words (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] {L : ℕ}
    (hlen : ∀ᵐ x ∂D, x.toList.length = L) {f g : FreeMonoid α → ℝ}
    (h : ∀ x, x.toList.length = L → f x ≤ g x) : ∫ x, f x ∂D ≤ ∫ x, g x ∂D := by
  rw [integral_eq_sum_words D hlen f, integral_eq_sum_words D hlen g]
  exact Finset.sum_le_sum fun x hx =>
    mul_le_mul_of_nonneg_left (h x (mem_wordsOf.1 hx)) measureReal_nonneg

open scoped Classical in
theorem integral_ite_words (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] {L : ℕ}
    (hlen : ∀ᵐ x ∂D, x.toList.length = L) (P : FreeMonoid α → Prop) (a : ℝ) :
    ∫ x, a * (if P x then 1 else 0) ∂D = a * D.real {x | P x} := by
  rw [integral_eq_sum_words D hlen, real_eq_sum_words D hlen, Finset.mul_sum]
  refine Finset.sum_congr rfl fun x _ => ?_
  simp only [Set.mem_ofPred_eq]
  split_ifs <;> ring

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} {Q : Type*}

omit [Fintype α] [DecidableEq α] in
theorem exists_good_of_le {P : Ω → Prop} {r : ℝ} (h : μ.real {ω | ¬ P ω} ≤ r) :
    ∃ E : Set Ω, μ.real E ≤ r ∧ ∀ ω ∉ E, P ω :=
  ⟨_, h, fun ω hω => by simpa using hω⟩

theorem quality_holds [IsProbabilityMeasure μ] (A : DFA (FreeMonoid α) Q)
    (O : Oracle μ (FreeMonoid α)) (B : State) (F : Finset (FreeMonoid α)) (K : StageKnobs α)
    (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (L : ℕ)
    (seed probes : List (FreeMonoid α)) {f c ε : ℝ} (hf : 0 ≤ f) (hc : 0 ≤ c) (hε : 0 < ε)
    (hlen : ∀ᵐ x ∂D, x.toList.length = L) (hV : SuffixFree (F ∪ K.train F)) :
    ∃ E : Set Ω, μ.real E ≤ 5 * prefixMax D ((L + 1) / 2) / ε ^ 2 ∧ ∀ ω ∉ E,
      QualityHolds (readsAt O B F ω) A O B F
        (runPassK K (readsAt O B F ω) ((L + 1) / 2) (initialK K (readsAt O B F ω) seed) probes)
        D ((L + 1) / 2) L seed probes f c ε := by
  classical
  set k := (L + 1) / 2
  have hkL : k ≤ L := by omega
  have hu : 0 ≤ c * f := mul_nonneg hc hf
  have hT := harvest_holds_le (μ := μ) (triple_spec k) (fun t n => t.depth * n)
    (fun R t e x => (qProbeH_countP R t e k x).trans (Nat.mul_le_mul_left _ (visits_le R t e k x)))
    A O B F K D seed probes hu hε hkL hlen hV
  have hP := harvest_holds_le (μ := μ) (pair_spec k) (fun t n => t.depth * n)
    (fun R t e x => (qProbeH_countP R t e k x).trans (Nat.mul_le_mul_left _ (visits_le R t e k x)))
    A O B F K D seed probes hu hε hkL hlen hV
  have hS := harvest_holds_le (μ := μ) (ends_spec k (prefixOf · k) fun x => ⟨k, le_rfl, rfl⟩)
    (fun t _ => t.depth - 1)
    (fun R t e x => qSiftDeep_countP R.cut _ t) A O B F K D seed probes hu hε hkL hlen hV
  have hE := harvest_holds_le (μ := μ)
    (ends_spec k id fun x => ⟨max k x.toList.length, le_max_left _ _,
    (prefixOf_max x k).symm⟩) (fun t _ => t.depth - 1)
    (fun R t e x => qSiftDeep_countP R.cut _ t) A O B F K D seed probes hu hε hkL hlen hV
  have hB := harvest_holds_le (μ := μ) (blocked_spec k) (fun t _ => 2 * t.depth)
    (fun R t e x => qWalkB_countP R.cut t e k x) A O B F K D seed probes hu hε hkL hlen hV
  obtain ⟨E1, hE1, hc1⟩ := exists_good_of_le hT
  obtain ⟨E2, hE2, hc2⟩ := exists_good_of_le hP
  obtain ⟨E3, hE3, hc3⟩ := exists_good_of_le hS
  obtain ⟨E4, hE4, hc4⟩ := exists_good_of_le hE
  obtain ⟨E5, hE5, hc5⟩ := exists_good_of_le hB
  refine ⟨E1 ∪ E2 ∪ E3 ∪ E4 ∪ E5, ?_, fun ω hω => ?_⟩
  · refine (measureReal_union_le _ _).trans ?_
    refine (add_le_add (measureReal_union_le _ _) hE5).trans ?_
    refine (add_le_add (add_le_add (measureReal_union_le _ _) hE4) le_rfl).trans ?_
    refine (add_le_add (add_le_add (add_le_add (measureReal_union_le _ _) hE3) le_rfl)
      le_rfl).trans ?_
    have := add_le_add hE1 hE2
    rw [mul_div_assoc]
    linarith
  simp only [Set.mem_union, not_or] at hω
  obtain ⟨⟨⟨⟨n1, n2⟩, n3⟩, n4⟩, n5⟩ := hω
  have h1 := hc1 ω n1
  have h2 := hc2 ω n2
  have h3 := hc3 ω n3
  have h4 := hc4 ω n4
  have h5 := hc5 ω n5
  set R := readsAt O B F ω
  set s := passK O B F K k seed probes ω
  change QualityHolds R A O B F s D k L seed probes f c ε
  set Tn := (kPrefixes k (passReadSet k seed probes s.tree)).card
  have hpm : 0 ≤ prefixMax D k :=
    Real.iSup_nonneg fun p => by split_ifs <;> simp [measureReal_nonneg]
  have hT0 : (0 : ℝ) ≤ Tn := Nat.cast_nonneg _
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · -- triples
    have hsub : {x | ∃ j b, probeOutcome R s.tree s.edges k x = .triple j
        ∧ tripleRead R s.tree x j = some b ∧ WellRead A O B F (c * f) b}
        ⊆ {x | harvTriple ((qProbeH s.tree s.edges k x).run R.cut) ≠ []
          ∧ ∀ b ∈ harvTriple ((qProbeH s.tree s.edges k x).run R.cut),
            stateIndecision A O B F (A.state b) < c * f} := by
      rintro x ⟨j, b, hj, hb, hw⟩
      have hk := probeOutcome_triple_gt hj
      have hr := qProbeH_run R s.tree s.edges k x (j := j) (by rw [hj]; rfl) hk.le
      simp only [Set.mem_ofPred_eq]
      rw [hr, hj, hb]
      exact ⟨by simp [harvTriple], by simpa [harvTriple, WellRead] using hw⟩
    have hint : ∫ x, (gTags (fun t e x => qProbeH t e k x) R.cut s.tree s.edges x : ℝ) ∂D
        ≤ s.tree.depth * searchSteps L k * D.real {x | Searched R s.tree s.edges k x} := by
      refine (integral_le_words D hlen (g := fun x => s.tree.depth * (searchSteps L k
        * if Searched R s.tree s.edges k x then 1 else 0)) fun x hx => ?_).trans
        (le_of_eq ?_)
      · have h1 : (gTags (fun t e x => qProbeH t e k x) R.cut s.tree s.edges x : ℝ)
            ≤ s.tree.depth * visits R s.tree s.edges k x := by
          exact_mod_cast qProbeH_countP R s.tree s.edges k x
        exact h1.trans (mul_le_mul_of_nonneg_left (visits_le_steps R s.tree s.edges hx)
          (Nat.cast_nonneg _))
      · simp_rw [← mul_assoc]
        rw [integral_ite_words D hlen]
    refine (measureReal_mono hsub).trans (h1.trans ?_)
    push_cast
    linarith [mul_le_mul_of_nonneg_left hint hu]
  · -- pairs
    have hsub : {x | ∃ j b b', probeOutcome R s.tree s.edges k x = .pair j
        ∧ tripleRead R s.tree x j = some b ∧ tripleRead R s.tree x (j + 1) = some b'
        ∧ WellRead A O B F (c * f) b ∧ WellRead A O B F (c * f) b'}
        ⊆ {x | harvPair ((qProbeH s.tree s.edges k x).run R.cut) ≠ []
          ∧ ∀ b ∈ harvPair ((qProbeH s.tree s.edges k x).run R.cut),
            stateIndecision A O B F (A.state b) < c * f} := by
      rintro x ⟨j, b, b', hj, hb, hb', hw, hw'⟩
      have hk := probeOutcome_pair_gt hj
      have hr := qProbeH_run R s.tree s.edges k x (j := j) (by rw [hj]; rfl) hk.le
      simp only [Set.mem_ofPred_eq]
      rw [hr, hj, hb, hb']
      refine ⟨by simp [harvPair], fun z hz => ?_⟩
      simp only [harvPair, List.mem_cons, List.not_mem_nil, or_false] at hz
      rcases hz with rfl | rfl
      · exact hw
      · exact hw'
    have hint : ∫ x, (gTags (fun t e x => qProbeH t e k x) R.cut s.tree s.edges x : ℝ) ∂D
        ≤ s.tree.depth * searchSteps L k * D.real {x | Searched R s.tree s.edges k x} := by
      refine (integral_le_words D hlen (g := fun x => s.tree.depth * (searchSteps L k
        * if Searched R s.tree s.edges k x then 1 else 0)) fun x hx => ?_).trans
        (le_of_eq ?_)
      · have h1 : (gTags (fun t e x => qProbeH t e k x) R.cut s.tree s.edges x : ℝ)
            ≤ s.tree.depth * visits R s.tree s.edges k x := by
          exact_mod_cast qProbeH_countP R s.tree s.edges k x
        exact h1.trans (mul_le_mul_of_nonneg_left (visits_le_steps R s.tree s.edges hx)
          (Nat.cast_nonneg _))
      · simp_rw [← mul_assoc]
        rw [integral_ite_words D hlen]
    refine (measureReal_mono hsub).trans (h2.trans ?_)
    push_cast
    linarith [mul_le_mul_of_nonneg_left hint hu]
  · -- the starts
    have hsub : {x | ∃ b, s.tree.sift R.cut (prefixOf x k) = .inr b ∧ StartDeep R s.tree k x
        ∧ WellRead A O B F (c * f) b}
        ⊆ {x | ((qSiftDeep (prefixOf x k) s.tree).run R.cut).toList ≠ []
          ∧ ∀ b ∈ ((qSiftDeep (prefixOf x k) s.tree).run R.cut).toList,
            stateIndecision A O B F (A.state b) < c * f} := by
      rintro x ⟨b, hb, ⟨-, hl⟩, hw⟩
      simp only [Set.mem_ofPred_eq]
      rw [qSiftDeep_of_deep R.cut _ b s.tree hb hl]
      exact ⟨by simp, by simpa [WellRead] using hw⟩
    have hint : ∫ x, (gTags (fun t (_ : Edges α) x => qSiftDeep (prefixOf x k) t) R.cut s.tree
        s.edges x : ℝ) ∂D ≤ ((s.tree.depth - 1 : ℕ) : ℝ) := by
      refine (integral_le_words D hlen (g := fun _ => ((s.tree.depth - 1 : ℕ) : ℝ))
        fun x _ => ?_).trans (by simp)
      exact_mod_cast qSiftDeep_countP R.cut _ s.tree
    refine (measureReal_mono hsub).trans (h3.trans ?_)
    linarith [mul_le_mul_of_nonneg_left hint hu]
  · -- the wholes
    have hsub : {x | ∃ b, s.tree.sift R.cut x = .inr b ∧ EndDeep R s.tree x
        ∧ WellRead A O B F (c * f) b}
        ⊆ {x | ((qSiftDeep (id x) s.tree).run R.cut).toList ≠ []
          ∧ ∀ b ∈ ((qSiftDeep (id x) s.tree).run R.cut).toList,
            stateIndecision A O B F (A.state b) < c * f} := by
      rintro x ⟨b, hb, ⟨-, hl⟩, hw⟩
      simp only [Set.mem_ofPred_eq]
      rw [id, qSiftDeep_of_deep R.cut _ b s.tree hb hl]
      exact ⟨by simp, by simpa [WellRead] using hw⟩
    have hint : ∫ x, (gTags (fun t (_ : Edges α) x => qSiftDeep (id x) t) R.cut s.tree
        s.edges x : ℝ) ∂D ≤ ((s.tree.depth - 1 : ℕ) : ℝ) := by
      refine (integral_le_words D hlen (g := fun _ => ((s.tree.depth - 1 : ℕ) : ℝ))
        fun x _ => ?_).trans (by simp)
      exact_mod_cast qSiftDeep_countP R.cut _ s.tree
    refine (measureReal_mono hsub).trans (h4.trans ?_)
    linarith [mul_le_mul_of_nonneg_left hint hu]
  · -- the reads an unlearned edge leaves undecided
    have hsub : {x | ∃ w b, IsBlocked R s.tree s.edges k x
        ∧ probeOutcome R s.tree s.edges k x = .endUndecided w ∧ s.tree.sift R.cut w = .inr b
        ∧ WellRead A O B F (c * f) b}
        ⊆ {x | ((qWalkB s.tree s.edges k x).run R.cut).toList ≠ []
          ∧ ∀ b ∈ ((qWalkB s.tree s.edges k x).run R.cut).toList,
            stateIndecision A O B F (A.state b) < c * f} := by
      rintro x ⟨w, b, hB, hw, hb, hwr⟩
      simp only [Set.mem_ofPred_eq]
      rw [qWalkB_of_blocked R s.tree s.edges k x w b hB hw hb]
      exact ⟨by simp, by simpa [WellRead] using hwr⟩
    have hint : ∫ x, (gTags (fun t e x => qWalkB t e k x) R.cut s.tree s.edges x : ℝ) ∂D
        ≤ 2 * (s.tree.depth : ℝ) := by
      refine (integral_le_words D hlen (g := fun _ => 2 * (s.tree.depth : ℝ))
        fun x _ => ?_).trans (by simp)
      exact_mod_cast qWalkB_countP R.cut s.tree s.edges k x
    refine (measureReal_mono hsub).trans (h5.trans ?_)
    push_cast
    linarith [mul_le_mul_of_nonneg_left hint hu]

end OrthoDFA
