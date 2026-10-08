import OrthoDFA.Proofs.TripleBound

/-!
# `RoundAtK`

The batch's claims hold for any reads (`round_at_k_batch`). The triples' claim fails only where
the draws' shares sum past the fluctuation allowed (`triple_fail_sum`), which Chebyshev bounds in
each cell of the pass (`cell_tail_le`); the cells partition the noise.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory
open scoped ENNReal

variable {α : Type*} [Fintype α] [DecidableEq α]
variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} {Q : Type*}

theorem passCell_eq [IsProbabilityMeasure μ] (O : Oracle μ (FreeMonoid α)) (B : State)
    (F : Finset (FreeMonoid α)) (K : StageKnobs α) (k : ℕ) (seed probes : List (FreeMonoid α))
    {c : Finset (FreeMonoid α) × Finset (FreeMonoid α)} {ω₀ : Ω}
    (h₀ : ω₀ ∈ passCell O B F K k seed probes c) :
    passCell O B F K k seed probes c = pcell O (F ∪ K.train F) c ∩ cleanAll O := by
  ext ω; constructor
  · intro (hω : ω ∈ passCell O B F K k seed probes c); exact ⟨hω.1.1, hω.1.2⟩
  · rintro ⟨hP, hc'⟩; exact (passCell_const O B F K k seed probes h₀ hP hc').2

theorem measurableSet_passCell [IsProbabilityMeasure μ] (O : Oracle μ (FreeMonoid α))
    (B : State) (F : Finset (FreeMonoid α)) (K : StageKnobs α) (k : ℕ)
    (seed probes : List (FreeMonoid α)) (c : Finset (FreeMonoid α) × Finset (FreeMonoid α)) :
    MeasurableSet (passCell O B F K k seed probes c) := by
  by_cases h : (passCell O B F K k seed probes c).Nonempty
  · obtain ⟨ω₀, h₀⟩ := h
    rw [passCell_eq O B F K k seed probes h₀]
    exact (noiseAlg_le O _ _ ((measurableSet_noisePattern O _ c.2).inter
      (measurableSet_noiseClean O _))).inter (measurableSet_cleanAll O)
  · rw [Set.not_nonempty_iff_eq_empty.1 h]; exact MeasurableSet.empty

/-- The triples' claim over the oracle's noise. -/
theorem triple_holds_le [IsProbabilityMeasure μ] (A : DFA (FreeMonoid α) Q)
    (O : Oracle μ (FreeMonoid α)) (B : State) (F : Finset (FreeMonoid α)) (K : StageKnobs α)
    (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (k L : ℕ)
    (seed probes : List (FreeMonoid α)) {uGood ε : ℝ} (hu : 0 ≤ uGood) (hε : 0 < ε)
    (hlen : ∀ᵐ x ∂D, x.toList.length = L) (hV : SuffixFree (F ∪ K.train F)) :
    μ.real {ω | ¬ TripleHolds (readsAt O B F ω) A O B F
        (runPassK K (readsAt O B F ω) k (initialK K (readsAt O B F ω) seed) probes) D k L seed
        probes uGood ε}
      ≤ prefixMax D k / ε ^ 2 := by
  classical
  set V := F ∪ K.train F
  have hpm : 0 ≤ prefixMax D k :=
    Real.iSup_nonneg fun p => by split_ifs <;> simp [measureReal_nonneg]
  have hr : 0 ≤ prefixMax D k / ε ^ 2 := by positivity
  set Ev : Set Ω := {ω | ε * (1 + uGood * (passK O B F K k seed probes ω).tree.depth * L)
    < ∑ x ∈ wordsOf (α := α) L, D.real {x} * contrib A O B F uGood (readsAt O B F ω)
      (passK O B F K k seed probes ω).tree (passK O B F K k seed probes ω).edges
      (passReads O B F K k seed probes ω) k x}
  have hsub : {ω | ¬ TripleHolds (readsAt O B F ω) A O B F
      (runPassK K (readsAt O B F ω) k (initialK K (readsAt O B F ω) seed) probes) D k L seed
      probes uGood ε} ⊆ Ev := fun ω hω =>
    triple_fail_sum A O B F hu (readsAt O B F ω) _ _ _ D k L hlen hω
  by_cases hkL : k ≤ L
  swap
  · -- no draw of length `L` is long enough for a triple
    refine le_trans (le_of_eq ?_) hr
    rw [show {ω | ¬ TripleHolds (readsAt O B F ω) A O B F
        (runPassK K (readsAt O B F ω) k (initialK K (readsAt O B F ω) seed) probes) D k L seed
        probes uGood ε} = ∅ from Set.eq_empty_of_forall_notMem fun ω hω => hω ?_]
    · simp
    have h0 : D.real {x | ∃ j b, probeOutcome (readsAt O B F ω)
        (runPassK K (readsAt O B F ω) k (initialK K (readsAt O B F ω) seed) probes).tree
        (runPassK K (readsAt O B F ω) k (initialK K (readsAt O B F ω) seed) probes).edges k x
          = .triple j ∧ tripleRead (readsAt O B F ω)
          (runPassK K (readsAt O B F ω) k (initialK K (readsAt O B F ω) seed) probes).tree x j
          = some b ∧ stateIndecision A O B F (A.state b) < uGood} = 0 := by
      rw [measureReal_def, measure_mono_null (fun x hx => ?_) (ae_iff.1 hlen),
        ENNReal.toReal_zero]
      obtain ⟨j, b, hj, -⟩ := hx
      have hkj := probeOutcome_triple_gt hj
      obtain ⟨ps, hi, hw, hbr⟩ := probeOutcome_search _ hj trivial
      obtain ⟨-, -, -, -, hhi, -⟩ := walkCheck_inr _ hw
      have := bracketAt_triple_lt (α := α) _ ps _ _ _ _ hbr
      simp only [Set.mem_ofPred_eq]
      omega
    unfold TripleHolds
    rw [h0]
    have hv : 0 ≤ ∫ x, (visits (readsAt O B F ω)
        (runPassK K (readsAt O B F ω) k (initialK K (readsAt O B F ω) seed) probes).tree
        (runPassK K (readsAt O B F ω) k (initialK K (readsAt O B F ω) seed) probes).edges k x
          : ℝ) ∂D := integral_nonneg fun x => Nat.cast_nonneg _
    have hpm1 : 0 ≤ prefixMax D (k + 1) :=
      Real.iSup_nonneg fun p => by split_ifs <;> simp [measureReal_nonneg]
    positivity
  -- the cells of the pass
  set cells := fun c : Finset (FreeMonoid α) × Finset (FreeMonoid α) =>
    passCell O B F K k seed probes c
  have hcover : Ev ⊆ (cleanAll O)ᶜ ∪ ⋃ c, cells c ∩ Ev := by
    intro ω hω
    by_cases hcl : ω ∈ cleanAll O
    · refine .inr (Set.mem_iUnion.2 ⟨(passReads O B F K k seed probes ω,
        noisePattern O (vBits V (passReads O B F K k seed probes ω)) ω), ?_, hω⟩)
      exact ⟨⟨⟨rfl, fun y _ => hcl y⟩, hcl⟩, rfl⟩
    · exact .inl hcl
  have hcell : ∀ c, μ (cells c ∩ Ev) ≤ μ (cells c) * ENNReal.ofReal (prefixMax D k / ε ^ 2) := by
    intro c
    by_cases hne : (cells c).Nonempty
    · obtain ⟨ω₀, h₀⟩ := hne
      have := cell_tail_le O B F K k seed probes A D hkL hu hε hV h₀
      rw [← ENNReal.ofReal_toReal (measure_ne_top μ _), ← measureReal_def,
        ← ENNReal.ofReal_toReal (measure_ne_top μ (cells c)), ← measureReal_def,
        ← ENNReal.ofReal_mul measureReal_nonneg]
      refine ENNReal.ofReal_le_ofReal (le_trans this (le_of_eq ?_))
      ring
    · rw [Set.not_nonempty_iff_eq_empty.1 hne]; simp
  have hdisj : Pairwise (Function.onFun Disjoint cells) := by
    intro c c' hcc'
    rw [Function.onFun, Set.disjoint_left]
    intro ω hω hω'
    apply hcc'
    have h1 : c.1 = c'.1 := hω.2.symm.trans hω'.2
    have h2 : c.2 = c'.2 := by
      have a : noisePattern O (vBits V c.1) ω = c.2 := hω.1.1.1
      have b : noisePattern O (vBits V c'.1) ω = c'.2 := hω'.1.1.1
      rw [h1] at a
      exact a.symm.trans b
    exact Prod.ext h1 h2
  have hmeas : ∀ c, MeasurableSet (cells c) := measurableSet_passCell O B F K k seed probes
  have htot : μ Ev ≤ ENNReal.ofReal (prefixMax D k / ε ^ 2) := by
    calc μ Ev ≤ μ ((cleanAll O)ᶜ ∪ ⋃ c, cells c ∩ Ev) := measure_mono hcover
      _ ≤ μ (cleanAll O)ᶜ + μ (⋃ c, cells c ∩ Ev) := measure_union_le _ _
      _ = μ (⋃ c, cells c ∩ Ev) := by rw [measure_cleanAll_compl O, zero_add]
      _ ≤ ∑' c, μ (cells c ∩ Ev) := measure_iUnion_le _
      _ ≤ ∑' c, μ (cells c) * ENNReal.ofReal (prefixMax D k / ε ^ 2) :=
          ENNReal.tsum_le_tsum hcell
      _ = (∑' c, μ (cells c)) * ENNReal.ofReal (prefixMax D k / ε ^ 2) := ENNReal.tsum_mul_right
      _ = μ (⋃ c, cells c) * ENNReal.ofReal (prefixMax D k / ε ^ 2) := by
          rw [measure_iUnion hdisj hmeas]
      _ ≤ 1 * ENNReal.ofReal (prefixMax D k / ε ^ 2) := by gcongr; exact prob_le_one
      _ = _ := one_mul _
  calc μ.real _ ≤ μ.real Ev := measureReal_mono hsub (measure_ne_top _ _)
    _ ≤ prefixMax D k / ε ^ 2 := by
        rw [measureReal_def]; exact ENNReal.toReal_le_of_le_ofReal hr htot

theorem round_at_k_holds : RoundAtK := by
  intro α _ _ Ω _ μ _ Q A O B F K D _ k L ng n₀ seed probes acc θp f a δ uGood ε hacc0 hacc1 hθp0
    hθp1 hf ha hδ hu hε hlen hV
  exact ⟨fun R => round_at_k_batch K R D k ng n₀ seed probes hacc0 hacc1 hθp0 hθp1 hf ha hδ,
    triple_holds_le A O B F K D k L seed probes hu hε hlen hV⟩

end OrthoDFA
