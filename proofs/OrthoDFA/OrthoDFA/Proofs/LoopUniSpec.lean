import OrthoDFA.Proofs.LoopColl
import OrthoDFA.Proofs.LoopE1

/-!
# The uniform pool's cut at the run's noise

`UniSpec`: some state of the schedule returns, and at each that does the family cuts all but
`(513/512) εcov` of the uniform pool the noiseless way and leaves at most `2ι` of it undecided,
reading the run's own noise. Unlike `ClusterSpec`, these are the masses of the strings the noise
the run drew misreads, not chances over a fresh noise.
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α] {Ω : Type*} [MeasurableSpace Ω]
  {μ : Measure Ω} {K : ℕ}

/-- Some state of `sched` returns, and at each that does the family cuts all but `(513/512) εc`
of `Du` the noiseless way and leaves at most `2ι` of it undecided, at the run's noise. -/
def UniSpec (O : Oracle μ (FreeMonoid α)) (pops : Finset (Option (Fin K))) (sched : Finset State)
    (η₀ ι εc a : ℝ) (Du : Measure (FreeMonoid α)) (x : Run Ω (FreeMonoid α) (Option (Fin K))) :
    Prop :=
  (∃ B ∈ sched, x ∈ ret O.mq pops none η₀ ι εc a B)
    ∧ ∀ B ∈ sched, x ∈ ret O.mq pops none η₀ ι εc a B →
      Du.real {p | ¬ cutCorrect O B.lo (B.hi + 1) (familyAt O.mq pops x B) p x.1}
          ≤ 513 / 512 * εc
        ∧ Du.real {p | ¬ decided O.mq B.lo (B.hi + 1) (familyAt O.mq pops x B) p x.1} ≤ 2 * ι

variable [IsProbabilityMeasure μ]

/-- `F`'s cut of `p` is fixed by the answers at `p` followed by `F`'s members. -/
lemma measurableSet_cut_decided (O : Oracle μ (FreeMonoid α)) (lo hi : ℕ)
    (F : Finset (FreeMonoid α)) (p : FreeMonoid α) :
    MeasurableSet {ω | ¬ cutCorrect O lo hi F p ω} ∧ MeasurableSet {ω | ¬ decided O.mq lo hi F p ω} := by
  classical
  have hv : ∀ ω ω', (∀ w ∈ F.image (p * ·), (O.mq w ω = 1 ↔ O.mq w ω' = 1)) →
      voteCount O.mq F p ω = voteCount O.mq F p ω' := fun ω ω' h =>
    voteCount_alike O fun v hv => h _ (Finset.mem_image_of_mem _ hv)
  refine ⟨measurableSet_of_reads O (F.image (p * ·)) _ fun ω ω' h hω => ?_,
    measurableSet_of_reads O (F.image (p * ·)) _ fun ω ω' h hω => ?_⟩
  · simp only [Set.mem_ofPred_eq, cutCorrect] at hω ⊢
    rwa [← hv ω ω' h]
  · simp only [Set.mem_ofPred_eq, decided] at hω ⊢
    rwa [← hv ω ω' h]

/-- The misread masses are measurable in the noise. -/
lemma measurable_uniMass (O : Oracle μ (FreeMonoid α)) (lo hi : ℕ) (F : Finset (FreeMonoid α))
    (Du : Measure (FreeMonoid α)) [IsFiniteMeasure Du] :
    Measurable (fun ω => Du.real {p | ¬ cutCorrect O lo hi F p ω})
      ∧ Measurable (fun ω => Du.real {p | ¬ decided O.mq lo hi F p ω}) := by
  refine ⟨?_, ?_⟩
  · exact (measurable_measure_of_mem Du (fun ω => {p | ¬ cutCorrect O lo hi F p ω})
      fun p => (measurableSet_cut_decided O lo hi F p).1).ennreal_toReal
  · exact (measurable_measure_of_mem Du (fun ω => {p | ¬ decided O.mq lo hi F p ω})
      fun p => (measurableSet_cut_decided O lo hi F p).2).ennreal_toReal

theorem measurableSet_uniSpec (O : Oracle μ (FreeMonoid α)) (pops : Finset (Option (Fin K)))
    (sched : Finset State) (η₀ ι εc a : ℝ) (Du : Measure (FreeMonoid α)) [IsFiniteMeasure Du] :
    MeasurableSet {x : Run Ω (FreeMonoid α) (Option (Fin K)) |
      UniSpec O pops sched η₀ ι εc a Du x} := by
  classical
  set Q : State → Finset (FreeMonoid α) → Ω → Prop := fun B F ω =>
    Du.real {p | ¬ cutCorrect O B.lo (B.hi + 1) F p ω} ≤ 513 / 512 * εc
      ∧ Du.real {p | ¬ decided O.mq B.lo (B.hi + 1) F p ω} ≤ 2 * ι
  have hQω : ∀ B F, MeasurableSet {ω | Q B F ω} := fun B F =>
    (measurableSet_le (measurable_uniMass O B.lo (B.hi + 1) F Du).1 measurable_const).inter
      (measurableSet_le (measurable_uniMass O B.lo (B.hi + 1) F Du).2 measurable_const)
  have hQ : ∀ B, MeasurableSet {x : Run Ω (FreeMonoid α) (Option (Fin K)) |
      Q B (familyAt O.mq pops x B) x.1} := fun B => by
    have := measurableSet_of_data (fun x : Run Ω (FreeMonoid α) (Option (Fin K)) =>
      clusterAt O.mq pops x B) (measurableSet_clusterAt O pops B)
      (fun F => Prod.fst ⁻¹' {ω | Q B (insert 1 F) ω}) (fun F => measurable_fst (hQω B _))
    convert this using 1
    ext x
    rfl
  have hset : {x : Run Ω (FreeMonoid α) (Option (Fin K)) | UniSpec O pops sched η₀ ι εc a Du x}
      = (⋃ B ∈ sched, (ret O.mq pops none η₀ ι εc a B : Set _))
        ∩ ⋂ B ∈ sched, ((ret O.mq pops none η₀ ι εc a B : Set _)ᶜ
          ∪ {x | Q B (familyAt O.mq pops x B) x.1}) := by
    ext x
    simp only [UniSpec, Set.mem_ofPred_eq, Set.mem_inter_iff, Set.mem_iUnion, Set.mem_iInter,
      Set.mem_union, Set.mem_compl_iff, exists_prop, Q]
    refine and_congr Iff.rfl (forall₂_congr fun B hB => ?_)
    tauto
  rw [hset]
  exact (Finset.measurableSet_biUnion _ fun B _ => measurableSet_ret O pops none _ _ _ _ B).inter
    (Finset.measurableSet_biInter _ fun B _ =>
      (measurableSet_ret O pops none _ _ _ _ B).compl.union (hQ B))

end OrthoDFA

end
