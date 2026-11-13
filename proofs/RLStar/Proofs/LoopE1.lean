import RLStar.Proofs.LoopRoundP

/-!
# A round's clustering on one noise

Round `r`'s clustering reads, off a collision, only strings no earlier round read. So given the
earlier rounds it fails its claim with at most the chance it has on fresh noise
(`loop_cluster_round_le_persist`).
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α] {Ω : Type*} [MeasurableSpace Ω]
  {μ : Measure Ω} [IsProbabilityMeasure μ] {K : ℕ}

theorem measurableSet_clusterSpec (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α))
    (D' : Measure (FreeMonoid α)) (εcov slack : ℝ) (samplers : List (List (FreeMonoid α))) :
    MeasurableSet {x : Run Ω (FreeMonoid α) (Option (Fin K)) |
      ClusterSpec settings O D' εcov slack samplers x} := by
  classical
  set pops := loopPopulations K samplers.length
  set sched := settings.schedule pops
  set Q : State → Finset (FreeMonoid α) → Prop := fun B F =>
    ∫ p, miscutProb O B.lo (B.hi + 1) F p ∂D' ≤ εcov + slack
      ∧ ∀ j ∈ pops, ∫ p, undecidedProb O B.lo (B.hi + 1) F p ∂(loopDistributions D' samplers j)
        ≤ 2 * settings.indecisionLimit + slack
  have hQ : ∀ B, MeasurableSet {x : Run Ω (FreeMonoid α) (Option (Fin K)) |
      Q B (familyAt O.mq pops x B)} := fun B => by
    have := measurableSet_of_data (fun x : Run Ω (FreeMonoid α) (Option (Fin K)) =>
      clusterAt O.mq pops x B) (measurableSet_clusterAt O pops B)
      (fun F => if Q B (insert 1 F) then Set.univ else ∅) (fun F => by split_ifs <;> simp)
    convert this using 1
    ext x
    simp only [Set.mem_setOf_eq, familyAt]
    split_ifs with h <;> simp [h]
  have hset : {x : Run Ω (FreeMonoid α) (Option (Fin K)) |
      ClusterSpec settings O D' εcov slack samplers x}
      = (⋃ B ∈ sched, (ret O.mq pops none settings.η₀ settings.indecisionLimit settings.εcov
          settings.gateLevel B : Set _))
        ∩ ⋂ B ∈ sched, ((ret O.mq pops none settings.η₀ settings.indecisionLimit settings.εcov
            settings.gateLevel B : Set _)ᶜ
          ∪ {x | Q B (familyAt O.mq pops x B)}) := by
    ext x
    simp only [ClusterSpec, Set.mem_setOf_eq, Set.mem_inter_iff, Set.mem_iUnion, Set.mem_iInter,
      Set.mem_union, Set.mem_compl_iff, exists_prop]
    refine and_congr Iff.rfl (forall₂_congr fun B hB => ?_)
    tauto
  rw [hset]
  exact (Finset.measurableSet_biUnion _ fun B _ => measurableSet_ret O pops none _ _ _ _ B).inter
    (Finset.measurableSet_biInter _ fun B _ =>
      (measurableSet_ret O pops none _ _ _ _ B).compl.union (hQ B))

theorem measurableSet_specAt (O : Oracle μ (FreeMonoid α)) (pops : Finset (Option (Fin K)))
    (sched : Finset State) (η₀ ι εc a : ℝ) (Q : State → Finset (FreeMonoid α) → Prop) :
    MeasurableSet {x : Run Ω (FreeMonoid α) (Option (Fin K)) |
      (∃ B ∈ sched, x ∈ ret O.mq pops none η₀ ι εc a B)
        ∧ ∀ B ∈ sched, x ∈ ret O.mq pops none η₀ ι εc a B → Q B (familyAt O.mq pops x B)} := by
  classical
  have hQ : ∀ B, MeasurableSet {x : Run Ω (FreeMonoid α) (Option (Fin K)) |
      Q B (familyAt O.mq pops x B)} := fun B => by
    have := measurableSet_of_data (fun x : Run Ω (FreeMonoid α) (Option (Fin K)) =>
      clusterAt O.mq pops x B) (measurableSet_clusterAt O pops B)
      (fun F => if Q B (insert 1 F) then Set.univ else ∅) (fun F => by split_ifs <;> simp)
    convert this using 1
    ext x
    simp only [Set.mem_setOf_eq, familyAt]
    split_ifs with h <;> simp [h]
  have hset : {x : Run Ω (FreeMonoid α) (Option (Fin K)) |
      (∃ B ∈ sched, x ∈ ret O.mq pops none η₀ ι εc a B)
        ∧ ∀ B ∈ sched, x ∈ ret O.mq pops none η₀ ι εc a B → Q B (familyAt O.mq pops x B)}
      = (⋃ B ∈ sched, (ret O.mq pops none η₀ ι εc a B : Set _))
        ∩ ⋂ B ∈ sched, ((ret O.mq pops none η₀ ι εc a B : Set _)ᶜ ∪ {x | Q B (familyAt O.mq pops x
            B)}) := by
    ext x
    simp only [Set.mem_setOf_eq, Set.mem_inter_iff, Set.mem_iUnion, Set.mem_iInter,
      Set.mem_union, Set.mem_compl_iff, exists_prop]
    refine and_congr Iff.rfl (forall₂_congr fun B hB => ?_)
    tauto
  rw [hset]
  exact (Finset.measurableSet_biUnion _ fun B _ => measurableSet_ret O pops none _ _ _ _ B).inter
    (Finset.measurableSet_biInter _ fun B _ =>
      (measurableSet_ret O pops none _ _ _ _ B).compl.union (hQ B))

section Draws

variable (settings : LoopSettings K) (D Dsf : Measure (FreeMonoid α))

omit [IsProbabilityMeasure μ] in
lemma measurableSet_prefsOf [IsProbabilityMeasure D] (h : SpecHist α) (Q : Finset (FreeMonoid α)) :
    MeasurableSet {c : ClusterRest α K | prefsOf settings D Dsf h c =
        Q} :=
  ((measurable_idealRun (Ω := Unit) (K := K) D Dsf h).comp
    (measurable_const.prodMk measurable_id)) (measurableSet_roundPrefs (Ω := Unit) _ _ Q)

omit [IsProbabilityMeasure μ] in
lemma measurableSet_suffsOf [IsProbabilityMeasure D] (h : SpecHist α) (Q : Finset (FreeMonoid α)) :
    MeasurableSet {c : ClusterRest α K | suffsOf settings D Dsf h c =
        Q} := by
  classical
  have hm : ∀ Q', MeasurableSet {c : ClusterRest α K |
      roundSuffs (settings.schedule (loopPopulations K h.pops.length))
        (((), runOf settings D Dsf h c) : Run Unit (FreeMonoid α) (Option (Fin K))) = Q'} := fun
            Q' =>
    ((measurable_idealRun (Ω := Unit) (K := K) D Dsf h).comp
      (measurable_const.prodMk measurable_id)) (measurableSet_roundSuffs (Ω := Unit) _ Q')
  have := measurableSet_of_data (fun c : ClusterRest α K =>
      roundSuffs (settings.schedule (loopPopulations K h.pops.length))
        (((), runOf settings D Dsf h c) : Run Unit (FreeMonoid α) (Option (Fin K)))) hm
    (fun Q' => if Q'.erase 1 = Q then Set.univ else ∅) (fun Q' => by split_ifs <;> simp)
  convert this using 1
  ext c
  simp only [Set.mem_setOf_eq, suffsOf]
  split_ifs with h'
  · exact iff_of_true h' (Set.mem_univ c)
  · exact iff_of_false h' (Set.notMem_empty c)

omit [IsProbabilityMeasure μ] in
lemma measurableSet_noColl [IsProbabilityMeasure D] (Ls : ℕ) (c : List (HarvestCode α)) (r : ℕ) (hr
    : r < K) :
    MeasurableSet {x : Fin K → settings.RoundDraws α |
      NoColl settings D Dsf Ls c x r hr} := by
  classical
  set dat : (Fin K → settings.RoundDraws α) →
      Fin K → Finset (FreeMonoid α) × Finset (FreeMonoid α) := fun x i =>
    (prefsOf settings D Dsf (histAt c i) (x i).1,
      suffsOf settings D Dsf (histAt c i) (x i).1)
  have hdat : ∀ y, MeasurableSet {x | dat x = y} := by
    intro y
    have : {x | dat x = y} = ⋂ i, ((fun x : Fin K → settings.RoundDraws α =>
        (x i).1) ⁻¹' ({c' | prefsOf settings D Dsf (histAt c i) c' = (y i).1}
          ∩ {c' | suffsOf settings D Dsf (histAt c i) c' = (y i).2})) := by
      ext x
      simp only [Set.mem_setOf_eq, Set.mem_iInter, Set.mem_preimage, Set.mem_inter_iff, dat,
        funext_iff, Prod.ext_iff]
    rw [this]
    exact MeasurableSet.iInter fun i => (measurable_fst.comp (measurable_pi_apply i))
      ((measurableSet_prefsOf settings D Dsf _ _).inter (measurableSet_suffsOf settings D Dsf _ _))
  set Φ : (Fin K → Finset (FreeMonoid α) × Finset (FreeMonoid α)) → Prop := fun y =>
    (∀ i : Fin K, i.1 ≤ r → ∀ v ∈ (y i).2, v.toList.length = Ls)
      ∧ (∀ i : Fin K, i.1 ≤ r → ∀ p ∈ (y i).1, p.toList.length < Ls)
      ∧ (∀ i : Fin K, i.1 < r → Disjoint (y i).2 (y ⟨r, hr⟩).2)
      ∧ (∀ i : Fin K, i.1 < r → Disjoint (y i).1 (y ⟨r, hr⟩).1)
  have := measurableSet_of_data dat hdat (fun y => if Φ y then Set.univ else ∅)
    (fun y => by split_ifs <;> simp)
  convert this using 1
  ext x
  simp only [Set.mem_setOf_eq]
  split_ifs with h'
  · exact iff_of_true h' (Set.mem_univ x)
  · exact iff_of_false h' (Set.notMem_empty x)

end Draws

section Round

variable (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α)) (P : PrefixDistribution α)
  (Dsf : Measure (FreeMonoid α)) [IsProbabilityMeasure Dsf]

instance : IsProbabilityMeasure (restLaw α K) := by unfold restLaw; infer_instance

instance : IsProbabilityMeasure (settings.roundLaw α) := by
  unfold LoopSettings.roundLaw lazyLaw; infer_instance

instance : IsProbabilityMeasure (settings.measure μ α) := by
  unfold LoopSettings.measure; infer_instance

/-- Round `r`'s clustering, where every earlier clustering returned, the earlier rounds read at
most `m` of each population and `ms` of the suffixes, and nothing collides, fails its claim with
chance at most `δc`: off what the earlier rounds read it fails its claim at `slack - m` with that
chance, and a population's average moves by at most `m` off them. -/
theorem loop_cluster_round_le_persist (Ls : ℕ) (εcov δc slack cap capSf m ms pAP : ℝ)
    (hδc : 0 ≤ δc)
    (hm : m ≤ 1 / 2) (hms : ms ≤ 1 / 2)
    (hclust : ∀ pops : Finset (Option (Fin K)), none ∈ pops →
      ∀ (L : Option (Fin K) → Measure (FreeMonoid α)) (Lsf : Measure (FreeMonoid α)),
        (∀ j, IsProbabilityMeasure (L j)) → IsProbabilityMeasure Lsf →
        (∀ j ∈ pops, L j {p | Ls ≤ p.toList.length} = 0) →
        (∀ j ∈ pops, ∀ t, (L j).real {t} ≤ 2 * cap) →
        2 * settings.εcov ≤ (L none).real {p | p ∈ O.L} →
        2 * settings.εcov ≤ (L none).real {p | p ∉ O.L} →
        Lsf {v | v.toList.length ≠ Ls} = 0 →
        pAP - ms ≤ Lsf.real {v | ∀ p, p * v ∈ O.L ↔ p ∈ O.L} →
        (∀ t, Lsf.real {t} ≤ 2 * capSf) →
        (runMeasure μ L Lsf).real
          {x | ¬ ((∃ B ∈ settings.schedule pops, streams (fun _ => ∅) none x
                ∈ ret O.mq pops none settings.η₀ settings.indecisionLimit settings.εcov
                  settings.gateLevel B)
            ∧ ∀ B ∈ settings.schedule pops, streams (fun _ => ∅) none x
              ∈ ret O.mq pops none settings.η₀ settings.indecisionLimit settings.εcov
                settings.gateLevel B →
              ∫ p, miscutProb O B.lo (B.hi + 1)
                  (familyAt O.mq pops (streams (fun _ => ∅) none x) B) p ∂(L none)
                  ≤ εcov + (slack - m)
                ∧ ∀ j ∈ pops, ∫ p, undecidedProb O B.lo (B.hi + 1)
                  (familyAt O.mq pops (streams (fun _ => ∅) none x) B) p
                  ∂(L j) ≤ 2 * settings.indecisionLimit + (slack - m))} ≤ δc)
    (hLs : P.L < Ls) (hDat : ∀ t, P.D.real {t} ≤ cap)
    (hbalL : 2 * settings.εcov + m ≤ P.D.real {p | p ∈ O.L})
    (hbalN : 2 * settings.εcov + m ≤ P.D.real {p | p ∉ O.L})
    (hDsf : Dsf {v | v.toList.length ≠ Ls} = 0)
    (hpAP : pAP ≤ Dsf.real {v | ∀ p, p * v ∈ O.L ↔ p ∈ O.L}) (hsfat : ∀ t, Dsf.real {t} ≤ capSf)
    (r : ℕ) (hr : r < K) :
    (settings.measure μ α).real
      {p | HistOK settings O P.D Dsf p.1 p.2 r
        ∧ (histCodes settings O P.D Dsf p.1 p.2 r).length = r
        ∧ (∀ ℓ ∈ (decodeHist (histCodes settings O P.D Dsf p.1 p.2 r)).pops,
          (∀ t, (listLaw ℓ).real {t} ≤ cap) ∧ listLaw ℓ {q | Ls ≤ q.toList.length} = 0)
        ∧ (∀ j ∈ loopPopulations K r, (loopDistributions P.D (decodeHist
            (histCodes settings O P.D Dsf p.1 p.2 r)).pops j).real
              ↑(decodeHist
                (histCodes settings O P.D Dsf p.1 p.2 r)).prefs ≤ m)
        ∧ Dsf.real ↑(decodeHist
            (histCodes settings O P.D Dsf p.1 p.2 r)).suffs ≤ ms
        ∧ ¬ ClusterSpec settings O P.D εcov slack (decodeHist
            (histCodes settings O P.D Dsf p.1 p.2 r)).pops
          (idealRun P.D Dsf (decodeHist
            (histCodes settings O P.D Dsf p.1 p.2 r)) (p.1, (p.2 ⟨r, hr⟩).1))
        ∧ NoColl settings P.D Dsf Ls (histCodes settings O P.D Dsf p.1 p.2 r) p.2 r hr}
      ≤ δc := by
  sorry -- closes by: main's proof with settings.roundLaw; list atoms for samplerLaws

end Round

end OrthoDFA
