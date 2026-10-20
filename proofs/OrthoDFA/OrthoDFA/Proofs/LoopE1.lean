import OrthoDFA.Proofs.LoopRoundP

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
    (D' : Measure (FreeMonoid α)) (εcov slack : ℝ) (harvests : List (Measure (FreeMonoid α))) :
    MeasurableSet {x : Run Ω (FreeMonoid α) (Option (Fin K)) |
      ClusterSpec settings O D' εcov slack harvests x} := by
  classical
  set pops := loopPopulations K harvests.length
  set sched := settings.schedule pops
  set Q : State → Finset (FreeMonoid α) → Prop := fun B F =>
    ∫ p, miscutProb O B.lo (B.hi + 1) F p ∂D' ≤ εcov + slack
      ∧ ∀ j ∈ pops, ∫ p, undecidedProb O B.lo (B.hi + 1) F p ∂(loopDistributions D' harvests j)
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
      ClusterSpec settings O D' εcov slack harvests x}
      = (⋃ B ∈ sched, (ret O.mq pops none settings.η₀ settings.indecisionLimit settings.εcov settings.gateLevel B : Set _))
        ∩ ⋂ B ∈ sched, ((ret O.mq pops none settings.η₀ settings.indecisionLimit settings.εcov settings.gateLevel B : Set _)ᶜ
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
        ∩ ⋂ B ∈ sched, ((ret O.mq pops none η₀ ι εc a B : Set _)ᶜ ∪ {x | Q B (familyAt O.mq pops x B)}) := by
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
lemma measurableSet_prefsOf (h : LoopHist α) (Q : Finset (FreeMonoid α)) :
    MeasurableSet {c : ClusterRest α K | prefsOf settings D Dsf h c = Q} :=
  ((measurable_clusterRun (Ω := Unit) (K := K) D Dsf h).comp
    (measurable_const.prodMk measurable_id)) (measurableSet_roundPrefs (Ω := Unit) _ _ Q)

omit [IsProbabilityMeasure μ] in
lemma measurableSet_suffsOf (h : LoopHist α) (Q : Finset (FreeMonoid α)) :
    MeasurableSet {c : ClusterRest α K | suffsOf settings D Dsf h c = Q} := by
  classical
  have hm : ∀ Q', MeasurableSet {c : ClusterRest α K |
      roundSuffs (settings.schedule (loopPopulations K h.pops.length))
        (((), runOf D Dsf h c) : Run Unit (FreeMonoid α) (Option (Fin K))) = Q'} := fun Q' =>
    ((measurable_clusterRun (Ω := Unit) (K := K) D Dsf h).comp
      (measurable_const.prodMk measurable_id)) (measurableSet_roundSuffs (Ω := Unit) _ Q')
  have := measurableSet_of_data (fun c : ClusterRest α K =>
      roundSuffs (settings.schedule (loopPopulations K h.pops.length))
        (((), runOf D Dsf h c) : Run Unit (FreeMonoid α) (Option (Fin K)))) hm
    (fun Q' => if Q'.erase 1 = Q then Set.univ else ∅) (fun Q' => by split_ifs <;> simp)
  convert this using 1
  ext c
  simp only [Set.mem_setOf_eq, suffsOf]
  split_ifs with h'
  · exact iff_of_true h' (Set.mem_univ c)
  · exact iff_of_false h' (Set.notMem_empty c)

omit [IsProbabilityMeasure μ] in
lemma measurableSet_noColl (Ls : ℕ) (c : List (HarvestCode α)) (r : ℕ) (hr : r < K) :
    MeasurableSet {x : Fin K → RoundDraws α K settings.level3.probeBudget |
      NoColl settings D Dsf Ls c x r hr} := by
  classical
  set dat : (Fin K → RoundDraws α K settings.level3.probeBudget) →
      Fin K → Finset (FreeMonoid α) × Finset (FreeMonoid α) := fun x i =>
    (prefsOf settings D Dsf (histAt settings D c i) (x i).1,
      suffsOf settings D Dsf (histAt settings D c i) (x i).1)
  have hdat : ∀ y, MeasurableSet {x | dat x = y} := by
    intro y
    have : {x | dat x = y} = ⋂ i, ((fun x : Fin K → RoundDraws α K settings.level3.probeBudget =>
        (x i).1) ⁻¹' ({c' | prefsOf settings D Dsf (histAt settings D c i) c' = (y i).1}
          ∩ {c' | suffsOf settings D Dsf (histAt settings D c i) c' = (y i).2})) := by
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

instance (T : ℕ) : IsProbabilityMeasure (roundLaw α K T) := by
  unfold roundLaw; infer_instance

instance (T : ℕ) : IsProbabilityMeasure (loopMeasure μ α K T) := by
  unfold loopMeasure; infer_instance

/-- Round `r`'s clustering, where every earlier clustering returned, the earlier rounds read at
most `m` of each population and `ms` of the suffixes, and nothing collides, fails its claim with
chance at most `δc`: off what the earlier rounds read it fails its claim at `slack - m` with that
chance, and a population's average moves by at most `m` off them. -/
theorem loop_cluster_round_le_persist (Ls N : ℕ) (εcov δc slack cap capSf m ms pAP : ℝ)
    (hδc : 0 ≤ δc) (hN : P.L + settings.level3.probeBudget + 2 ≤ N)
    (hm : m ≤ 1 / 2) (hms : ms ≤ 1 / 2)
    (hclust : ∀ pops : Finset (Option (Fin K)), none ∈ pops →
      ∀ (L : Option (Fin K) → Measure (FreeMonoid α)) (Lsf : Measure (FreeMonoid α)),
        (∀ j, IsProbabilityMeasure (L j)) → IsProbabilityMeasure Lsf →
        (∀ j ∈ pops, L j {p | Ls ≤ p.toList.length} = 0) →
        (∀ j ∈ pops, ∀ t, (L j).real {t} ≤ 2 * cap) →
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
    (hLs : P.L < Ls) (hDat : ∀ t, P.D.real {t} ≤ cap) (hDsf : Dsf {v | v.toList.length ≠ Ls} = 0)
    (hpAP : pAP ≤ Dsf.real {v | ∀ p, p * v ∈ O.L ↔ p ∈ O.L}) (hsfat : ∀ t, Dsf.real {t} ≤ capSf)
    (r : ℕ) (hr : r < K) :
    (loopMeasure μ α K settings.level3.probeBudget).real
      {p | HistOK settings O P.D Dsf N p.1 p.2 r
        ∧ (histCodes settings O P.D Dsf N p.1 p.2 r).length = r
        ∧ (∀ ν' ∈ (decodeHist P.D settings.level3.walkStart
            (histCodes settings O P.D Dsf N p.1 p.2 r)).pops,
          (∀ t, ν'.real {t} ≤ cap) ∧ ν' {q | Ls ≤ q.toList.length} = 0)
        ∧ (∀ j ∈ loopPopulations K r, (loopDistributions P.D (decodeHist P.D
            settings.level3.walkStart (histCodes settings O P.D Dsf N p.1 p.2 r)).pops j).real
              ↑(decodeHist P.D settings.level3.walkStart
                (histCodes settings O P.D Dsf N p.1 p.2 r)).prefs ≤ m)
        ∧ Dsf.real ↑(decodeHist P.D settings.level3.walkStart
            (histCodes settings O P.D Dsf N p.1 p.2 r)).suffs ≤ ms
        ∧ ¬ ClusterSpec settings O P.D εcov slack (decodeHist P.D settings.level3.walkStart
            (histCodes settings O P.D Dsf N p.1 p.2 r)).pops
          (clusterRun P.D Dsf (decodeHist P.D settings.level3.walkStart
            (histCodes settings O P.D Dsf N p.1 p.2 r)) (p.1, (p.2 ⟨r, hr⟩).1))
        ∧ NoColl settings P.D Dsf Ls (histCodes settings O P.D Dsf N p.1 p.2 r) p.2 r hr}
      ≤ δc := by
  classical
  set T := settings.level3.probeBudget
  set dec := decodeHist P.D settings.level3.walkStart with hdec
  obtain ⟨ω₀⟩ := nonempty_of_isProbabilityMeasure μ
  set ν := roundLaw α K T
  have hσ : MeasurePreserving (fun d : RoundDraws α K T => ((() : Unit), d)) ν
      ((Measure.dirac ()).prod ν) :=
    ⟨measurable_const.prodMk measurable_id, by rw [Measure.dirac_prod]⟩
  set Y := Option (List (HarvestCode α))
  set W : Y → (Fin K → RoundDraws α K T) → Finset (FreeMonoid α) := fun y x => match y with
    | some c => histReads settings P.D Dsf N c x r ω₀
    | none => ∅
  set G : Y → Prop := fun y => ∃ c, y = some c ∧ c.length = r
  set Good : List (HarvestCode α) → Prop := fun c => c.length = r
    ∧ (∀ ν' ∈ (dec c).pops, (∀ t, ν'.real {t} ≤ cap) ∧ ν' {q | Ls ≤ q.toList.length} = 0)
    ∧ (∀ j ∈ loopPopulations K r, (loopDistributions P.D (dec c).pops j).real ↑(dec c).prefs ≤ m)
    ∧ Dsf.real ↑(dec c).suffs ≤ ms
  set Bad : Y → Set (Ω × RoundDraws α K T) := fun y => match y with
    | some c => {q | Good c
        ∧ ¬ ClusterSpec settings O P.D εcov slack (dec c).pops
          (clusterRun P.D Dsf (dec c) (q.1, q.2.1))}
    | none => ∅
  set Z : Y → RoundDraws α K T → Finset (FreeMonoid α) := fun y d => match y with
    | some c => schedReads (settings.schedule (loopPopulations K (dec c).pops.length))
        (loopPopulations K (dec c).pops.length) (clusterRun P.D Dsf (dec c) (ω₀, d.1))
    | none => ∅
  set Dj : Y → Set (Fin K → RoundDraws α K T) := fun y => match y with
    | some c => {x | NoColl settings P.D Dsf Ls c x r hr}
    | none => Set.univ
  have hcl : ∀ c : List (HarvestCode α), Measurable (fun q : Ω × RoundDraws α K T =>
      clusterRun P.D Dsf (dec c) ((q.1, q.2.1) : ClusterDraws Ω α K)) := fun c =>
    (measurable_clusterRun P.D Dsf (dec c)).comp
      (measurable_fst.prodMk (measurable_fst.comp measurable_snd))
  have hmain := persist_stage_le O ν (Measure.dirac ()) ν (fun d => ((() : Unit), d)) hσ Prod.snd
    measurable_snd (fun w => Prod.ext rfl rfl) ⟨r, hr⟩ (stageH settings O P.D Dsf N r)
    (fun ω x x' hxx' _ => stageH_congr_draws settings O P.D Dsf N r ω x x' fun i hi => hxx' i hi)
    (measurableSet_stageH settings O P.D Dsf N P.length_le hN r) W G
    (fun y x ω ω' hG hs h => by
      obtain ⟨c, rfl, -⟩ := hG
      exact stageH_stop settings O P.D Dsf N P.length_le hN r c x ω ω' ω₀ hs h)
    Bad (fun y hy => by
      rcases y with _ | c
      · rfl
      · exact Set.eq_empty_of_forall_notMem fun q hq => hy ⟨c, rfl, hq.1.1⟩)
    (fun y => by
      rcases y with _ | c
      · exact MeasurableSet.empty
      · exact (MeasurableSet.const _).inter
          (hcl c (measurableSet_clusterSpec settings O P.D εcov slack (dec c).pops)).compl)
    Z (fun y d ω ω' h hq => by
      rcases y with _ | c
      · exact hq
      · refine ⟨hq.1, fun hs => hq.2 ?_⟩
        exact (clusterSpec_congr settings O P.D P.D Dsf εcov slack (dec c) d.1 h).2 hs)
    Dj (fun y => by
      rcases y with _ | c
      · exact MeasurableSet.univ
      · exact measurableSet_noColl settings P.D Dsf Ls c r hr)
    (fun y x hG hx => by
      obtain ⟨c, rfl, hcr⟩ := hG
      have e : histAt settings P.D c r = dec c := by
        simp only [histAt, List.take_of_length_le (le_of_eq hcr), hdec]
      have := disjoint_hist_round settings P.D Dsf N Ls c x r hr ω₀ hx
      rw [e] at this
      exact this)
    (ENNReal.ofReal δc) (fun y => by
      rcases y with _ | c
      · show (μ.prod ν) ∅ ≤ _
        simp
      · by_cases hH : Good c
        swap
        · have : Bad (some c) = ∅ := Set.eq_empty_of_forall_notMem fun q hq => hH hq.1
          rw [this, measure_empty]; exact zero_le
        obtain ⟨hcr, hpg, hpm, hsm⟩ := hH
        set h := dec c
        have hpops := decodePop_prob settings P.D c
        have : ∀ j, IsProbabilityMeasure (loopDistributions (K := K) P.D h.pops j) :=
          isProbabilityMeasure_loopDistributions P.D h.pops hpops
        have hlen : h.pops.length = r := by
          simp only [h, dec, decodeHist, List.length_map, hcr]
        have hgetD : ∀ i : Fin K, some i ∈ loopPopulations K h.pops.length →
            h.pops.getD i (Measure.dirac 1) ∈ h.pops := by
          intro i hi
          have := lt_of_some_mem_loopPops hi
          rw [List.getD_eq_getElem (hn := this)]
          exact List.getElem_mem _
        set pops := loopPopulations K h.pops.length
        have hpm' : ∀ j ∈ pops, (loopDistributions P.D h.pops j).real ↑h.prefs ≤ m := fun j hj =>
          hpm j (by rw [← hlen]; exact hj)
        have hsupp : ∀ j ∈ pops,
            loopDistributions P.D h.pops j {p | Ls ≤ p.toList.length} = 0 := by
          intro j hj
          cases j with
          | none =>
            refine measure_mono_null (fun x hx => ?_) (ae_iff.1 P.length_le)
            simp only [Set.mem_ofPred_eq, not_le] at hx ⊢
            omega
          | some i => exact (hpg _ (hgetD i hj)).2
        have hat : ∀ j ∈ pops, ∀ t, (loopDistributions P.D h.pops j).real {t} ≤ cap := by
          intro j hj t
          cases j with
          | none => exact hDat t
          | some i => exact (hpg _ (hgetD i hj)).1 t
        set L : Option (Fin K) → Measure (FreeMonoid α) := fun j => popOff P.D h.pops j h.prefs
        have hLj : ∀ j ∈ pops, L j = offLaw (loopDistributions P.D h.pops j) h.prefs := fun j hj =>
          popOff_of_ne (compl_ne_zero ((hpm' j hj).trans hm))
        have hsm' : Dsf.real ↑h.suffs ≤ 1 / 2 := hsm.trans hms
        have hc := hclust pops (Finset.mem_insert_self _ _) L (offLaw Dsf h.suffs)
          (fun j => inferInstance) inferInstance
          (fun j hj => by
            rw [hLj j hj]
            exact offLaw_null (compl_ne_zero ((hpm' j hj).trans hm)) (hsupp j hj))
          (fun j hj t => by
            rw [hLj j hj]
            exact (offLaw_atom_le ((hpm' j hj).trans hm) t).trans (by linarith [hat j hj t]))
          (offLaw_null (compl_ne_zero hsm') hDsf)
          (by linarith [le_offLaw_real hsm' {v : FreeMonoid α | ∀ p, p * v ∈ O.L ↔ p ∈ O.L}])
          (fun t => (offLaw_atom_le hsm' t).trans (by linarith [hsfat t]))
        have hcr' := measurePreserving_clusterDraws (μ := μ) (K := K) P.D Dsf h
        set Tset := {x : Run Ω (FreeMonoid α) (Option (Fin K)) |
          ¬ ClusterSpec settings O P.D εcov slack h.pops x}
        have hTm : MeasurableSet (streams (Ω := Ω) (fun _ => ∅) none ⁻¹' Tset) :=
          measurable_streams _ none
            (measurableSet_clusterSpec settings O P.D εcov slack h.pops).compl
        have hmarg : MeasurePreserving (fun q : Ω × RoundDraws α K T =>
            ((q.1, q.2.1) : ClusterDraws Ω α K)) (μ.prod ν) (clusterLaw μ α K) :=
          (MeasurePreserving.id μ).prod measurePreserving_fst
        have hsub : streams (Ω := Ω) (fun _ => ∅) none ⁻¹' Tset
            ⊆ {x | ¬ ((∃ B ∈ settings.schedule pops, streams (fun _ => ∅) none x
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
                  ∂(L j) ≤ 2 * settings.indecisionLimit + (slack - m))} := by
          intro y hx hs
          set x := streams (fun _ => ∅) none y
          apply hx
          refine ⟨hs.1, fun B hB hret => ⟨?_, fun j hj => ?_⟩⟩
          · have h1 := integral_le_offLaw (loopDistributions (K := K) P.D h.pops none) h.prefs
              (f := miscutProb O B.lo (B.hi + 1) (familyAt O.mq pops x B))
              (fun p => measureReal_nonneg) (fun p => measureReal_le_one)
            have h2 := (hs.2 B hB hret).1
            rw [hLj none (Finset.mem_insert_self _ _)] at h2
            have h3 := hpm' none (Finset.mem_insert_self _ _)
            show ∫ p, miscutProb O B.lo (B.hi + 1) (familyAt O.mq pops x B) p
              ∂(loopDistributions (K := K) P.D h.pops none) ≤ εcov + slack
            linarith
          · have h1 := integral_le_offLaw (loopDistributions P.D h.pops j) h.prefs
              (f := undecidedProb O B.lo (B.hi + 1) (familyAt O.mq pops x B))
              (fun p => measureReal_nonneg) (fun p => measureReal_le_one)
            have h2 := (hs.2 B hB hret).2 j hj
            rw [hLj j hj] at h2
            have h3 := hpm' j hj
            linarith
        calc (μ.prod ν) (Bad (some c))
            ≤ (μ.prod ν) ((fun q : Ω × RoundDraws α K T => ((q.1, q.2.1) : ClusterDraws Ω α K)) ⁻¹'
                (clusterDraws P.D Dsf h ⁻¹' (streams (fun _ => ∅) none ⁻¹' Tset))) :=
              measure_mono fun q hq => hq.2
          _ = clusterLaw μ α K (clusterDraws P.D Dsf h ⁻¹' (streams (fun _ => ∅) none ⁻¹' Tset)) :=
              hmarg.measure_preimage (hcr'.measurable hTm).nullMeasurableSet
          _ = runMeasure μ L (offLaw Dsf h.suffs) (streams (fun _ => ∅) none ⁻¹' Tset) :=
              hcr'.measure_preimage hTm.nullMeasurableSet
          _ ≤ runMeasure μ L (offLaw Dsf h.suffs)
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
                  ∂(L j) ≤ 2 * settings.indecisionLimit + (slack - m))} :=
              measure_mono hsub
          _ ≤ ENNReal.ofReal δc := by
              rw [← ofReal_measureReal]
              exact ENNReal.ofReal_le_ofReal hc)
  refine ENNReal.toReal_le_of_le_ofReal hδc ((measure_mono ?_).trans hmain)
  rintro p ⟨hok, hlen, hgood, hpm, hsm, hspec, hnc⟩
  have hst : stageH settings O P.D Dsf N r p.1 p.2
      = some (histCodes settings O P.D Dsf N p.1 p.2 r) := by
    simp only [stageH, if_pos hok]
  show (p.1, p.2 ⟨r, hr⟩) ∈ Bad (stageH settings O P.D Dsf N r p.1 p.2)
    ∧ p.2 ∈ Dj (stageH settings O P.D Dsf N r p.1 p.2)
  rw [hst]
  exact ⟨⟨⟨hlen, hgood, hpm, hsm⟩, hspec⟩, hnc⟩

end Round

end OrthoDFA
