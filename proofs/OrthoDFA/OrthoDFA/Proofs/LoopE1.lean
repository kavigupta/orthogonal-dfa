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
      = (⋃ B ∈ sched, (ret O.mq pops none settings.indecisionLimit settings.gateLevel B : Set _))
        ∩ ⋂ B ∈ sched, ((ret O.mq pops none settings.indecisionLimit settings.gateLevel B : Set _)ᶜ
          ∪ {x | Q B (familyAt O.mq pops x B)}) := by
    ext x
    simp only [ClusterSpec, Set.mem_setOf_eq, Set.mem_inter_iff, Set.mem_iUnion, Set.mem_iInter,
      Set.mem_union, Set.mem_compl_iff, exists_prop]
    refine and_congr Iff.rfl (forall₂_congr fun B hB => ?_)
    tauto
  rw [hset]
  exact (Finset.measurableSet_biUnion _ fun B _ => measurableSet_ret O pops none _ _ B).inter
    (Finset.measurableSet_biInter _ fun B _ =>
      (measurableSet_ret O pops none _ _ B).compl.union (hQ B))

section Draws

variable (settings : LoopSettings K) (D : Measure (FreeMonoid α))

omit [MeasurableSpace Ω] [IsProbabilityMeasure μ] in
lemma measurable_prefDraw (harvests : List (Measure (FreeMonoid α)))
    (hpops : ∀ ν ∈ harvests, IsProbabilityMeasure ν) (j : Option (Fin K)) (a : ℕ) (b : Bool) :
    Measurable fun c : ClusterRest α K => prefDraw harvests c j a b := by
  have hrun : Measurable fun c : ClusterRest α K =>
      (clusterRun (Ω := Unit) harvests ((), c) : Run Unit (FreeMonoid α) (Option (Fin K))) :=
    (measurePreserving_clusterRun (μ := Measure.dirac ()) (K := K) (Measure.dirac (1 : FreeMonoid α))
      (Measure.dirac 1) harvests hpops).measurable.comp (measurable_const.prodMk measurable_id)
  unfold prefDraw runOf
  cases b
  · exact (measurable_prf j a).comp hrun
  · exact (measurable_cert j a).comp hrun

lemma measurableSet_noCollD (Ls NS NP : ℕ) (c : List (HarvestCode α)) (r : ℕ) (hr : r < K) :
    MeasurableSet {x : Fin K → RoundDraws α K settings.level4.probeBudget |
      NoCollD settings D Ls NS NP c x r hr} := by
  classical
  have hpops : ∀ i : ℕ, ∀ ν ∈ popsAt settings D c i, IsProbabilityMeasure ν := fun i ν h => by
    obtain ⟨c', -, rfl⟩ := List.mem_map.1 h
    unfold decodePop
    infer_instance
  have hsf : ∀ (i : Fin K) (a : ℕ), Measurable fun x : Fin K → RoundDraws α K
      settings.level4.probeBudget => (x i).1.1 a :=
    fun i a => (measurable_pi_apply a).comp (measurable_fst.comp (measurable_fst.comp
      (measurable_pi_apply i)))
  have hpd : ∀ (i : Fin K) (h : List (Measure (FreeMonoid α))), (∀ ν ∈ h, IsProbabilityMeasure ν) →
      ∀ j a b, Measurable fun x : Fin K → RoundDraws α K settings.level4.probeBudget =>
        prefDraw h (x i).1 j a b :=
    fun i h hh j a b => (measurable_prefDraw h hh j a b).comp (measurable_fst.comp
      (measurable_pi_apply i))
  have hone : ∀ {f : (Fin K → RoundDraws α K settings.level4.probeBudget) → FreeMonoid α},
      Measurable f → ∀ P : FreeMonoid α → Prop, Measurable fun x => P (f x) :=
    fun hf P => (measurable_from_top (f := P)).comp hf
  have htwo : ∀ {f g : (Fin K → RoundDraws α K settings.level4.probeBudget) → FreeMonoid α},
      Measurable f → Measurable g → Measurable fun x => f x ≠ g x :=
    fun hf hg => (measurable_of_countable (fun q : FreeMonoid α × FreeMonoid α => q.1 ≠ q.2)).comp
      (hf.prodMk hg)
  unfold NoCollD
  refine measurableSet_setOfPred.2 (Measurable.and ?_ (Measurable.and ?_ (Measurable.and ?_ ?_)))
  · exact Measurable.forall fun i => measurable_const.imp (Measurable.forall fun a =>
      measurable_const.imp (hone (hsf i a) (fun v => v.toList.length = Ls)))
  · exact Measurable.forall fun i => measurable_const.imp (Measurable.forall fun j =>
      Measurable.forall fun a => Measurable.forall fun b => measurable_const.imp
        (hone (hpd i _ (hpops i) j a b) (fun v => v.toList.length < Ls)))
  · exact Measurable.forall fun i => measurable_const.imp (Measurable.forall fun a =>
      measurable_const.imp (Measurable.forall fun a' => measurable_const.imp
        (htwo (hsf i a) (hsf ⟨r, hr⟩ a'))))
  · exact Measurable.forall fun i => measurable_const.imp (Measurable.forall fun j =>
      Measurable.forall fun a => Measurable.forall fun b => measurable_const.imp
        (Measurable.forall fun j' => Measurable.forall fun a' => Measurable.forall fun b' =>
          measurable_const.imp (htwo (hpd i _ (hpops i) j a b)
            (hpd ⟨r, hr⟩ _ (hpops r) j' a' b'))))

end Draws

section Round

variable (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α)) (P : PrefixDistribution α)
  (Dsf : Measure (FreeMonoid α)) [IsProbabilityMeasure Dsf]

instance : IsProbabilityMeasure (restLaw P.D Dsf K) := by unfold restLaw; infer_instance

instance (T : ℕ) : IsProbabilityMeasure (roundLaw P.D Dsf K T) := by
  unfold roundLaw; infer_instance

instance (T : ℕ) : IsProbabilityMeasure (loopMeasure μ P.D Dsf K T) := by
  unfold loopMeasure; infer_instance

/-- Round `r`'s clustering, where every earlier clustering returned and nothing collides, fails
its claim with chance at most `δc`. -/
theorem loop_cluster_round_le_persist (Ls NS NP N : ℕ) (εcov δc slack cap : ℝ) (hδc : 0 ≤ δc)
    (hC : 2 ≤ settings.level4.maxLeaves) (hN : P.L + settings.level4.maxLeaves ≤ N)
    (hNS : ∀ pops, schedNS (settings.schedule pops) ≤ NS)
    (hNP : ∀ pops, schedNP (settings.schedule pops) ≤ NP)
    (hclust : ∀ pops : Finset (Option (Fin K)), none ∈ pops →
      ∀ (D : Option (Fin K) → Measure (FreeMonoid α)),
        (∀ j, IsProbabilityMeasure (D j)) →
        (∀ j ∈ pops, D j {p | Ls ≤ p.toList.length} = 0) →
        (∀ j ∈ pops, collisionMass (D j) ≤ cap) →
        (runMeasure μ D Dsf).real
          {x | ¬ ((∃ B ∈ settings.schedule pops,
                x ∈ ret O.mq pops none settings.indecisionLimit settings.gateLevel B)
            ∧ ∀ B ∈ settings.schedule pops,
              x ∈ ret O.mq pops none settings.indecisionLimit settings.gateLevel B →
              ∫ p, miscutProb O B.lo (B.hi + 1) (familyAt O.mq pops x B) p ∂(D none)
                  ≤ εcov + slack
                ∧ ∀ j ∈ pops, ∫ p, undecidedProb O B.lo (B.hi + 1) (familyAt O.mq pops x B) p
                  ∂(D j) ≤ 2 * settings.indecisionLimit + slack)} ≤ δc)
    (hLs : P.L < Ls) (hcolD : collisionMass P.D ≤ cap) (r : ℕ) (hr : r < K) :
    (loopMeasure μ P.D Dsf K settings.level4.probeBudget).real
      {p | HistOK settings O P.D N p.1 p.2 r
        ∧ (histCodes settings O P.D N p.1 p.2 r).length = r
        ∧ (∀ ν' ∈ (histCodes settings O P.D N p.1 p.2 r).map
            (decodePop P.D settings.level4.walkStart),
          collisionMass ν' ≤ cap ∧ ν' {q | Ls ≤ q.toList.length} = 0)
        ∧ ¬ ClusterSpec settings O P.D εcov slack ((histCodes settings O P.D N p.1 p.2 r).map
            (decodePop P.D settings.level4.walkStart))
          (clusterRun ((histCodes settings O P.D N p.1 p.2 r).map
            (decodePop P.D settings.level4.walkStart)) (p.1, (p.2 ⟨r, hr⟩).1))
        ∧ NoCollD settings P.D Ls NS NP (histCodes settings O P.D N p.1 p.2 r) p.2 r hr} ≤ δc := by
  classical
  set T := settings.level4.probeBudget
  set dec := decodePop P.D settings.level4.walkStart with hdec
  obtain ⟨ω₀⟩ := nonempty_of_isProbabilityMeasure μ
  set ν := roundLaw P.D Dsf K T
  have hσ : MeasurePreserving (fun d : RoundDraws α K T => ((() : Unit), d)) ν
      ((Measure.dirac ()).prod ν) :=
    ⟨measurable_const.prodMk measurable_id, by rw [Measure.dirac_prod]⟩
  set Y := Option (List (HarvestCode α))
  set W : Y → (Fin K → RoundDraws α K T) → Finset (FreeMonoid α) := fun y x => match y with
    | some c => histReads settings P.D N c x r ω₀
    | none => ∅
  set G : Y → Prop := fun y => ∃ c, y = some c ∧ c.length = r
  set Bad : Y → Set (Ω × RoundDraws α K T) := fun y => match y with
    | some c => {q | c.length = r ∧ (∀ ν' ∈ c.map dec, collisionMass ν' ≤ cap
        ∧ ν' {q | Ls ≤ q.toList.length} = 0)
        ∧ ¬ ClusterSpec settings O P.D εcov slack (c.map dec) (clusterRun (c.map dec) (q.1, q.2.1))}
    | none => ∅
  set Z : Y → RoundDraws α K T → Finset (FreeMonoid α) := fun y d => match y with
    | some c => schedReads (settings.schedule (loopPopulations K (c.map dec).length))
        (loopPopulations K (c.map dec).length) (clusterRun (c.map dec) (ω₀, d.1))
    | none => ∅
  set Dj : Y → Set (Fin K → RoundDraws α K T) := fun y => match y with
    | some c => {x | NoCollD settings P.D Ls NS NP c x r hr}
    | none => Set.univ
  have hcl : ∀ c : List (HarvestCode α), Measurable (fun q : Ω × RoundDraws α K T =>
      clusterRun (c.map dec) ((q.1, q.2.1) : ClusterDraws Ω α K)) := fun c =>
    (measurePreserving_clusterRun (μ := μ) (K := K) (Measure.dirac (1 : FreeMonoid α))
      (Measure.dirac 1) _ (decodePop_prob settings P.D c)).measurable.comp
      (measurable_fst.prodMk (measurable_fst.comp measurable_snd))
  have hmain := persist_stage_le O ν (Measure.dirac ()) ν (fun d => ((() : Unit), d)) hσ Prod.snd
    measurable_snd (fun w => Prod.ext rfl rfl) ⟨r, hr⟩ (stageH settings O P.D N r)
    (fun ω x x' hxx' _ => stageH_congr_draws settings O P.D N r ω x x' fun i hi => hxx' i hi)
    (measurableSet_stageH settings O P.D N P.length_le hC hN r) W G
    (fun y x ω ω' hG hs h => by
      obtain ⟨c, rfl, -⟩ := hG
      exact stageH_stop settings O P.D N P.length_le hC hN r c x ω ω' ω₀ hs h)
    Bad (fun y hy => by
      rcases y with _ | c
      · rfl
      · refine Set.eq_empty_of_forall_notMem fun q hq => hy ⟨c, rfl, hq.1⟩)
    (fun y => by
      rcases y with _ | c
      · exact MeasurableSet.empty
      · exact (MeasurableSet.const _).inter ((MeasurableSet.const _).inter
          (hcl c (measurableSet_clusterSpec settings O P.D εcov slack (c.map dec))).compl))
    Z (fun y d ω ω' h hq => by
      rcases y with _ | c
      · exact hq
      · refine ⟨hq.1, hq.2.1, fun hs => hq.2.2 ?_⟩
        exact (clusterSpec_congr settings O P.D εcov slack (c.map dec) d.1 h).2 hs)
    Dj (fun y => by
      rcases y with _ | c
      · exact MeasurableSet.univ
      · exact measurableSet_noCollD settings P.D Ls NS NP c r hr)
    (fun y x hG hx => by
      obtain ⟨c, rfl, hcr⟩ := hG
      have hpop : popsAt settings P.D c r = c.map dec := by
        simp only [popsAt, List.take_of_length_le (le_of_eq hcr)]
        rfl
      have hnc := noColl_of_noCollD settings P.D (fun i => hNP _) hx
      have := disjoint_hist_round settings P.D N Ls NS c x r hr ω₀ (fun i => hNS _) hnc
      rw [hpop] at this
      exact this)
    (ENNReal.ofReal δc) (fun y => by
      rcases y with _ | c
      · show (μ.prod ν) ∅ ≤ _
        simp
      · by_cases hH : c.length = r ∧ ∀ ν' ∈ c.map dec, collisionMass ν' ≤ cap
            ∧ ν' {q | Ls ≤ q.toList.length} = 0
        swap
        · have : Bad (some c) = ∅ := Set.eq_empty_of_forall_notMem fun q hq =>
            hH ⟨hq.1, hq.2.1⟩
          rw [this, measure_empty]; exact zero_le
        set H := c.map dec
        have hpops := decodePop_prob settings P.D c
        have : ∀ j, IsProbabilityMeasure (loopDistributions (K := K) P.D H j) :=
          isProbabilityMeasure_loopDistributions P.D H hpops
        have hgetD : ∀ i : Fin K, some i ∈ loopPopulations K H.length →
            H.getD i (Measure.dirac 1) ∈ H := by
          intro i hi
          have := lt_of_some_mem_loopPops hi
          rw [List.getD_eq_getElem (hn := this)]
          exact List.getElem_mem _
        have hsupp : ∀ j ∈ loopPopulations K H.length,
            loopDistributions P.D H j {p | Ls ≤ p.toList.length} = 0 := by
          intro j hj
          cases j with
          | none =>
            refine measure_mono_null (fun x hx => ?_) (ae_iff.1 P.length_le)
            simp only [Set.mem_ofPred_eq, not_le] at hx ⊢
            omega
          | some i => exact (hH.2 _ (hgetD i hj)).2
        have hcolp : ∀ j ∈ loopPopulations K H.length,
            collisionMass (loopDistributions P.D H j) ≤ cap := by
          intro j hj
          cases j with
          | none => exact hcolD
          | some i => exact (hH.2 _ (hgetD i hj)).1
        have hc := hclust (loopPopulations K H.length) (Finset.mem_insert_self _ _)
          (loopDistributions P.D H) (isProbabilityMeasure_loopDistributions P.D H hpops) hsupp hcolp
        have hcr := measurePreserving_clusterRun (μ := μ) (K := K) P.D Dsf H hpops
        set Tset := {x : Run Ω (FreeMonoid α) (Option (Fin K)) |
          ¬ ClusterSpec settings O P.D εcov slack H x}
        have hTm : MeasurableSet Tset := (measurableSet_clusterSpec settings O P.D εcov slack H).compl
        have hmarg : MeasurePreserving (fun q : Ω × RoundDraws α K T =>
            ((q.1, q.2.1) : ClusterDraws Ω α K)) (μ.prod ν) (clusterLaw μ P.D Dsf K) :=
          (MeasurePreserving.id μ).prod measurePreserving_fst
        calc (μ.prod ν) (Bad (some c))
            ≤ (μ.prod ν) ((fun q : Ω × RoundDraws α K T => ((q.1, q.2.1) : ClusterDraws Ω α K)) ⁻¹'
                (clusterRun H ⁻¹' Tset)) := measure_mono fun q hq => hq.2.2
          _ = clusterLaw μ P.D Dsf K (clusterRun H ⁻¹' Tset) :=
              hmarg.measure_preimage (hcr.measurable hTm).nullMeasurableSet
          _ = runMeasure μ (loopDistributions P.D H) Dsf Tset :=
              hcr.measure_preimage hTm.nullMeasurableSet
          _ ≤ ENNReal.ofReal δc := by
              rw [← ofReal_measureReal]
              refine ENNReal.ofReal_le_ofReal ?_
              convert hc using 2
              ext x
              rfl)
  refine ENNReal.toReal_le_of_le_ofReal hδc ((measure_mono ?_).trans hmain)
  rintro p ⟨hok, hlen, hgood, hspec, hnc⟩
  have hst : stageH settings O P.D N r p.1 p.2 = some (histCodes settings O P.D N p.1 p.2 r) := by
    simp only [stageH, if_pos hok]
  show (p.1, p.2 ⟨r, hr⟩) ∈ Bad (stageH settings O P.D N r p.1 p.2)
    ∧ p.2 ∈ Dj (stageH settings O P.D N r p.1 p.2)
  rw [hst]
  exact ⟨⟨hlen, hgood, hspec⟩, hnc⟩

end Round

end OrthoDFA
