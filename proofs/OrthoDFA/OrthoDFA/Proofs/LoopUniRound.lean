import OrthoDFA.Proofs.LoopUniSpec

/-!
# A round's cut of the uniform pool, on one noise

Round `r`'s `UniSpec` is fixed by the answers at what its clustering reads and at the uniform
pool's strings, those no earlier round's clustering read, followed by nothing or by the round's
suffixes (`uniReads`). Off a collision, and where the earlier rounds' data hold the prefixes they
read, no earlier round read any of these. So given the earlier rounds the spec fails with at most
the chance it has on fresh noise (`loop_uni_round_le_persist`), whatever the earlier rounds'
populations are.
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α] {Ω : Type*} [MeasurableSpace Ω]
  {μ : Measure Ω} {K : ℕ}

variable [IsProbabilityMeasure μ]

lemma clusterAt_sub_roundSuffs (O : Oracle μ (FreeMonoid α)) {sched : Finset State}
    (pops : Finset (Option (Fin K))) {B : State} (hB : B ∈ sched)
    (x : Run Ω (FreeMonoid α) (Option (Fin K))) :
    familyAt O.mq pops x B ⊆ insert 1 (roundSuffs sched x) := by
  intro v hv
  rcases Finset.mem_insert.1 hv with h | hv
  · exact h ▸ Finset.mem_insert_self _ _
  rw [clusterAt_eq, clusterAt_eq_clusterOf] at hv
  exact poolAt_sub_roundSuffs hB _ (clusterOf_subset O B.sc B.scd _ _ B.k x.1
    (Finset.mem_insert_self _ _) hv)

omit [DecidableEq α] [IsProbabilityMeasure μ] in
lemma measureReal_congr_on {Du : Measure (FreeMonoid α)} {T : Set (FreeMonoid α)}
    (hT : Du Tᶜ = 0) {A A' : Set (FreeMonoid α)} (h : ∀ p ∈ T, (p ∈ A ↔ p ∈ A')) :
    Du.real A = Du.real A' := by
  refine congrArg ENNReal.toReal (measure_congr ?_)
  refine (ae_iff.2 hT).mono fun p hp => ?_
  exact propext (h p hp)

/-- The uniform pool's realized cut, at two noises the reads of the clustering and of the pool's
strings `T` followed by the round's suffixes cannot tell apart. -/
theorem uniSpec_congr (O : Oracle μ (FreeMonoid α)) (sched : Finset State)
    (pops : Finset (Option (Fin K))) (hnone : none ∈ pops) (η₀ ι εc a : ℝ)
    (Du : Measure (FreeMonoid α)) (T : Finset (FreeMonoid α)) (hT : Du (↑T)ᶜ = 0) {ω ω' : Ω}
    (dr : ((ℕ → FreeMonoid α) × (Option (Fin K) → ℕ → FreeMonoid α))
      × (Option (Fin K) → ℕ → FreeMonoid α))
    (h1 : ∀ w ∈ schedReads sched pops ((ω, dr) : Run Ω (FreeMonoid α) (Option (Fin K))),
      (O.mq w ω = 1 ↔ O.mq w ω' = 1))
    (h2 : ∀ w ∈ readSet T (insert 1 (roundSuffs sched
      ((ω, dr) : Run Ω (FreeMonoid α) (Option (Fin K))))), (O.mq w ω = 1 ↔ O.mq w ω' = 1))
    (hs : UniSpec O pops sched η₀ ι εc a Du ((ω, dr) : Run Ω (FreeMonoid α) (Option (Fin K)))) :
    UniSpec O pops sched η₀ ι εc a Du ((ω', dr) : Run Ω (FreeMonoid α) (Option (Fin K))) := by
  classical
  have hret : ∀ B ∈ sched, (((ω, dr) : Run Ω (FreeMonoid α) (Option (Fin K)))
      ∈ ret O.mq pops none η₀ ι εc a B ↔ ((ω', dr) : Run Ω (FreeMonoid α) (Option (Fin K)))
        ∈ ret O.mq pops none η₀ ι εc a B) := fun B hB =>
    ret_congr O sched pops none hnone _ _ _ _ hB dr h1
  have hfam : ∀ B ∈ sched, familyAt O.mq pops ((ω, dr) : Run Ω (FreeMonoid α) (Option (Fin K))) B
      = familyAt O.mq pops ((ω', dr) : Run Ω (FreeMonoid α) (Option (Fin K))) B := fun B hB => by
    unfold familyAt
    exact congrArg (insert 1) (clusterAt_congr_sched O sched pops hB dr h1)
  have hvote : ∀ B ∈ sched, ∀ p ∈ T,
      voteCount O.mq (familyAt O.mq pops ((ω, dr) : Run Ω (FreeMonoid α) (Option (Fin K))) B) p ω
        = voteCount O.mq (familyAt O.mq pops ((ω, dr) : Run Ω (FreeMonoid α) (Option (Fin K))) B)
          p ω' := fun B hB p hp =>
    voteCount_alike O fun v hv =>
      h2 _ (mem_readSet hp (clusterAt_sub_roundSuffs O pops hB _ hv))
  obtain ⟨⟨B₀, hB₀, hr₀⟩, hall⟩ := hs
  refine ⟨⟨B₀, hB₀, (hret B₀ hB₀).1 hr₀⟩, fun B hB hr => ?_⟩
  obtain ⟨hc, hu⟩ := hall B hB ((hret B hB).2 hr)
  have e1 : Du.real {p | ¬ cutCorrect O B.lo (B.hi + 1)
        (familyAt O.mq pops ((ω, dr) : Run Ω (FreeMonoid α) (Option (Fin K))) B) p ω}
      = Du.real {p | ¬ cutCorrect O B.lo (B.hi + 1)
        (familyAt O.mq pops ((ω', dr) : Run Ω (FreeMonoid α) (Option (Fin K))) B) p ω'} := by
    refine measureReal_congr_on hT fun p hp => ?_
    simp only [Set.mem_ofPred_eq, cutCorrect, ← hfam B hB, hvote B hB p hp]
  have e2 : Du.real {p | ¬ decided O.mq B.lo (B.hi + 1)
        (familyAt O.mq pops ((ω, dr) : Run Ω (FreeMonoid α) (Option (Fin K))) B) p ω}
      = Du.real {p | ¬ decided O.mq B.lo (B.hi + 1)
        (familyAt O.mq pops ((ω', dr) : Run Ω (FreeMonoid α) (Option (Fin K))) B) p ω'} := by
    refine measureReal_congr_on hT fun p hp => ?_
    simp only [Set.mem_ofPred_eq, decided, ← hfam B hB, hvote B hB p hp]
  exact ⟨e1 ▸ hc, e2 ▸ hu⟩

omit [IsProbabilityMeasure μ] in
lemma offLaw_long_null {ν : Measure (FreeMonoid α)} [IsFiniteMeasure ν] (T : Finset (FreeMonoid α))
    {Ls : ℕ} (hLs : 0 < Ls) (hν : ν {p | Ls ≤ p.toList.length} = 0) :
    offLaw ν T {p | Ls ≤ p.toList.length} = 0 := by
  by_cases h0 : ν (↑T)ᶜ = 0
  · unfold offLaw popLaw
    rw [if_pos (Or.inl (by rw [Measure.restrict_apply_univ]; exact h0))]
    rw [Measure.dirac_apply' _ (measurableSet_of_countable _)]
    simp only [Set.indicator, Set.mem_ofPred_eq, FreeMonoid.toList_one, List.length_nil]
    split_ifs with hc
    · omega
    · exact if_neg hc
  · rw [offLaw_eq h0, Measure.smul_apply, Measure.restrict_apply (measurableSet_of_countable _),
      measure_mono_null Set.inter_subset_left hν, smul_zero]

omit [DecidableEq α] [IsProbabilityMeasure μ] in
/-- Off its prefixes, a law supported on strings at most `L` long is supported on the strings at
most `L` long that are none of them. -/
lemma offLaw_short_null {ν : Measure (FreeMonoid α)} [IsFiniteMeasure ν] (T : Finset (FreeMonoid α))
    {L : ℕ} (hlen : ∀ᵐ x ∂ν, x.toList.length ≤ L) (h0 : ν (↑T)ᶜ ≠ 0) :
    offLaw ν T (↑(shortStrings L \ T))ᶜ = 0 := by
  classical
  rw [offLaw_eq h0, Measure.smul_apply, Measure.restrict_apply (measurableSet_of_countable _)]
  have : ν ((↑(shortStrings L \ T))ᶜ ∩ (↑T)ᶜ) = 0 := by
    refine measure_mono_null (fun x hx => ?_) (ae_iff.1 hlen)
    simp only [Set.mem_inter_iff, Set.mem_compl_iff, Finset.coe_sdiff, Set.mem_sdiff,
      Finset.mem_coe, not_and, not_not] at hx
    simp only [Set.mem_ofPred_eq, not_le]
    by_contra hc
    exact hx.2 (hx.1 (mem_shortStrings.2 (by omega)))
  rw [this, smul_zero]

/-- `uniSpec_congr` at a round's clustering run. -/
theorem uniSpec_congr_run (O : Oracle μ (FreeMonoid α)) (sched : Finset State)
    (pops : Finset (Option (Fin K))) (hnone : none ∈ pops) (η₀ ι εc a : ℝ)
    (Du : Measure (FreeMonoid α)) (T : Finset (FreeMonoid α)) (hT : Du (↑T)ᶜ = 0)
    (D Dsf : Measure (FreeMonoid α)) (h : LoopHist α) {ω ω' : Ω} (c : ClusterRest α K)
    (h1 : ∀ w ∈ schedReads sched pops (clusterRun D Dsf h (ω, c)),
      (O.mq w ω = 1 ↔ O.mq w ω' = 1))
    (h2 : ∀ w ∈ readSet T (insert 1 (roundSuffs sched (clusterRun D Dsf h (ω, c)))),
      (O.mq w ω = 1 ↔ O.mq w ω' = 1))
    (hs : UniSpec O pops sched η₀ ι εc a Du (clusterRun D Dsf h (ω, c))) :
    UniSpec O pops sched η₀ ι εc a Du (clusterRun D Dsf h (ω', c)) :=
  uniSpec_congr O sched pops hnone η₀ ι εc a Du T hT (runOf D Dsf h c) h1 h2 hs

section Round

variable (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α)) (P : PrefixDistribution α)
  (Dsf : Measure (FreeMonoid α)) [IsProbabilityMeasure Dsf]

omit [IsProbabilityMeasure μ] [IsProbabilityMeasure Dsf] in
lemma measurableSet_prefsOf_sub (h : LoopHist α) (T : Finset (FreeMonoid α)) :
    MeasurableSet {c : ClusterRest α K | prefsOf settings P.D Dsf h c ⊆ T} := by
  classical
  have : {c : ClusterRest α K | prefsOf settings P.D Dsf h c ⊆ T}
      = ⋃ Q ∈ T.powerset, {c | prefsOf settings P.D Dsf h c = Q} := by
    ext c
    simp only [Set.mem_ofPred_eq, Set.mem_iUnion, Finset.mem_powerset, exists_prop]
    exact ⟨fun h => ⟨_, h, rfl⟩, fun ⟨Q, hQ, e⟩ => e ▸ hQ⟩
  rw [this]
  exact Finset.measurableSet_biUnion _ fun Q _ => measurableSet_prefsOf settings P.D Dsf h Q

omit [IsProbabilityMeasure Dsf] in
/-- Off a collision, and where the earlier rounds' prefixes are among the data's, no earlier round
read the uniform pool's strings off those prefixes, alone or followed by round `r`'s suffixes. -/
theorem disjoint_hist_uni (N Ls : ℕ) (hLs : P.L < Ls) (c : List (HarvestCode α))
    (x : Fin K → RoundDraws α K settings.level3.probeBudget) (r : ℕ) (hr : r < K) (ω : Ω)
    (hnc : NoColl settings P.D Dsf Ls c x r hr)
    (hpre : ∀ i : Fin K, i.1 < r → prefsOf settings P.D Dsf (histAt settings P.D c i) (x i).1
      ⊆ (decodeHist P.D settings.level3.walkStart c).prefs) :
    Disjoint (histReads settings P.D Dsf N c x r ω)
      (readSet (shortStrings P.L \ (decodeHist P.D settings.level3.walkStart c).prefs)
        (insert 1 (roundSuffs (settings.schedule (loopPopulations K (histAt settings P.D c r).pops.length))
          (((), runOf P.D Dsf (histAt settings P.D c r) (x ⟨r, hr⟩).1) :
            Run Unit (FreeMonoid α) (Option (Fin K)))))) := by
  classical
  obtain ⟨hsl, hpl, hsd, -⟩ := hnc
  refine Finset.disjoint_left.2 fun w hw hw' => ?_
  obtain ⟨i, hir, hsh⟩ := mem_histReads_shape settings P.D Dsf N c x r ω hw
  obtain ⟨⟨p, v⟩, hpv, rfl⟩ := Finset.mem_image.1 hw'
  obtain ⟨hp, hv⟩ := Finset.mem_product.1 hpv
  obtain ⟨hpL, hpT⟩ := Finset.mem_sdiff.1 hp
  have hpl' : p.toList.length ≤ P.L := mem_shortStrings.1 hpL
  dsimp only at hsh
  by_cases hv1 : v = 1
  · subst hv1
    rw [mul_one] at hsh
    rcases hsh with ⟨p', hp', e⟩ | ⟨z, v', hv', e⟩
    · exact hpT (e ▸ hpre i hir hp')
    · have := congrArg (fun t : FreeMonoid α => t.toList.length) e
      simp only [length_mul] at this
      have := hsl i hir.le v' hv'
      omega
  · have hvs : v ∈ suffsOf settings P.D Dsf (histAt settings P.D c r) (x ⟨r, hr⟩).1 :=
      Finset.mem_erase.2 ⟨hv1, (Finset.mem_insert.1 hv).resolve_left hv1⟩
    have hvl := hsl ⟨r, hr⟩ le_rfl v hvs
    rcases hsh with ⟨p', hp', e⟩ | ⟨z, v', hv', e⟩
    · have := congrArg (fun t : FreeMonoid α => t.toList.length) e
      simp only [length_mul] at this
      have := hpl i hir.le p' hp'
      omega
    · have hvl' := hsl i hir.le v' hv'
      obtain ⟨hvv, -⟩ := suffix_eq_of_mul_eq e.symm (hvl'.trans hvl.symm)
      exact Finset.disjoint_left.1 (hsd i hir) hv' (hvv ▸ hvs)

/-- Round `r`'s clustering, where every earlier clustering returned, the earlier rounds read at
most `m` of the uniform pool and `ms` of the suffixes, their prefixes are among the data's and
nothing collides, misses its cut of the uniform pool at the run's noise with chance at most `δc`,
whatever the earlier populations hold. -/
theorem loop_uni_round_le_persist (Ls N : ℕ) (δc capU capSf m ms pAP : ℝ)
    (hδc : 0 ≤ δc) (hN : P.L + settings.level3.probeBudget + 2 ≤ N)
    (hm : m ≤ 1 / 2) (hms : ms ≤ 1 / 2)
    (huni : ∀ pops : Finset (Option (Fin K)), none ∈ pops →
      ∀ (L : Option (Fin K) → Measure (FreeMonoid α)) (Lsf : Measure (FreeMonoid α)),
        (∀ j, IsProbabilityMeasure (L j)) → IsProbabilityMeasure Lsf →
        (∀ j ∈ pops, L j {p | Ls ≤ p.toList.length} = 0) →
        (∀ t, (L none).real {t} ≤ 2 * capU) →
        Lsf {v | v.toList.length ≠ Ls} = 0 →
        pAP - ms ≤ Lsf.real {v | ∀ p, p * v ∈ O.L ↔ p ∈ O.L} →
        (∀ t, Lsf.real {t} ≤ 2 * capSf) →
        (runMeasure μ L Lsf).real
          {x | ¬ UniSpec O pops (settings.schedule pops) settings.η₀ settings.indecisionLimit
            settings.εcov settings.gateLevel (L none) (streams (fun _ => ∅) none x)} ≤ δc)
    (hLs : P.L < Ls) (hDat : ∀ t, P.D.real {t} ≤ capU) (hDsf : Dsf {v | v.toList.length ≠ Ls} = 0)
    (hpAP : pAP ≤ Dsf.real {v | ∀ p, p * v ∈ O.L ↔ p ∈ O.L}) (hsfat : ∀ t, Dsf.real {t} ≤ capSf)
    (r : ℕ) (hr : r < K) :
    (loopMeasure μ α K settings.level3.probeBudget).real
      {p | HistOK settings O P.D Dsf N p.1 p.2 r
        ∧ (histCodes settings O P.D Dsf N p.1 p.2 r).length = r
        ∧ (∀ ν' ∈ (decodeHist P.D settings.level3.walkStart
            (histCodes settings O P.D Dsf N p.1 p.2 r)).pops, ν' {q | Ls ≤ q.toList.length} = 0)
        ∧ P.D.real ↑(decodeHist P.D settings.level3.walkStart
            (histCodes settings O P.D Dsf N p.1 p.2 r)).prefs ≤ m
        ∧ Dsf.real ↑(decodeHist P.D settings.level3.walkStart
            (histCodes settings O P.D Dsf N p.1 p.2 r)).suffs ≤ ms
        ∧ ¬ UniSpec O (loopPopulations K r) (settings.schedule (loopPopulations K r))
            settings.η₀ settings.indecisionLimit settings.εcov settings.gateLevel
            (offLaw P.D (decodeHist P.D settings.level3.walkStart
              (histCodes settings O P.D Dsf N p.1 p.2 r)).prefs)
            (clusterRun P.D Dsf (decodeHist P.D settings.level3.walkStart
              (histCodes settings O P.D Dsf N p.1 p.2 r)) (p.1, (p.2 ⟨r, hr⟩).1))
        ∧ NoColl settings P.D Dsf Ls (histCodes settings O P.D Dsf N p.1 p.2 r) p.2 r hr
        ∧ ∀ i : Fin K, i.1 < r → prefsOf settings P.D Dsf
            (histAt settings P.D (histCodes settings O P.D Dsf N p.1 p.2 r) i) (p.2 i).1
          ⊆ (decodeHist P.D settings.level3.walkStart
            (histCodes settings O P.D Dsf N p.1 p.2 r)).prefs}
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
    ∧ (∀ ν' ∈ (dec c).pops, ν' {q | Ls ≤ q.toList.length} = 0)
    ∧ P.D.real ↑(dec c).prefs ≤ m
    ∧ Dsf.real ↑(dec c).suffs ≤ ms
  set Spec : List (HarvestCode α) → Run Ω (FreeMonoid α) (Option (Fin K)) → Prop := fun c x =>
    UniSpec O (loopPopulations K (dec c).pops.length)
      (settings.schedule (loopPopulations K (dec c).pops.length))
      settings.η₀ settings.indecisionLimit settings.εcov settings.gateLevel
      (offLaw P.D (dec c).prefs) x
  set Bad : Y → Set (Ω × RoundDraws α K T) := fun y => match y with
    | some c => {q | Good c ∧ ¬ Spec c (clusterRun P.D Dsf (dec c) (q.1, q.2.1))}
    | none => ∅
  set Z : Y → RoundDraws α K T → Finset (FreeMonoid α) := fun y d => match y with
    | some c => schedReads (settings.schedule (loopPopulations K (dec c).pops.length))
        (loopPopulations K (dec c).pops.length) (clusterRun P.D Dsf (dec c) (ω₀, d.1))
      ∪ readSet (shortStrings P.L \ (dec c).prefs)
        (insert 1 (roundSuffs (settings.schedule (loopPopulations K (dec c).pops.length))
          (clusterRun P.D Dsf (dec c) (ω₀, d.1))))
    | none => ∅
  set Dj : Y → Set (Fin K → RoundDraws α K T) := fun y => match y with
    | some c => {x | NoColl settings P.D Dsf Ls c x r hr
        ∧ ∀ i : Fin K, i.1 < r → prefsOf settings P.D Dsf (histAt settings P.D c i) (x i).1
          ⊆ (dec c).prefs}
    | none => Set.univ
  have hcl : ∀ c : List (HarvestCode α), Measurable (fun q : Ω × RoundDraws α K T =>
      clusterRun P.D Dsf (dec c) ((q.1, q.2.1) : ClusterDraws Ω α K)) := fun c =>
    (measurable_clusterRun P.D Dsf (dec c)).comp
      (measurable_fst.prodMk (measurable_fst.comp measurable_snd))
  have hlenc : ∀ c : List (HarvestCode α), Good c → (dec c).pops.length = r := fun c hc => by
    simp only [dec, decodeHist, List.length_map, hc.1]
  have hhist : ∀ c : List (HarvestCode α), c.length = r → histAt settings P.D c r = dec c :=
    fun c hcr => by simp only [histAt, List.take_of_length_le (le_of_eq hcr), hdec]
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
          (hcl c (measurableSet_uniSpec O _ _ _ _ _ _ (offLaw P.D (dec c).prefs))).compl)
    Z (fun y d ω ω' h hq => by
      rcases y with _ | c
      · exact hq
      · refine ⟨hq.1, fun hs => hq.2 ?_⟩
        have hc := hq.1
        have h0 : P.D (↑(dec c).prefs)ᶜ ≠ 0 := compl_ne_zero (hc.2.2.1.trans hm)
        have hZ1 : ∀ w ∈ schedReads (settings.schedule (loopPopulations K (dec c).pops.length))
            (loopPopulations K (dec c).pops.length) (clusterRun P.D Dsf (dec c) (ω', d.1)),
            (O.mq w ω' = 1 ↔ O.mq w ω = 1) := fun w hw => (h w (Finset.mem_union_left _ hw)).symm
        have hZ2 : ∀ w ∈ readSet (shortStrings P.L \ (dec c).prefs)
            (insert 1 (roundSuffs (settings.schedule (loopPopulations K (dec c).pops.length))
              (clusterRun P.D Dsf (dec c) (ω', d.1)))),
            (O.mq w ω' = 1 ↔ O.mq w ω = 1) := fun w hw => (h w (Finset.mem_union_right _ hw)).symm
        exact uniSpec_congr_run O _ _ (Finset.mem_insert_self _ _) _ _ _ _ _
          (shortStrings P.L \ (dec c).prefs) (offLaw_short_null _ P.length_le h0) P.D Dsf (dec c)
          d.1 hZ1 hZ2 hs)
    Dj (fun y => by
      rcases y with _ | c
      · exact MeasurableSet.univ
      · refine (measurableSet_noColl settings P.D Dsf Ls c r hr).inter ?_
        have hm' : MeasurableSet (⋂ i : Fin K, {x : Fin K → RoundDraws α K T | i.1 < r →
            prefsOf settings P.D Dsf (histAt settings P.D c i) (x i).1 ⊆ (dec c).prefs}) := by
          refine MeasurableSet.iInter fun i => ?_
          by_cases hi : i.1 < r
          · simp only [hi, true_implies]
            exact (measurable_fst.comp (measurable_pi_apply i))
              (measurableSet_prefsOf_sub settings P Dsf _ _)
          · simp [hi]
        have e : (⋂ i : Fin K, {x : Fin K → RoundDraws α K T | i.1 < r →
            prefsOf settings P.D Dsf (histAt settings P.D c i) (x i).1 ⊆ (dec c).prefs})
            = {x | ∀ i : Fin K, i.1 < r →
              prefsOf settings P.D Dsf (histAt settings P.D c i) (x i).1 ⊆ (dec c).prefs} := by
          ext x; simp
        rw [e] at hm'
        exact hm')
    (fun y x hG hx => by
      obtain ⟨c, rfl, hcr⟩ := hG
      have e := hhist c hcr
      refine Finset.disjoint_union_right.2 ⟨?_, ?_⟩
      · have := disjoint_hist_round settings P.D Dsf N Ls c x r hr ω₀ hx.1
        rw [e] at this
        exact this
      · have := disjoint_hist_uni settings P Dsf N Ls hLs c x r hr ω₀ hx.1 hx.2
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
        have hsupp : ∀ j ∈ pops,
            loopDistributions P.D h.pops j {p | Ls ≤ p.toList.length} = 0 := by
          intro j hj
          cases j with
          | none =>
            refine measure_mono_null (fun x hx => ?_) (ae_iff.1 P.length_le)
            simp only [Set.mem_ofPred_eq, not_le] at hx ⊢
            omega
          | some i => exact hpg _ (hgetD i hj)
        set L : Option (Fin K) → Measure (FreeMonoid α) := fun j => popOff P.D h.pops j h.prefs
        have hLn : L none = offLaw P.D h.prefs := popOff_of_ne (compl_ne_zero (hpm.trans hm))
        have hsm' : Dsf.real ↑h.suffs ≤ 1 / 2 := hsm.trans hms
        have hc := huni pops (Finset.mem_insert_self _ _) L (offLaw Dsf h.suffs)
          (fun j => inferInstance) inferInstance
          (fun j hj => by
            show popOff P.D h.pops j h.prefs _ = 0
            unfold popOff
            split_ifs
            · exact offLaw_long_null _ (by omega) (hsupp none (Finset.mem_insert_self _ _))
            · exact offLaw_long_null _ (by omega) (hsupp j hj))
          (fun t => by
            rw [hLn]
            exact (offLaw_atom_le (hpm.trans hm) t).trans (by linarith [hDat t]))
          (offLaw_null (compl_ne_zero hsm') hDsf)
          (by linarith [le_offLaw_real hsm' {v : FreeMonoid α | ∀ p, p * v ∈ O.L ↔ p ∈ O.L}])
          (fun t => (offLaw_atom_le hsm' t).trans (by linarith [hsfat t]))
        have hcr' := measurePreserving_clusterDraws (μ := μ) (K := K) P.D Dsf h
        set Tset := {x : Run Ω (FreeMonoid α) (Option (Fin K)) | ¬ Spec c x}
        have hTm : MeasurableSet (streams (Ω := Ω) (fun _ => ∅) none ⁻¹' Tset) :=
          measurable_streams _ none (measurableSet_uniSpec O _ _ _ _ _ _ _).compl
        have hmarg : MeasurePreserving (fun q : Ω × RoundDraws α K T =>
            ((q.1, q.2.1) : ClusterDraws Ω α K)) (μ.prod ν) (clusterLaw μ α K) :=
          (MeasurePreserving.id μ).prod measurePreserving_fst
        calc (μ.prod ν) (Bad (some c))
            ≤ (μ.prod ν) ((fun q : Ω × RoundDraws α K T => ((q.1, q.2.1) : ClusterDraws Ω α K)) ⁻¹'
                (clusterDraws P.D Dsf h ⁻¹' (streams (fun _ => ∅) none ⁻¹' Tset))) :=
              measure_mono fun q hq => hq.2
          _ = clusterLaw μ α K (clusterDraws P.D Dsf h ⁻¹' (streams (fun _ => ∅) none ⁻¹' Tset)) :=
              hmarg.measure_preimage (hcr'.measurable hTm).nullMeasurableSet
          _ = runMeasure μ L (offLaw Dsf h.suffs) (streams (fun _ => ∅) none ⁻¹' Tset) :=
              hcr'.measure_preimage hTm.nullMeasurableSet
          _ ≤ ENNReal.ofReal δc := by
              rw [← ofReal_measureReal]
              rw [hLn] at hc
              exact ENNReal.ofReal_le_ofReal hc)
  refine ENNReal.toReal_le_of_le_ofReal hδc ((measure_mono ?_).trans hmain)
  rintro p ⟨hok, hlen, hpg, hpm, hsm, hspec, hnc, hpre⟩
  have hst : stageH settings O P.D Dsf N r p.1 p.2
      = some (histCodes settings O P.D Dsf N p.1 p.2 r) := by
    simp only [stageH, if_pos hok]
  have hpl : (dec (histCodes settings O P.D Dsf N p.1 p.2 r)).pops.length = r := by
    simp only [dec, decodeHist, List.length_map, hlen]
  show (p.1, p.2 ⟨r, hr⟩) ∈ Bad (stageH settings O P.D Dsf N r p.1 p.2)
    ∧ p.2 ∈ Dj (stageH settings O P.D Dsf N r p.1 p.2)
  rw [hst]
  refine ⟨⟨⟨hlen, hpg, hpm, hsm⟩, ?_⟩, hnc, hpre⟩
  show ¬ UniSpec O (loopPopulations K (dec _).pops.length) _ _ _ _ _ _ _
  rw [hpl]
  exact hspec

end Round

end OrthoDFA

end
